"""Real pi0.5 work; no device-global synchronization and no global RNG reseeding."""
import os,sys,json,time,hashlib,csv,threading,contextlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/hengyi/Workspace/01_VLA-Serving/Environment-and-Earlier-Runs/vla_followup_20261020')
OLD=Path('/home/hengyi/Workspace/01_VLA-Serving/Environment-and-Earlier-Runs/vla_ladder_20261020')
GPU=int(os.environ.get('VLA_GPU','0'))
os.environ.update(CUDA_VISIBLE_DEVICES=os.environ["PI_MPS_UUID"],MUJOCO_EGL_DEVICE_ID=str(GPU),TOKENIZER_DIR=str(SOURCE/'assets/.cache/openpi'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',JAX_PLATFORMS='cpu')
os.sched_setaffinity(0,set(range(GPU*8,GPU*8+8)))
sys.path.insert(0,str(SOURCE))
import numpy as np
import torch
from evaluate import load_model
from train import configure,Samples,fingerprint
from rlinf.models.embodiment.openpi.modules import model as model_module
torch.set_num_threads(4)

def save(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def write_csv(path,rows,fields=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if not fields:fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def stamp():return time.monotonic_ns()
def finish():
 e=torch.cuda.Event();e.record();e.synchronize()
def timed(name,fn):
 start=stamp();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
 a.record()
 with torch.cuda.nvtx.range(name):out=fn()
 b.record();b.synchronize()
 end=stamp()
 return out,{'cpu_submit_ns':start,'gpu_done_ns':end,'wall_ms':(end-start)/1e6,'cuda_ms':a.elapsed_time(b),'memory_allocated':torch.cuda.memory_allocated(),'memory_reserved':torch.cuda.memory_reserved()}
def stats(values):
 return {k:float(v) for k,v in zip(['mean','p50','p95','p99','max'],[np.mean(values),*np.percentile(values,[50,95,99]),max(values)])}

class Monitor:
 def __enter__(self):
  import pynvml as nv
  self.nv=nv;nv.nvmlInit();self.h=nv.nvmlDeviceGetHandleByIndex(GPU);self.rows=[];self.stop=threading.Event()
  def sample():
   while not self.stop.is_set():
    self.rows.append({'ns':stamp(),'device_bytes':nv.nvmlDeviceGetMemoryInfo(self.h).used,
      'process_bytes':sum(x.usedGpuMemory for x in nv.nvmlDeviceGetComputeRunningProcesses(self.h) if x.pid==os.getpid())})
    self.stop.wait(.1)
  self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
 def __exit__(self,*args):self.stop.set();self.thread.join();self.nv.nvmlShutdown()

class Serving:
 def __init__(self):
  self.model=load_model().eval()
  for p in self.model.parameters():p.requires_grad_(False)
  self.initial=fingerprint(self.model.named_parameters())
  with np.load(SOURCE/'benchmark/inputs.npz',allow_pickle=False) as z:self.corpus={k:z[k] for k in z.files}
  self.stream=torch.cuda.Stream()
  self.reference=None
 def input(self,i):
  j=i%len(self.corpus['states']);c=self.corpus
  obs=self.model.env_obs_to_observation({'main_images':c['images'][j:j+1], 'wrist_images':c['wrists'][j:j+1], 'states':c['states'][j:j+1], 'task_descriptions':[str(c['prompts'][j])]})
  generator=torch.Generator(device='cuda').manual_seed(71000+i)
  noise=torch.randn((1,self.model.action_horizon,self.model.action_dim),device='cuda',generator=generator)
  return obs,noise
 def prefix(self,obs):
  obs=model_module.preprocess_observation(obs,train=False)
  _,mask,kv=self.model.build_prefix_cache(obs)
  return obs,mask,kv
 def action(self,prepared,noise):
  obs,mask,kv=prepared;x=noise;t=1.;dt=-1./self.model.num_steps
  while t>=-dt/2:
   tt=torch.full((obs.state.shape[0],),t,device=obs.state.device,dtype=torch.float32)
   out=self.model.run_suffix(obs,x,tt,kv,mask)
   x=x+dt*self.model.velocity_from_suffix(out);t+=dt
  return x
 def infer(self,i):
  with torch.cuda.stream(self.stream),torch.inference_mode():
   start=stamp();obs,noise=self.input(i)
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);c=torch.cuda.Event(enable_timing=True);d=torch.cuda.Event(enable_timing=True)
   a.record()
   with torch.cuda.nvtx.range('I_PREFIX'):prepared=self.prefix(obs)
   b.record()
   with torch.cuda.nvtx.range('I_ACTION'):raw=self.action(prepared,noise)
   c.record()
   with torch.cuda.nvtx.range('I_POST'):out=self.model.decode_actions(raw,obs.state).cpu()
   d.record();d.synchronize();end=stamp()
   assert torch.isfinite(out).all()
   return out,{'start_ns':start,'finish_ns':end,'I_PREFIX_ms':a.elapsed_time(b),'I_ACTION_ms':b.elapsed_time(c),'I_POST_ms':c.elapsed_time(d),'model_ms':a.elapsed_time(d),'service_ms':(end-start)/1e6}
 def verify(self):
  records=[]
  for i in [0,1,17,63]:
   with torch.cuda.stream(self.stream),torch.inference_mode():
    obs,noise=self.input(i)
    original=self.model.sample_actions(obs,num_steps=self.model.num_steps,noise=noise)
    split=self.action(self.prefix(obs),noise)
    finish();difference=(original-split).abs().max().item()
    assert torch.allclose(original,split,atol=1e-6,rtol=1e-6),difference
    records.append({'input':i,'max_abs':difference})
  self.reference=self.infer(0)[0]
  return records
 def unchanged(self,full=False):
  output=self.infer(0)[0];diff=(output-self.reference).abs().max().item()
  assert torch.equal(output,self.reference),diff
  if full:assert fingerprint(self.model.named_parameters())==self.initial
  return {'probe_max_abs':diff,'full_hash_checked':full,'unchanged':True}

class Training:
 def __init__(self,tier,cache_steps=16):
  self.tier=tier;self.model=load_model();self.pairs=configure(self.model,tier)
  self.cfg=json.loads((SOURCE/'experiment.json').read_text())
  self.masters=[torch.nn.Parameter(p.detach().float().clone()) for _,p in self.pairs]
  self.optimizer=torch.optim.AdamW(self.masters,lr=self.cfg['lr'],betas=(.9,.95),eps=1e-8,weight_decay=self.cfg['weight_decay'],foreach=False)
  self.saved=torch.load(OLD/f'training/{tier}/resume.pt',map_location='cpu',weights_only=True,mmap=True)
  assert self.saved['step']==1000 and self.saved['master_names']==[n for n,p in self.pairs]
  self.frozen=fingerprint((n,p) for n,p in self.model.named_parameters() if not p.requires_grad)
  self.stream=torch.cuda.Stream();self.cache=[]
  samples=Samples(self.model);batches=json.loads((OLD/'sample_sequence.json').read_text())[:cache_steps]
  # Cache transformed tensors in host memory, preserving the original preprocessing.
  from torch.utils._pytree import tree_map
  import dataclasses
  for batch in batches:
   for episode,index in batch:
    obs,actions=samples.get(episode,index)
    cpu_obs=dataclasses.replace(obs,**{f.name:tree_map(lambda x:x.cpu() if torch.is_tensor(x) else x,getattr(obs,f.name)) for f in dataclasses.fields(obs)})
    self.cache.append((cpu_obs,actions.cpu()))
  del samples
  finish();self.reset()
 def reset(self):
  finish();self.loss=None
  self.model.load_state_dict(self.saved['parameters'],strict=False)
  with torch.no_grad():
   for m,v in zip(self.masters,self.saved['masters']):m.copy_(v)
  state={**self.saved['optimizer'],'state':{k:{n:v.clone() if n=='step' else v for n,v in vals.items()} for k,vals in self.saved['optimizer']['state'].items()}}
  self.optimizer.load_state_dict(state);self.optimizer.zero_grad(set_to_none=True);self.model.zero_grad(set_to_none=True)
  self.model.train();self.iteration=0;self.micro=0;self.next_kind='T_FWD'
  finish();self.stream.wait_stream(torch.cuda.current_stream())
 def forward(self):
  from torch.utils._pytree import tree_map
  import dataclasses
  index=self.iteration*8+self.micro;obs,actions=self.cache[index%len(self.cache)]
  obs=dataclasses.replace(obs,**{f.name:tree_map(lambda x:x.to('cuda') if torch.is_tensor(x) else x,getattr(obs,f.name)) for f in dataclasses.fields(obs)})
  actions=actions.to('cuda');seed=self.cfg['seed']+index
  # CPU generator feeds augmentation, separate generator feeds flow noise, numpy feeds Beta(1.5,1).
  aug=torch.Generator().manual_seed(seed)
  noise=torch.randn(actions.shape,generator=torch.Generator().manual_seed(seed+100000)).to('cuda')
  t=torch.tensor(np.random.default_rng(seed+200000).beta(1.5,1.,size=1)*.999+.001,device='cuda',dtype=torch.float32)
  self.loss=self.model.compute_loss(obs,actions,train=True,rng=aug,noise=noise,time=t).mean()
  self.next_kind='T_BWD';return self.loss
 def backward(self):
  (self.loss/8).backward();self.loss=None;self.micro+=1
  self.next_kind='T_OPT' if self.micro==8 else 'T_FWD'
 def opt(self):
  for (_,p),master in zip(self.pairs,self.masters):
   assert p.grad is not None
   master.grad=p.grad.detach().float()
  torch.nn.utils.clip_grad_norm_(self.masters,self.cfg['clip_grad'],error_if_nonfinite=True)
  self.optimizer.step()
  with torch.no_grad():
   for (_,p),master in zip(self.pairs,self.masters):p.copy_(master)
  self.optimizer.zero_grad(set_to_none=True);self.model.zero_grad(set_to_none=True)
  self.iteration+=1;self.micro=0;self.next_kind='T_FWD'
 def segment(self,split):
  it=self.iteration;micro=self.micro
  kind=self.next_kind if split else ('T_MICRO_OPT' if micro==7 else 'T_MICRO')
  def operation():
   if split:return {'T_FWD':self.forward,'T_BWD':self.backward,'T_OPT':self.opt}[kind]()
   self.forward();self.backward()
   if self.next_kind=='T_OPT':self.opt()
  with torch.cuda.stream(self.stream):_,row=timed(kind,operation)
  row.update(iteration_id=it,microbatch_id=micro,segment_kind=kind,optimizer_completed=int(self.iteration>it))
  return row
 def check_frozen(self):
  assert fingerprint((n,p) for n,p in self.model.named_parameters() if not p.requires_grad)==self.frozen
  return True
 def correctness(self,outdir):
  import gc
  output=[];references=[];gradient_max=0.;loss_max=0.
  # Compare all accumulated gradients and all final trainable parameters, not a sampled layer.
  for split in [False,True]:
   self.reset();losses=[]
   for it in range(5):
    for micro in range(8):
     with torch.cuda.stream(self.stream):
      loss=self.forward()
      if split:finish()
      self.backward();finish()
      losses.append(float(loss.detach()))
     if micro==7:
      if not split:
       references.append([p.grad.detach().cpu().clone() for _,p in self.pairs])
      else:
       for (_,p),ref in zip(self.pairs,references[it]):
        current=p.grad.detach().cpu();gradient_max=max(gradient_max,(current-ref).abs().max().item())
        torch.testing.assert_close(current,ref,rtol=1e-5,atol=1e-6)
     with torch.cuda.stream(self.stream):
      if micro==7:self.opt();finish()
    if split:time.sleep(.002) # Only the correctness check adds a host pause, never benchmark work.
   if not split:
    ref_losses=losses;ref_params=[p.detach().cpu().clone() for _,p in self.pairs]
   else:
    loss_max=max(abs(x-y) for x,y in zip(losses,ref_losses));assert np.allclose(losses,ref_losses,rtol=1e-5,atol=1e-6)
    param_max=0.
    for (_,p),ref in zip(self.pairs,ref_params):
     current=p.detach().cpu();param_max=max(param_max,(current-ref).abs().max().item());torch.testing.assert_close(current,ref,rtol=1e-5,atol=1e-6)
  result={'passed':True,'updates_per_path':5,'microbatches_per_update':8,'all_gradients_compared':True,'all_trainable_parameters_compared':True,'loss_max_abs':loss_max,'gradient_max_abs':gradient_max,'parameter_max_abs':param_max,'frozen_unchanged':self.check_frozen()}
  save(outdir/'training_correctness.json',result);del references,ref_params;gc.collect();self.reset();return result
