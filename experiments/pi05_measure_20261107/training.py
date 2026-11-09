import sys,os,time,json,traceback,gc,contextlib,dataclasses,math
from pathlib import Path
sys.path.insert(0,'/home/hengyi/vla_overnight_20261104')
from core import *
from torch.utils._pytree import tree_map,tree_leaves
OUT=Path('/home/hengyi/pi05_measure_20261107/training');OUT.mkdir(exist_ok=True)
class VisionLoRA(torch.nn.Module):
 def __init__(self,base,rank=16):
  super().__init__();self.base=base;self.rank=rank
  self.A=torch.nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=base.weight.dtype));torch.nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
  self.B=torch.nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=base.weight.dtype))
 def forward(self,x):return self.base(x)+torch.nn.functional.linear(torch.nn.functional.linear(x,self.A),self.B)
class BenchTrain:
 def __init__(self,vision=False):
  self.vision=vision;self.tier='action_vision_lora' if vision else 'action';self.model=load_model();configure(self.model,'L2');torch.manual_seed(20261107)
  self.lora_modules=[]
  if vision:
   for name,module in list(self.model.img.named_modules()):
    if isinstance(module,torch.nn.Linear) and name.endswith(('mlp.fc1','mlp.fc2')):
     parent,leaf=name.rsplit('.',1);setattr(self.model.img.get_submodule(parent),leaf,VisionLoRA(module));self.lora_modules.append('img.'+name)
   assert self.lora_modules
   self.model.gradient_checkpointing_enable({'use_reentrant':False})
  self.pairs=[(n,p) for n,p in self.model.named_parameters() if p.requires_grad]
  self.frozen=fingerprint((n,p) for n,p in self.model.named_parameters() if not p.requires_grad)
  self.masters=[torch.nn.Parameter(p.detach().float().clone()) for _,p in self.pairs]
  self.optimizer=torch.optim.AdamW(self.masters,lr=2.5e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,foreach=False)
  self.cache=[];samples=Samples(self.model);seq=json.loads((OLD/'sample_sequence.json').read_text())[:60]
  for batch in seq:
   for ep,idx in batch:
    ob,ac=samples.get(ep,idx);self.cache.append((tree_map(lambda x:x.cpu() if torch.is_tensor(x) else x,dataclasses.asdict(ob)),ac.cpu()))
  self.obs_type=type(ob);del samples;self.iteration=0;self.micro=0;self.next_kind='T_FWD';self.stream=torch.cuda.Stream();finish()
 def inputs(self,index):
  d,a=self.cache[index%len(self.cache)];obs=self.obs_type(**tree_map(lambda x:x.cuda() if torch.is_tensor(x) else x,d));actions=a.cuda().float()
  seed=20261020+index;noise=torch.randn(actions.shape,generator=torch.Generator().manual_seed(seed+100000)).cuda();t=torch.tensor(np.random.default_rng(seed+200000).beta(1.5,1.,size=1)*.999+.001,device='cuda',dtype=torch.float32)
  return obs,actions,noise,t,torch.Generator().manual_seed(seed)
 def prepare(self,index):
  obs,actions,noise,t,rng=self.inputs(index)
  obs=model_module.preprocess_observation(obs,train=True,rng=rng);obs=model_module._observation_to_dtype(obs,self.model.embed_dtype)
  return obs,t[:,None,None]*noise+(1-t[:,None,None])*actions,noise-actions,t
 def prefix(self,obs):
  with contextlib.nullcontext() if self.vision else torch.no_grad():_,mask,kv=self.model.build_prefix_cache(obs)
  if not self.vision:assert all(not x.requires_grad and x.grad_fn is None for x in tree_leaves(kv) if torch.is_tensor(x))
  return mask,kv
 def forward(self):
  obs,x,u,t=self.prepare(self.iteration*8+self.micro);mask,kv=self.prefix(obs)
  self.loss=(self.model.velocity_from_suffix(self.model.run_suffix(obs,x,t,kv,mask))-u).square().mean();self.next_kind='T_BWD';return self.loss
 def backward(self):
  (self.loss/8).backward();self.loss=None;self.micro+=1;self.next_kind='T_OPT' if self.micro==8 else 'T_FWD'
 def opt(self):
  for (_,p),m in zip(self.pairs,self.masters):
   assert p.grad is not None;m.grad=p.grad.detach().float()
  torch.nn.utils.clip_grad_norm_(self.masters,1.,error_if_nonfinite=True);self.optimizer.step()
  with torch.no_grad():
   for (_,p),m in zip(self.pairs,self.masters):p.copy_(m)
  self.optimizer.zero_grad(set_to_none=True);self.model.zero_grad(set_to_none=True);self.iteration+=1;self.micro=0;self.next_kind='T_FWD'
 def segment(self,split=True):
  it=self.iteration;micro=self.micro;kind=self.next_kind
  with torch.cuda.stream(self.stream):_,row=timed(kind,{'T_FWD':self.forward,'T_BWD':self.backward,'T_OPT':self.opt}[kind])
  return {**row,'iteration_id':it,'microbatch_id':micro,'segment_kind':kind,'optimizer_completed':int(self.iteration>it)}
 def parity(self):
  ob,ac,n,t,rng=self.inputs(0);native=self.model.compute_loss(ob,ac,train=True,rng=rng,noise=n,time=t).mean();native.backward()
  refs=[p.grad.detach().float().cpu() for _,p in self.pairs];value=float(native.detach());self.model.zero_grad(set_to_none=True);del native
  split=self.forward();split.backward();delta2=ref2=0.;maxabs=0.
  for (_,p),ref in zip(self.pairs,refs):
   x=p.grad.detach().float().cpu();delta2+=float((x-ref).square().sum());ref2+=float(ref.square().sum());maxabs=max(maxabs,float((x-ref).abs().max()))
  result={'native_loss':value,'split_loss':float(split.detach()),'loss_abs_diff':abs(value-float(split.detach())),'gradient_relative_l2':(delta2/max(ref2,1e-30))**.5,'gradient_max_abs':maxabs,'all_trainable_gradients_checked':True}
  self.model.zero_grad(set_to_none=True);self.loss=None;self.next_kind='T_FWD';del refs,split;gc.collect();torch.cuda.empty_cache();return result
 def measured_step(self):
  torch.cuda.reset_peak_memory_stats();start=time.perf_counter();totals=dict(preprocess_ms=0.,prefix_ms=0.,action_ms=0.,backward_ms=0.,optimizer_ms=0.)
  for j in range(8):
   (obs,x,u,t),row=timed('preprocess',lambda:self.prepare(self.iteration*8+j));totals['preprocess_ms']+=row['cuda_ms']
   (mask,kv),row=timed('prefix',lambda:self.prefix(obs));totals['prefix_ms']+=row['cuda_ms']
   def action():return (self.model.velocity_from_suffix(self.model.run_suffix(obs,x,t,kv,mask))-u).square().mean()
   self.loss,row=timed('action',action);totals['action_ms']+=row['cuda_ms']
   _,row=timed('backward',self.backward);totals['backward_ms']+=row['cuda_ms'];del kv,mask,obs,x,u,t
  _,row=timed('optimizer',self.opt);totals['optimizer_ms']=row['cuda_ms'];totals.update(step=self.iteration,wall_ms=(time.perf_counter()-start)*1000,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());return totals

def main():
 for vision in [False,True]:
  tier='action_vision_lora' if vision else 'action';out=OUT/tier;out.mkdir(exist_ok=True)
  save(OUT/'status.json',{'state':'loading','tier':tier,'gpu':GPU,'pid':os.getpid()});tr=BenchTrain(vision)
  save(out/'config.json',{'model':'RLinf-Pi05-LIBERO-130-fullshot-SFT','physical_gpu':GPU,'microbatch':1,'accumulation':8,'effective_batch':8,'warmup_steps':10,'measured_steps':50,'lora_rank':16 if vision else None,'lora_alpha':16 if vision else None,'lora_modules':tr.lora_modules,'updated_parameters':[{'name':n,'numel':p.numel(),'dtype':str(p.dtype)} for n,p in tr.pairs],'prefix_no_grad':not vision,'gradient_checkpointing':vision,'optimizer':'AdamW FP32 master','prefix_measurement':'sequential prefix KV then action suffix; native joint loss/gradient parity checked','saved_weights_modified':False})
  parity=tr.parity();save(out/'parity.json',parity)
  if parity['gradient_relative_l2']>.1 or parity['loss_abs_diff']>.01:raise RuntimeError(f'Native vs split mismatch {parity}')
  rows=[]
  for step in range(60):
   row=tr.measured_step();row['warmup']=step<10
   with (out/'raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   save(OUT/'status.json',{'state':'measuring','tier':tier,'step':step+1,'total':60,'last_step_seconds':row['wall_ms']/1000,'gpu':GPU})
   if step>=10:rows.append(row)
  save(out/'summary.json',{k:stats([r[k] for r in rows]) for k in rows[0] if k not in ['warmup','step']})
  assert fingerprint((n,p) for n,p in tr.model.named_parameters() if not p.requires_grad)==tr.frozen
  save(out/'integrity.json',{'frozen_parameters_unchanged':True,'checkpoint_writes':0});del tr;gc.collect();torch.cuda.empty_cache()
 save(OUT/'status.json',{'state':'complete','gpu':GPU})
if __name__=='__main__':
 try:main()
 except Exception:save(OUT/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
