"""Historical-order A0/A1/A2 training, common pinned-host async transfer pipeline."""
import os,sys,time,json,argparse,importlib.util,subprocess,traceback,dataclasses,io,math,gc
from pathlib import Path
R=Path(__file__).resolve().parent; BASE=R
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['A0','A1','A2'],required=True);ap.add_argument('--phase',choices=['solo','co','preflight'],required=True);ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=20261108);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--family',default='lora32_heads');ap.add_argument('--sequence');ap.add_argument('--perturb');ap.add_argument('--split',required=True);args=ap.parse_args();OUT=Path(args.out);OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(BASE))
if os.environ.get('PI_MPS')=='1':
 os.environ['PI_MPS_UUID']=subprocess.check_output(['nvidia-smi','-i',os.environ['VLA_GPU'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();os.environ['CUDA_VISIBLE_DEVICES']=os.environ['PI_MPS_UUID'];spec=importlib.util.spec_from_file_location('core',BASE/'core_mps.py');mod=importlib.util.module_from_spec(spec);sys.modules['core']=mod;spec.loader.exec_module(mod)
from common import *
from torch.utils._pytree import tree_flatten,tree_unflatten
from PIL import Image
import pyarrow.parquet as pq

def obs_map(fn,o):return type(o)(**{f.name:tree_map(lambda x:fn(x) if torch.is_tensor(x) else x,getattr(o,f.name)) for f in dataclasses.fields(o)})
def stat(a):return {k:float(np.percentile(a,v)) for k,v in [('p50',50),('p95',95),('p99',99)]}|{'mean':float(np.mean(a)),'max':float(max(a))} if a else None
def event():return torch.cuda.Event(enable_timing=True)
def append(p,row):
 with p.open('a') as f:f.write(json.dumps(row)+'\n')
class Engine:
 def __init__(self,mode,small=False):
  self.seed=args.seed;self.learning_rate=args.lr;self.schedule='cosine';self.total_steps=1000;torch.manual_seed(self.seed);np.random.seed(self.seed);self.mode=mode;self.model=load_model();self.pairs=setup_family(self.model,args.family);self.install_perturb();self.stream=torch.cuda.current_stream();self.copy=torch.cuda.Stream(priority=0);assert self.copy.priority==0
  self.sequence=json.loads((Path(args.sequence) if args.sequence else R/'sequences'/f'{self.seed}.json').read_text());self.micros=[step[j:j+4] for step in self.sequence for j in [0,4]]
  if small:self.micros=self.micros[:2]*3
  self.keys=list(dict.fromkeys(tuple(x) for m in self.micros for x in m));self.data={};self.cache={};self.hits=self.misses=0;self.mi=0;self.prefetched=None;self.step_id=0;self.load_cpu()
  self.masters=[torch.nn.Parameter(p.detach().float().clone()) for _,p in self.pairs];self.ema=[m.detach().clone() for m in self.masters];self.optim=torch.optim.AdamW(self.masters,lr=self.learning_rate,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,foreach=False)
  cpus=[int(x) for x in os.environ.get('LEARNING_CPUS',','.join(map(str,range(GPU*16,GPU*16+16)))).split(',')];pin_threads(cpus)
  save(OUT/'train_device.json',{'physical_gpu':GPU,'visible_sm_count':torch.cuda.get_device_properties(0).multi_processor_count,'copy_stream_priority':self.copy.priority,'mps_percentage':os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'),'affinity':{p.name:sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()}})
  if args.phase=='co':assert torch.cuda.get_device_properties(0).multi_processor_count<84
 def install_perturb(self):
  from perturbations import install
  install(self.model,json.loads(Path(args.perturb).read_text()) if args.perturb else {'kind':'exposure','factor':.05})
 def load_cpu(self):
  split=json.loads(Path(args.split).read_text());tables={}
  for idx,key in enumerate(self.keys):
   ep,start=key
   if ep not in tables:tables[ep]=pq.read_table(Path(split['data_root'])/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet')
   table=tables[ep];row=table.slice(start,1).to_pylist()[0];assert row['task_index'] in [split['history_data_index'],split['target_data_index']]
   raw={'observation/image':np.array(Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')),'observation/wrist_image':np.array(Image.open(io.BytesIO(row['wrist_image']['bytes'])).convert('RGB')),'observation/state':np.array(row['state'],dtype=np.float32),'actions':np.array(table['actions'].slice(start,10).to_pylist(),dtype=np.float32),'prompt':split['target'] if row['task_index']==split['target_data_index'] else split['history']}
   processed=self.model._input_transform_fn(raw);batch=tree_map(lambda x:torch.as_tensor(np.asarray(x).copy())[None],processed);ac=batch.pop('actions').float();obs=model_module.Observation.from_dict(batch);obs=dataclasses.replace(obs,state=obs.state.float());obs=model_module.preprocess_observation(obs,train=False);obs=model_module._observation_to_dtype(obs,self.model.embed_dtype);self.data[key]=(obs_map(lambda x:x.contiguous().pin_memory(),obs),ac.pin_memory())
   if idx%100==0:save(OUT/'status.json',{'state':'loading_inputs','mode':self.mode,'samples':idx,'total':len(self.keys),'gpu':GPU})
  del tables;gc.collect()
 def store(self,key,values):
  leaves,spec=tree_flatten(values);layouts=[];hosts=[];ready_compute=event();ready_compute.record(self.stream);begin,end=event(),event()
  with torch.cuda.stream(self.copy):
   self.copy.wait_event(ready_compute);begin.record()
   for q in leaves:
    h=torch.empty(q.shape,dtype=q.dtype,pin_memory=True);h.copy_(q.detach(),non_blocking=True);q.record_stream(self.copy);hosts.append(h)
   end.record()
  self.cache[key]=(tree_unflatten(hosts,spec),end)
  return begin,end
 def fetch(self,key,records):
  values,ready=self.cache[key];a,b=event(),event()
  with torch.cuda.stream(self.copy):
   self.copy.wait_event(ready);a.record();result=tree_map(lambda x:x.to('cuda',non_blocking=True),values);b.record()
  records.append((a,b,'prefix'));return result,b
 def prepare(self,mi):
  t0=time.monotonic();submit_ns=time.monotonic_ns();parts=[];prefixes=[];records=[];events=[]
  for j,item in enumerate(self.micros[mi]):
   key=tuple(item);obs,ac=self.data[key];seed=self.seed+mi*4+j
   noise=torch.randn(ac.shape,generator=torch.Generator().manual_seed(seed+100000)).pin_memory();t=torch.tensor(np.random.default_rng(seed+200000).beta(1.5,1.,size=1)*.999+.001,dtype=torch.float32).pin_memory()
   a,b=event(),event()
   with torch.cuda.stream(self.copy):
    a.record();o=obs_map(lambda x:x.to('cuda',non_blocking=True),obs);actions=ac.to('cuda',non_blocking=True);n=noise.to('cuda',non_blocking=True);tt=t.to('cuda',non_blocking=True);b.record()
   records.append((a,b,'input'));events.append(b);parts.append((o,actions,n,tt,noise,t))
   if self.mode!='A0' and key in self.cache:pref,done=self.fetch(key,records);prefixes.append(pref);events.append(done)
   else:prefixes.append(None)
  done=event();done.record(self.copy)
  return {'mi':mi,'parts':parts,'prefixes':prefixes,'records':records,'done':done,'host_prepare_ms':(time.monotonic()-t0)*1000,'prefetch_submit_ns':submit_ns,'prefetch_enqueued_ns':time.monotonic_ns()}
 def loss(self,p):
  wait_a,wait_b=event(),event();wait_a.record(self.stream);self.stream.wait_event(p['done']);wait_b.record(self.stream);p['wait_events']=[(wait_a,wait_b)];values=[];d2h=[]
  for j,item in enumerate(self.micros[p['mi']]):
   key=tuple(item);obs=p['parts'][j][0];val=p['prefixes'][j]
   if self.mode!='A0' and key in self.cache:
    self.hits+=1
    if val is None:
     val,done=self.fetch(key,p['records']);a,b=event(),event();a.record();self.stream.wait_event(done);b.record();p['wait_events'].append((a,b))
   else:
    self.misses+=1
    with torch.no_grad():_,mask,kv=self.model.build_prefix_cache(obs)
    val=(mask,kv)
    if self.mode!='A0':d2h.append(self.store(key,val))
   values.append(val)
  p['d2h']=d2h;obs=cat_obs([x[0] for x in p['parts']]);actions,noise,t=[torch.cat([x[k] for x in p['parts']],0) for k in [1,2,3]];x=t[:,None,None]*noise+(1-t[:,None,None])*actions;mask,kv=tree_map(lambda *xs:torch.cat(xs,0),*values)
  # All copied buffers survive until their consuming stream reaches these operations.
  for part in p['parts']:
   obs_map(lambda q:(q.record_stream(self.stream) or q),part[0])
   for q in part[1:4]:q.record_stream(self.stream)
  for val in values:
   for q in tree_flatten(val)[0]:q.record_stream(self.stream)
  return (self.model.velocity_from_suffix(self.model.run_suffix(obs,x,t,kv,mask))-(noise-actions)).square().mean()
 def populate(self):
  start=time.monotonic()
  for i,key in enumerate(self.keys):
   obs,_=self.data[key]
   with torch.no_grad():
    device=obs_map(lambda x:x.to('cuda',non_blocking=True),obs);_,mask,kv=self.model.build_prefix_cache(device);self.store(key,(mask,kv))
   if i%16==15:self.copy.synchronize()
   if i%100==0:save(OUT/'status.json',{'state':'A2_service_prefix_population','samples':i,'total':len(self.keys),'gpu':GPU})
  self.copy.synchronize();save(OUT/'population.json',{'seconds':time.monotonic()-start,'samples':len(self.cache),'included_in_training_throughput':False,'path':'frozen eval/no_grad build_prefix_cache B1, same service prefix path as prior prototype; no augmentation'})
 def parity(self):
  from task_samples import TaskSamples
  original_samples=TaskSamples(self.model,args.split);checks=[]
  split=json.loads(Path(args.split).read_text());targets={e['episode_index'] for e in split['target_train']};check_keys=[next(k for k in self.keys if k[0] in targets),next(k for k in self.keys if k[0] not in targets)]
  for key in check_keys:
   actual,actions=original_samples.get(*key);actual=model_module.preprocess_observation(actual,train=False);actual=model_module._observation_to_dtype(actual,self.model.embed_dtype);host,labels=self.data[key]
   for f in dataclasses.fields(actual):
    aa=tree_flatten(getattr(actual,f.name))[0];bb=tree_flatten(getattr(host,f.name))[0]
    for a,b in zip(aa,bb):
     if torch.is_tensor(a):torch.testing.assert_close(a.cpu(),b,rtol=0,atol=0)
   torch.testing.assert_close(actions.float().cpu(),labels,rtol=0,atol=0);checks.append({'key':key,'exact_match':True})
  save(OUT/'input_parity.json',checks);del original_samples
  old=self.mode;self.mode='A1';self.cache={};p=self.prepare(0);loss=self.loss(p);loss.backward();ref=[q.grad.detach().float().cpu() for _,q in self.pairs];ref_loss=float(loss);self.model.zero_grad(set_to_none=True);self.copy.synchronize();p=self.prepare(0);loss=self.loss(p);loss.backward();num=den=mx=0.
  for (_,q),g in zip(self.pairs,ref):
   actual=q.grad.detach().float().cpu();num+=float((actual-g).square().sum());den+=float(g.square().sum());mx=max(mx,float((actual-g).abs().max()))
  result={'gradient_relative_l2':math.sqrt(num/max(den,1e-30)),'gradient_max_abs':mx,'recompute_loss':ref_loss,'cache_loss':float(loss),'cache_transfer_low_priority':self.copy.priority};save(OUT/'parity.json',result);assert result['gradient_relative_l2']<1e-6,result
  self.model.zero_grad(set_to_none=True);self.copy.synchronize();self.cache={};self.hits=self.misses=0;self.mode=old;del ref,p,loss;gc.collect();torch.cuda.empty_cache()
 def step(self):
  start=time.monotonic_ns();ps=[];losses=[];h0,m0=self.hits,self.misses
  for _ in range(2):
   p=self.prefetched or self.prepare(self.mi);self.prefetched=None;loss=self.loss(p);(loss/2).backward();losses.append(loss.detach());ps.append(p);self.mi+=1
   # Enqueue next-microbatch transfers while current backward/optimizer executes.
   if self.mi<len(self.micros):self.prefetched=self.prepare(self.mi)
  for (_,q),master in zip(self.pairs,self.masters):master.grad=q.grad.detach().float()
  group_norms={}
  if getattr(self,'vision',False) or getattr(self,'record_group_norms',False):
   for group,visual in [('visual',True),('action',False)]:
    squares=[m.grad.float().square().sum() for (n,q),m in zip(self.pairs,self.masters) if n.startswith('img.')==visual];group_norms[group+'_raw_gradient_l2']=float(torch.stack(squares).sum().sqrt()) if squares else 0.
  norm=torch.nn.utils.clip_grad_norm_(self.masters,1.,error_if_nonfinite=True);self.optim.param_groups[0]['lr']=self.learning_rate*((self.step_id+1)/50 if self.step_id+1<=50 else (.1+.9*.5*(1+math.cos(math.pi*min(1.,(self.step_id+1-50)/(self.total_steps-50))))) if self.schedule=='cosine' else 1.);self.optim.step()
  with torch.no_grad():
   for (_,q),master,e in zip(self.pairs,self.masters,self.ema):q.copy_(master);e.mul_(.999).add_(master,alpha=.001)
  self.optim.zero_grad(set_to_none=True);self.model.zero_grad(set_to_none=True);done=event();done.record(self.stream);done.synchronize();finish=time.monotonic_ns();self.step_id+=1
  # D2H stores are part of the miss cost and must complete before returning a timed step.
  for p in ps:
   for a,b in p['d2h']:b.synchronize()
  end=time.monotonic_ns();row={'step':self.step_id,'learning_rate':self.optim.param_groups[0]['lr'],'start_ns':start,'finish_ns':end,'compute_finish_ns':finish,'seconds':(end-start)/1e9,'loss':float(torch.stack(losses).mean()),'hits':self.hits-h0,'misses':self.misses-m0,'cumulative_hits':self.hits,'cumulative_misses':self.misses,'resident_samples':len(self.cache),'h2d_ms':sum(a.elapsed_time(b) for p in ps for a,b,_ in p['records']),'prefix_h2d_ms':sum(a.elapsed_time(b) for p in ps for a,b,k in p['records'] if k=='prefix'),'input_h2d_ms':sum(a.elapsed_time(b) for p in ps for a,b,k in p['records'] if k=='input'),'d2h_ms':sum(a.elapsed_time(b) for p in ps for a,b in p['d2h']),'compute_wait_for_h2d_ms':sum(a.elapsed_time(b) for p in ps for a,b in p['wait_events']),'host_prefetch_prepare_ms':sum(p['host_prepare_ms'] for p in ps),'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
  row['microbatch_transfers']=[{'microbatch':p['mi'],'prefetch_submit_ns':p['prefetch_submit_ns'],'prefetch_enqueued_ns':p['prefetch_enqueued_ns'],'h2d_ms':sum(a.elapsed_time(b) for a,b,_ in p['records']),'prefix_h2d_ms':sum(a.elapsed_time(b) for a,b,k in p['records'] if k=='prefix')} for p in ps]
  row.update(group_norms);assert math.isfinite(row['loss']);return row
