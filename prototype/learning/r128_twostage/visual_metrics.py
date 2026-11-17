"""Observational diagnostics with fixed samples/noise and RNG/module-mode restoration."""
from pathlib import Path
import json,csv,time,random,math,contextlib
import torch,numpy as np
from cache_engine import obs_map,cat_obs,model_module,save
from task_samples import TaskSamples
class VisualMetrics:
 def __init__(self,tr,out,cfg):
  self.tr=tr;self.out=Path(out);self.cfg=cfg;self.root=self.out/'metrics';self.root.mkdir(exist_ok=True);self.manifest=json.loads(Path(cfg['metrics_manifest']).read_text());self.data=[]
  with self.preserve():
   loader=TaskSamples(tr.model,cfg['split'])
   for i,key in enumerate(self.manifest['adaptation']+self.manifest['history']):
    obs,actions=loader.get(*key);obs=model_module.preprocess_observation(obs,train=False);obs=model_module._observation_to_dtype(obs,tr.model.embed_dtype)
    noise=torch.randn(actions.shape,generator=torch.Generator().manual_seed(self.manifest['fixed_noise_seed']+i));t=torch.tensor([np.random.default_rng(self.manifest['fixed_flow_time_seed']+i).beta(1.5,1.)*.999+.001],dtype=torch.float32)
    self.data.append((obs_map(lambda x:x.detach().cpu(),obs),actions.detach().float().cpu(),noise,t))
   del loader
  parent=self.root/'parent.pt'
  if parent.exists():self.parent=torch.load(parent,map_location='cpu',weights_only=False)
  else:
   assert tr.step_id==0,'parent diagnostic reference missing on resume'
   with self.preserve():self.parent=self.probes()
   self.atomic(parent,self.parent)
  snapshots=[p for p in self.root.glob('state_*.pt') if int(p.stem.split('_')[1])<=tr.step_id]
  if snapshots:
   p=max(snapshots,key=lambda p:int(p.stem.split('_')[1]));state=torch.load(p,map_location='cpu',weights_only=False);self.previous=state['probes'];self.previous_masters=state['masters'];self.last=state['step']
  else:self.previous=self.parent;self.previous_masters=[m.detach().cpu().clone() for m in tr.masters];self.last=0
  self.log=self.out/'visual_metrics.csv'
  if self.log.exists():
   rows=list(csv.DictReader(self.log.open()));valid=[x for x in rows if int(x['step'])<=tr.step_id]
   if len(valid)!=len(rows):
    self.log.rename(self.log.with_name(f'visual_metrics_before_retry_{time.time_ns()}.csv'))
    with self.log.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(valid)
  if not self.log.exists():self.record(0,initial=True)
 @contextlib.contextmanager
 def preserve(self):
  cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state();nr=np.random.get_state();pr=random.getstate();modes=[(m,m.training) for m in self.tr.model.modules()];self.tr.model.eval()
  try:
   with torch.no_grad():yield
  finally:
   for m,mode in modes:m.training=mode
   torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda);np.random.set_state(nr);random.setstate(pr)
   assert torch.equal(torch.get_rng_state(),cpu) and torch.equal(torch.cuda.get_rng_state(),cuda)
 def atomic(self,p,data):
  tmp=p.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(p)
 def probes(self):
  result=[]
  for item in self.data[:32]:
   obs=obs_map(lambda x:x.cuda(),item[0]);hidden,mask,kv=self.tr.model.build_prefix_cache(obs);result.append(hidden[mask].detach().cpu());del hidden,mask,kv,obs
  return result
 def heldout(self,start):
  values=[]
  for i in range(start,start+64,4):
   items=self.data[i:i+4];obs=cat_obs([obs_map(lambda x:x.cuda(),x[0]) for x in items]);ac,noise,t=[torch.cat([x[k] for x in items],0).cuda() for k in [1,2,3]];_,mask,kv=self.tr.model.build_prefix_cache(obs);xt=t[:,None,None]*noise+(1-t[:,None,None])*ac;v=self.tr.model.velocity_from_suffix(self.tr.model.run_suffix(obs,xt,t,kv,mask));values.append(float((v-(noise-ac)).square().mean()));del kv,v,obs
  return sum(values)/len(values)
 def record(self,step,initial=False):
  started=time.monotonic();model=self.tr.model;assert not model.llm.gradient_checkpointing and not model.img.encoder.gradient_checkpointing
  rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone();versions=[p._version for p in model.parameters()];modes=[m.training for m in model.modules()]
  with self.preserve():
   adaptation=self.heldout(0);history=self.heldout(64);current=self.parent if initial else self.probes()
   distance=lambda ref:float(np.mean([float((1-torch.nn.functional.cosine_similarity(a.float(),b.float(),dim=-1)).mean()) for a,b in zip(current,ref)]))
   delta={'visual':0.,'action':0.}
   masters=[m.detach().cpu().clone() for m in self.tr.masters]
   for (n,p),a,b in zip(self.tr.pairs,masters,self.previous_masters):delta['visual' if n.startswith('img.') else 'action']+=float((a-b).square().sum())
   rows=[json.loads(x) for x in (self.out/'steps.jsonl').read_text().splitlines()] if (self.out/'steps.jsonl').exists() else [];window=[x for x in rows if self.last<x['step']<=step]
   row={'step':step,'interval_start':self.last,'interval_steps':len(window),'train_loss_mean':float(np.mean([x['loss'] for x in window])) if window else None,'heldout_adaptation_loss':adaptation,'heldout_history_loss':history,'representation_cosine_distance_previous':0. if initial else distance(self.previous),'representation_cosine_distance_parent':0. if initial else distance(self.parent),'visual_parameter_delta_l2':math.sqrt(delta['visual']),'action_parameter_delta_l2':math.sqrt(delta['action'])}
   for group in ['visual','action']:
    values=[x[group+'_raw_gradient_l2'] for x in window];row[group+'_gradient_l2_mean']=float(np.mean(values)) if values else None;row[group+'_gradient_l2_max']=max(values) if values else None
   row['diagnostic_seconds']=time.monotonic()-started
  assert versions==[p._version for p in model.parameters()],'diagnostics modified weights'
  assert modes==[m.training for m in model.modules()] and torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state()),'diagnostics changed RNG or training mode'
  exists=self.log.exists()
  with self.log.open('a') as f:
   w=csv.DictWriter(f,fieldnames=list(row))
   if not exists:w.writeheader()
   w.writerow(row)
  self.previous=current;self.previous_masters=masters;self.last=step;self.atomic(self.root/f'state_{step}.pt',{'step':step,'probes':current,'masters':masters});self.prune_restart_states(step);save(self.out/'metrics_status.json',{'state':'complete_at_step','step':step,'observational_checks_passed':True,'diagnostic_seconds':row['diagnostic_seconds']})


 def prune_restart_states(self,step):
  # A training recovery point is committed every 250 steps. Keep its metric
  # snapshot plus newer snapshots; diagnostic CSV and EMA checkpoints remain.
  boundary=max(0,(step//250-1)*250)
  candidates=sorted(self.root.glob('state_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
  older=[p for p in candidates if int(p.stem.split('_')[1])<=boundary]
  for p in older[:-1]:p.unlink()
