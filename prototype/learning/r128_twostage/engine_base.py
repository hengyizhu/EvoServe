"""Bounded-memory cached LoRA learning, or differentiable vision-prefix recompute."""
import os,sys,json,time,argparse,traceback,hashlib,collections,fcntl,math,gc
from pathlib import Path
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out');ap.add_argument('--smoke',action='store_true');a=ap.parse_args();cfg=json.loads(Path(a.config).read_text());OUT=Path(a.out) if a.out else R/'training'/cfg['name'];OUT.mkdir(parents=True,exist_ok=True)
sys.argv=[str(R/'cache_engine.py'),'--mode','A0' if cfg['vision'] else 'A2','--phase','solo','--out',str(OUT),'--seed',str(cfg['seed']),'--lr',str(cfg['lr']),'--family',cfg['family'],'--sequence',cfg['sequence'],'--perturb',cfg['perturb'],'--split',cfg['split']]
import cache_engine as base
from cache_engine import torch,np,tree_map,tree_flatten,cat_obs,obs_map,event,save,append,GPU,fingerprint
CACHE=Path(cfg['cache_dir']);CACHE.mkdir(parents=True,exist_ok=True)
class Engine(base.Engine):
 def __init__(self):
  self.pinned=collections.OrderedDict();self.inflight=[];self.disk_reads=0;self.vision=cfg['vision'];self.record_group_norms=cfg.get('record_action_probes',False);super().__init__('A0' if self.vision else 'A2',small=a.smoke);self.total_steps=cfg.get('schedule_horizon',cfg['steps']);self.schedule='cosine'
 def cache_path(self,key):return CACHE/f'{key[0]}_{key[1]}.pt'
 def store(self,key,values):
  # D2H is outside the measured learning loop: only frozen prefixes use this.
  aa,bb=event(),event();aa.record(self.copy)
  with torch.cuda.stream(self.copy):
   self.copy.wait_stream(self.stream);host=tree_map(lambda x:x.detach().to('cpu',non_blocking=False),values);bb.record()
  bb.synchronize();path=self.cache_path(key)
  with path.with_suffix('.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if not path.exists():
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save(host,tmp);tmp.replace(path)
  self.cache[key]=str(path);return aa,bb
 def fetch(self,key,records):
  self.inflight=[(e,v) for e,v in self.inflight if not e.query()]
  if key in self.pinned:values=self.pinned.pop(key)
  else:
   values=tree_map(lambda x:x.pin_memory(),torch.load(self.cache[key],map_location='cpu',weights_only=True,mmap=True));self.disk_reads+=1
  self.pinned[key]=values
  while len(self.pinned)>32:self.pinned.popitem(last=False)
  aa,bb=event(),event()
  with torch.cuda.stream(self.copy):aa.record();result=tree_map(lambda x:x.to('cuda',non_blocking=True),values);bb.record()
  self.inflight.append((bb,values));records.append((aa,bb,'prefix'));return result,bb
 def populate(self):
  if self.vision:return
  if cfg.get('reuse_parent_cache'):
   source=Path(cfg['parent_cache_source']);metadata=json.loads((source/'identity.json').read_text())
   assert metadata['parent_sha256']=='4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683'
   assert metadata['perturb']==json.loads(Path(cfg['perturb']).read_text()) and metadata['prefix_batch']==1 and not metadata['augmentation']
   start=time.monotonic();missing=[k for k in self.keys if not (source/f'{k[0]}_{k[1]}.pt').exists()]
   assert not missing,f'Original parent cache incomplete: {len(missing)} missing; refusing to rebuild'
   split=json.loads(Path(cfg['split']).read_text());targets={e['episode_index'] for e in split['target_train']};check=[next(k for k in self.keys if k[0] in targets),next(k for k in self.keys if k[0] not in targets)]
   for key in check:
    with torch.no_grad():
     obs=obs_map(lambda x:x.cuda(),self.data[key][0]);_,mask,kv=self.model.build_prefix_cache(obs);expected=(mask,kv);actual=torch.load(source/f'{key[0]}_{key[1]}.pt',map_location='cuda',weights_only=True)
     aa,sa=tree_flatten(expected);bb,sb=tree_flatten(actual);assert sa==sb
     for x,y in zip(aa,bb):torch.testing.assert_close(x,y,rtol=0,atol=0)
     del obs,mask,kv,expected,actual,aa,bb
   self.cache={key:str(source/f'{key[0]}_{key[1]}.pt') for key in self.keys}
   save(OUT/'parent_cache_reuse.json',{'state':'passed','source':str(source),'entries':len(self.cache),'verification_keys':check,'exact_match':True,'rebuilt':0})
   save(OUT/'population.json',{'seconds':time.monotonic()-start,'reused':len(self.cache),'built':0,'unique_samples':len(self.cache),'cache_location':str(source),'includes_validation_and_index_time':True})
   return
  start=time.monotonic();reused=built=0
  for i,key in enumerate(self.keys):
   path=self.cache_path(key)
   if path.exists():self.cache[key]=str(path);reused+=1
   else:
    with torch.no_grad():
     obs=obs_map(lambda x:x.to('cuda',non_blocking=True),self.data[key][0]);_,mask,kv=self.model.build_prefix_cache(obs);self.store(key,(mask,kv));built+=1
   if i%50==0:save(OUT/'status.json',{'state':'prefix_cache_preparation','gpu':GPU,'samples':i,'total':len(self.keys),'reused':reused,'built':built})
  self.copy.synchronize();save(OUT/'population.json',{'seconds':time.monotonic()-start,'reused':reused,'built':built,'unique_samples':len(self.cache),'pinned_lru_limit':32,'cache_location':str(CACHE)})
 def loss(self,p):
  if not self.vision:return super().loss(p)
  aa,bb=event(),event();aa.record();self.stream.wait_event(p['done']);bb.record();p['wait_events']=[(aa,bb)];p['d2h']=[]
  obs=cat_obs([v[0] for v in p['parts']]);actions,noise,t=[torch.cat([v[k] for v in p['parts']],0) for k in [1,2,3]]
  _,mask,kv=self.model.build_prefix_cache(obs);assert any(x.requires_grad for x in tree_flatten(kv)[0] if torch.is_tensor(x))
  for part in p['parts']:
   obs_map(lambda x:(x.record_stream(self.stream) or x),part[0])
   for x in part[1:4]:x.record_stream(self.stream)
  self.misses+=4;x=t[:,None,None]*noise+(1-t[:,None,None])*actions
  return (self.model.velocity_from_suffix(self.model.run_suffix(obs,x,t,kv,mask))-(noise-actions)).square().mean()
 def vision_check(self):
  p=self.prepare(0);loss=self.loss(p);loss.backward();norms={n:float(v.grad.detach().float().norm()) for n,v in self.pairs if n.startswith('img.') and n.endswith('.B')};assert norms and all(math.isfinite(x) for x in norms.values()) and sum(norms.values())>0
  save(OUT/'vision_gradient_check.json',{'state':'passed','loss':float(loss.detach()),'vision_B_gradient_norms':norms,'checkpointing':self.model.llm.gradient_checkpointing and self.model.img.encoder.gradient_checkpointing})
  self.model.zero_grad(set_to_none=True);self.copy.synchronize();del p,loss;torch.cuda.empty_cache();self.hits=self.misses=0

def persist(tr,step,parent):
 payload={'parameters':{n:e.detach().to(dtype=p.dtype).cpu() for (n,p),e in zip(tr.pairs,tr.ema)},'family':cfg['family'],'seed':cfg['seed'],'step':step,'ema_decay':.999,'lr':cfg['lr'],'parent':parent,'weights_kind':'EMA','perturb':json.loads(Path(cfg['perturb']).read_text()),'eval_task_id':cfg['eval_task']}
 path=OUT/f'ema_{step}.pt';tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path);save(OUT/f'checkpoint_{step}.json',{'state':'ready','path':str(path)})
def resume_save(tr):
 payload={'step':tr.step_id,'masters':[x.detach().cpu() for x in tr.masters],'ema':[x.detach().cpu() for x in tr.ema],'optimizer':tree_map(lambda x:x.detach().cpu() if torch.is_tensor(x) else x,tr.optim.state_dict()),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'config':cfg}
 p=OUT/'resume.pt';tmp=p.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(p)
def main():
 identity={'parent_sha256':'4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683','perturb':json.loads(Path(cfg['perturb']).read_text()),'dtype':'validated BF16 observation / selective mixed model','prefix_batch':1,'augmentation':False,'schema':1}
 with (CACHE/'identity.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);p=CACHE/'identity.json'
  if p.exists():assert json.loads(p.read_text())==identity
  else:save(p,identity)
 tr=Engine();numel=sum(p.numel() for n,p in tr.pairs);assert numel>=20876320
 if not cfg['vision']:assert numel==20876320
 frozen=fingerprint((n,p) for n,p in tr.model.named_parameters() if not p.requires_grad)
 save(OUT/'config.json',dict(cfg,trainable_parameters=numel,cache_storage='shared NVMe plus32-sample pinned LRU' if not cfg['vision'] else 'none; differentiable prefix recomputation',parallel_learning=True))
 save(OUT/'trainable_modules.json',[{'name':n,'numel':p.numel(),'dtype':str(p.dtype)} for n,p in tr.pairs])
 if tr.vision:tr.vision_check()
 else:tr.parity();tr.pinned.clear()
 tr.populate();first=0;resume=OUT/'resume.pt'
 if resume.exists():
  state=torch.load(resume,map_location='cpu',weights_only=False);assert state['config']==cfg;first=state['step'];tr.step_id=first;tr.mi=first*2
  tr.optim.load_state_dict(state['optimizer'])
  with torch.no_grad():
   for (n,p),m,e,sm,se in zip(tr.pairs,tr.masters,tr.ema,state['masters'],state['ema']):m.copy_(sm);e.copy_(se);p.copy_(m)
  torch.set_rng_state(state['rng']);torch.cuda.set_rng_state(state['cuda_rng']);del state
  ck=OUT/f'ema_{first}.pt'
  if ck.exists():
   delta=torch.load(ck,map_location='cpu',weights_only=True)
   for (n,p),e in zip(tr.pairs,tr.ema):torch.testing.assert_close(e.detach().to(dtype=p.dtype).cpu(),delta['parameters'][n],rtol=0,atol=0)
   save(OUT/'resume_validation.json',{'state':'passed','step':first,'EMA_matches_saved_checkpoint':True});del delta
  # Drop only uncommitted steps after the recovery point; preserve them as audit evidence.
  log=OUT/'steps.jsonl'
  if log.exists():
   old=log.read_text().splitlines();valid=[]
   for line in old:
    try:r=json.loads(line)
    except json.JSONDecodeError:break
    if r['step']<=first:valid.append(line)
   if len(valid)!=len(old):log.rename(OUT/f'steps_before_retry_{time.time_ns()}.jsonl');log.write_text('\n'.join(valid)+'\n')
 if not resume.exists() and (OUT/'steps.jsonl').exists():
  (OUT/'steps.jsonl').rename(OUT/f'steps_uncommitted_{time.time_ns()}.jsonl')
 parent=json.loads(Path('/home/hengyi/Workspace/01_VLA-Serving/Environment-and-Earlier-Runs/vla_followup_20261020/audit/parent_download.json').read_text());total=2 if a.smoke else cfg['steps'];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 for step in range(first,total):
  row=tr.step();append(OUT/'steps.jsonl',row)
  if (step+1)%10==0:save(OUT/'status.json',{'state':'training','gpu':GPU,'step':step+1,'total_steps':total,'elapsed_s':time.monotonic()-start})
  if step+1 in cfg['checkpoints'] or (a.smoke and step+1==2):persist(tr,step+1,parent)
  if (step+1)%250==0 or step+1==total:resume_save(tr)
 assert fingerprint((n,p) for n,p in tr.model.named_parameters() if not p.requires_grad)==frozen
 rows=[json.loads(x) for x in (OUT/'steps.jsonl').read_text().splitlines()];assert sorted(x['step'] for x in rows)==list(range(1,total+1))
 save(OUT/'timing.json',{'steps':total,'mean_step_s':float(np.mean([x['seconds'] for x in rows])),'median_step_s':float(np.median([x['seconds'] for x in rows])),'peak_cuda_bytes':max(x['peak_cuda_bytes'] for x in rows),'cache_disk_reads':tr.disk_reads,'parallel_learning':True})
 save(OUT/'status.json',{'state':'complete','gpu':GPU,'steps':total,'frozen_unchanged':True,'smoke':a.smoke})
if __name__=='__main__':
 try:main()
 except Exception:save(OUT/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
