import os,sys,time,json,gc,traceback,dataclasses
from pathlib import Path
sys.path.insert(0,'/home/hengyi/pi05_optimized_20261107')
from training_mb4 import *
R=Path('/home/hengyi/pi05_cache_20261107/prototype');R.mkdir(exist_ok=True)
class CacheTrain(TrainMB4):
 def __init__(self):
  super().__init__(False);self.model.eval();self.host_cache={};self.cached=False
 def prepare(self,index):
  obs,actions,noise,t,rng=self.inputs(index)
  obs=model_module.preprocess_observation(obs,train=False);obs=model_module._observation_to_dtype(obs,self.model.embed_dtype)
  return obs,t[:,None,None]*noise+(1-t[:,None,None])*actions,noise-actions,t
 def prepare4(self,index):
  parts=[self.prepare(index+j) for j in range(4)];return cat_obs([x[0] for x in parts]),*[torch.cat([x[k] for x in parts],dim=0) for k in [1,2,3]]
 def build_host(self,n=64):
  for i in range(n):
   obs,*_=self.prepare(i)
   with torch.no_grad():mask,kv=self.prefix(obs)
   self.host_cache[i]=tree_map(lambda x:x.detach().cpu().pin_memory(),(mask,kv))
  self.cache=self.cache[:n]
 def get_cached(self,index,batch):
  cpu=tree_map(lambda *xs:torch.cat(xs,dim=0).pin_memory(),*[self.host_cache[(index+j)%len(self.host_cache)] for j in range(batch)])
  return tree_map(lambda x:x.cuda(non_blocking=True),cpu)
 def forward(self):
  index=self.iteration*8+self.micro*4;obs,x,u,t=self.prepare4(index)
  mask,kv=self.get_cached(index,4) if self.cached else self.prefix(obs)
  self.loss=(self.model.velocity_from_suffix(self.model.run_suffix(obs,x,t,kv,mask))-u).square().mean();self.next_kind='T_BWD';return self.loss
 def gradient(self,batch,cache=False):
  self.model.zero_grad(set_to_none=True);losses=[]
  for i in range(0,8,batch):
   obs,x,u,t=self.prepare(i) if batch==1 else self.prepare4(i)
   mask,kv=self.get_cached(i,batch) if cache else self.prefix(obs)
   loss=(self.model.velocity_from_suffix(self.model.run_suffix(obs,x,t,kv,mask))-u).square().mean();(loss*batch/8).backward();losses.append(float(loss))
  grads=[p.grad.detach().float().cpu() for _,p in self.pairs];self.model.zero_grad(set_to_none=True)
  return {'loss':float(np.mean(losses)),'grads':grads}
def compare(a,b):
 d=r=0.;mx=0.
 for x,y in zip(a['grads'],b['grads']):d+=float((x-y).square().sum());r+=float(x.square().sum());mx=max(mx,float((x-y).abs().max()))
 return {'reference_loss':a['loss'],'other_loss':b['loss'],'gradient_relative_l2':(d/max(r,1e-30))**.5,'gradient_max_abs':mx}
def main():
 save(R/'status.json',{'state':'loading','pid':os.getpid()});tr=CacheTrain();tr.build_host();save(R/'config.json',{'cache_samples':64,'prefix_batch_at_production':1,'training_microbatch':4,'accumulation':2,'augmentation':False,'model_mode':'eval (gradients enabled for action expert)','cache_mode':'pinned CPU, cat four samples then H2D per microbatch','updated_parameters':[n for n,p in tr.pairs]})
 results={};b1=tr.gradient(1);b4=tr.gradient(4);results['mb1_vs_mb4_noaug_eval']=compare(b1,b4);del b1
 cached=tr.gradient(4,True);results['service_B1_cache_vs_recompute_B4']=compare(b4,cached);del cached,b4;save(R/'gradient_checks.json',results)
 # Identical B4 prefix values, copied through host, isolate cache mechanism from batch numerics.
 obs,x,u,t=tr.prepare4(0);mask,kv=tr.prefix(obs);host=tree_map(lambda q:q.detach().cpu().pin_memory(),(mask,kv));restored=tree_map(lambda q:q.cuda(non_blocking=True),host)
 def grad(values):
  m,k=values;loss=(tr.model.velocity_from_suffix(tr.model.run_suffix(obs,x,t,k,m))-u).square().mean();loss.backward();g={'loss':float(loss),'grads':[p.grad.detach().float().cpu() for _,p in tr.pairs]};tr.model.zero_grad(set_to_none=True);return g
 ref=grad((mask,kv));other=grad(restored);results['same_B4_prefix_host_roundtrip']=compare(ref,other);del ref,other,host,restored,kv,mask,obs,x,u,t
 # Original augmented train-mode diagnostic retained, rather than overwritten.
 results['original_augmented_train_mode'] = json.load(open('/home/hengyi/pi05_optimized_20261107/training_mb4/action/microbatch_parity.json'))
 tr.model.train();b1=tr.gradient(1);b4=tr.gradient(4);results['mb1_vs_mb4_noaug_train']=compare(b1,b4);del b1,b4
 tr.model.eval();torch.backends.mha.set_fastpath_enabled(False);b1=tr.gradient(1);b4=tr.gradient(4);results['mb1_vs_mb4_noaug_eval_mha_fastpath_off']=compare(b1,b4);del b1,b4;torch.backends.mha.set_fastpath_enabled(True)
 samples=[tr.prepare(i)[0] for i in range(8)];batched=cat_obs(samples[:4]);results['input_audit']={'token_shapes':[list(o.tokenized_prompt.shape) for o in samples],'token_mask_valid_counts':[int(o.tokenized_prompt_mask.sum()) for o in samples],'batched_masks_identical_to_concatenated':all(torch.equal(getattr(batched,k),torch.cat([getattr(o,k) for o in samples[:4]],dim=0)) for k in ['tokenized_prompt','tokenized_prompt_mask']),'loss':'mean over B*10*32 then weighted by microbatch_size/8; equal-size windows, no variable-length action padding normalization'};save(R/'gradient_checks.json',results)
 gc.collect();torch.cuda.empty_cache();tr.cached=True;rows=[]
 for i in range(60):
  start=time.monotonic()
  while not tr.segment()['optimizer_completed']:pass
  row={'step':i,'seconds':time.monotonic()-start,'warmup':i<10};rows.append(row)
  with (R/'solo_raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  save(R/'status.json',{'state':'solo','step':i+1})
 save(R/'solo_summary.json',{'steps_per_s':50/sum(x['seconds'] for x in rows[10:]),'step_ms':stats([x['seconds']*1000 for x in rows[10:]]),'includes_H2D_and_host_batch_assembly':True,'cache_population_excluded':True})
 save(R/'status.json',{'state':'complete','next':'MPS30 coexecution'})
if __name__=='__main__':
 try:main()
 except Exception:save(R/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
