import sys,gc,traceback,dataclasses
from pathlib import Path
sys.path.insert(0,'/home/hengyi/pi05_measure_20261107')
from training import *
R=Path('/home/hengyi/pi05_optimized_20261107');OUT=R/'training_mb4';OUT.mkdir(exist_ok=True)
def cat_obs(items):
 return type(items[0])(**{f.name:tree_map(lambda *xs:torch.cat(xs,dim=0) if torch.is_tensor(xs[0]) else xs[0],*[getattr(o,f.name) for o in items]) for f in dataclasses.fields(items[0])})
class TrainMB4(BenchTrain):
 def prepare4(self,index):
  parts=[super(TrainMB4,self).prepare(index+j) for j in range(4)];return cat_obs([x[0] for x in parts]),*[torch.cat([x[k] for x in parts],dim=0) for k in [1,2,3]]
 def forward(self):
  obs,x,u,t=self.prepare4(self.iteration*8+self.micro*4);mask,kv=self.prefix(obs);self.loss=(self.model.velocity_from_suffix(self.model.run_suffix(obs,x,t,kv,mask))-u).square().mean();self.next_kind='T_BWD';return self.loss
 def backward(self):
  (self.loss/2).backward();self.loss=None;self.micro+=1;self.next_kind='T_OPT' if self.micro==2 else 'T_FWD'
 def measured_step(self):
  torch.cuda.reset_peak_memory_stats();start=time.perf_counter();totals=dict(preprocess_ms=0.,prefix_ms=0.,action_ms=0.,backward_ms=0.,optimizer_ms=0.)
  for j in range(2):
   (obs,x,u,t),row=timed('preprocess',lambda:self.prepare4(self.iteration*8+j*4));totals['preprocess_ms']+=row['cuda_ms']
   (mask,kv),row=timed('prefix',lambda:self.prefix(obs));totals['prefix_ms']+=row['cuda_ms']
   self.loss,row=timed('action',lambda:(self.model.velocity_from_suffix(self.model.run_suffix(obs,x,t,kv,mask))-u).square().mean());totals['action_ms']+=row['cuda_ms']
   _,row=timed('backward',self.backward);totals['backward_ms']+=row['cuda_ms'];del obs,x,u,t,mask,kv
  _,row=timed('optimizer',self.opt);totals['optimizer_ms']=row['cuda_ms'];totals.update(step=self.iteration,wall_ms=(time.perf_counter()-start)*1000,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());return totals
 def check_microbatch_equivalence(self):
  # Same eight observations, augmentation seeds, noise, times and initial weights.
  losses=[]
  for j in range(8):
   obs,x,u,t=super().prepare(j);mask,kv=self.prefix(obs);loss=(self.model.velocity_from_suffix(self.model.run_suffix(obs,x,t,kv,mask))-u).square().mean();losses.append(float(loss.detach()));(loss/8).backward()
  refs=[p.grad.detach().float().cpu() for _,p in self.pairs];self.model.zero_grad(set_to_none=True);newloss=[]
  for j in range(2):
   obs,x,u,t=self.prepare4(j*4);mask,kv=self.prefix(obs);loss=(self.model.velocity_from_suffix(self.model.run_suffix(obs,x,t,kv,mask))-u).square().mean();newloss.append(float(loss.detach()));(loss/2).backward()
  delta2=ref2=0.;maxabs=0.
  for (_,p),ref in zip(self.pairs,refs):
   v=p.grad.detach().float().cpu();delta2+=float((v-ref).square().sum());ref2+=float(ref.square().sum());maxabs=max(maxabs,float((v-ref).abs().max()))
  result={'microbatch1_loss':float(np.mean(losses)),'microbatch4_loss':float(np.mean(newloss)),'gradient_relative_l2':(delta2/max(ref2,1e-30))**.5,'gradient_max_abs':maxabs,'sample_count_each':8,'same_augmentations_and_noise':True}
  self.model.zero_grad(set_to_none=True);del refs,loss;gc.collect();torch.cuda.empty_cache();return result

def main():
 for vision in [False,True]:
  tier='action_vision_lora' if vision else 'action';out=OUT/tier;out.mkdir(exist_ok=True);save(OUT/'status.json',{'state':'loading','tier':tier,'gpu':GPU,'pid':os.getpid()});tr=TrainMB4(vision)
  cfg=json.load(open('/home/hengyi/pi05_measure_20261107/training/'+tier+'/config.json'));cfg.update(microbatch=4,accumulation=2,effective_batch=8,physical_gpu=GPU);save(out/'config.json',cfg)
  parity=tr.check_microbatch_equivalence();save(out/'microbatch_parity.json',parity)
  if parity['gradient_relative_l2']>.1:raise RuntimeError(str(parity))
  rows=[]
  for i in range(60):
   row=tr.measured_step();row['warmup']=i<10
   with (out/'raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if i>=10:rows.append(row)
   save(OUT/'status.json',{'state':'measuring','tier':tier,'step':i+1,'total':60,'last_step_s':row['wall_ms']/1000,'gpu':GPU})
  save(out/'summary.json',{k:stats([r[k] for r in rows]) for k in rows[0] if k not in ['step','warmup']});assert fingerprint((n,p) for n,p in tr.model.named_parameters() if not p.requires_grad)==tr.frozen;save(out/'integrity.json',{'frozen_parameters_unchanged':True});del tr;gc.collect();torch.cuda.empty_cache()
 save(OUT/'status.json',{'state':'complete'})
if __name__=='__main__':
 try:main()
 except Exception:save(OUT/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
