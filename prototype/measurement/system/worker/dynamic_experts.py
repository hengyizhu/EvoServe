"""Eight persistent experts, independently optimized; vmap only selected experts."""
import copy, math, os, time, types
from experts import Bank as PackedBank, SuffixLoss
from cache_engine import torch, np, event, save, OUT, tree_map

class Bank(PackedBank):
 def __init__(self,tr,n=8):
  self.tr=tr;self.total=n;self.n=int(os.environ.get('INITIAL_EXPERTS','1'));self.active=list(range(self.n));self.cursor=0
  self.wrapper=SuffixLoss(tr.model);self.names=['model.'+name for name,_ in tr.pairs];self.dtypes=[p.dtype for _,p in tr.pairs]
  self.weights=[[torch.nn.Parameter(p.detach().float().clone()) for _,p in tr.pairs] for _ in range(n)]
  self.emas=[[p.detach().clone() for p in ps] for ps in self.weights]
  self.optims=[torch.optim.AdamW(ps,lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,foreach=True) for ps in self.weights]
  self.counts=[0]*n;self.rngs=[np.random.default_rng(20261108+i) for i in range(n)]
 @property
 def masters(self):
  return [torch.stack([self.weights[e][j] for e in self.active]) for j in range(len(self.names))]
 def snapshot(self):
  cpu=lambda x:x.detach().cpu().clone() if torch.is_tensor(x) else copy.deepcopy(x)
  return dict(weights=tree_map(cpu,self.weights),emas=tree_map(cpu,self.emas),optims=[tree_map(cpu,o.state_dict()) for o in self.optims],counts=list(self.counts),cursor=self.cursor,rngs=[copy.deepcopy(r.bit_generator.state) for r in self.rngs])
 def restore(self,s):
  with torch.no_grad():
   for e in range(self.total):
    for p,q in zip(self.weights[e],s['weights'][e]):p.copy_(q)
    for p,q in zip(self.emas[e],s['emas'][e]):p.copy_(q)
    self.optims[e].load_state_dict(s['optims'][e]);self.optims[e].zero_grad(set_to_none=True);self.rngs[e].bit_generator.state=copy.deepcopy(s['rngs'][e])
  self.counts=list(s['counts']);self.cursor=s['cursor'];self.tr.stream.synchronize()
 def optimize(self):
  lrs=[]
  for e in self.active:
   torch.nn.utils.clip_grad_norm_(self.weights[e],1.,error_if_nonfinite=True,foreach=True)
   step=self.counts[e]+1;lr=self.tr.learning_rate*(step/50 if step<=50 else .1+.9*.5*(1+math.cos(math.pi*min(1.,(step-50)/(self.tr.total_steps-50)))))
   self.optims[e].param_groups[0]['lr']=lr;self.optims[e].step()
   with torch.no_grad():torch._foreach_mul_(self.emas[e],.999);torch._foreach_add_(self.emas[e],self.weights[e],alpha=.001)
   self.optims[e].zero_grad(set_to_none=True);self.counts[e]=step;lrs.append(lr)
  return lrs
 def validate(self):
  initial=self.snapshot();checks=[];tr=self.tr;tr.copy.synchronize()
  # Warm every supported merged count, then restore all weights/moments/EMA/counters.
  for n in [1,2,4,8]:
   self.n=n;self.active=list(range(n));ps=[tr.prepare(0) for _ in range(n)];loss=self.loss(ps)
   assert loss.shape==(n,) and bool(torch.isfinite(loss).all())
   (loss.sum()/2).backward()
   assert all(p.grad is None for e in range(n,self.total) for p in self.weights[e])
   before=list(self.counts);self.optimize();tr.stream.synchronize()
   assert self.counts==[x+int(e<n) for e,x in enumerate(before)]
   for e in range(n,self.total):
    assert all(torch.equal(p,q.to(p.device)) for p,q in zip(self.weights[e],initial['weights'][e]))
    assert all(torch.equal(p,q.to(p.device)) for p,q in zip(self.emas[e],initial['emas'][e]))
    assert len(self.optims[e].state)==0
   checks.append(dict(count=n,finite_loss=True,inactive_weights_ema_moments_unchanged=True,independent_optimizer_counters=list(self.counts)))
   del loss,ps;self.restore(initial)
  self.n=int(os.environ.get('INITIAL_EXPERTS','1'));self.active=list(range(self.n));tr.model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
  save(OUT/'expert_independence.json',dict(state='passed',total_experts=self.total,checks=checks,initial_states_restored=True,parameters_per_expert=sum(p.numel() for _,p in tr.pairs)))
 def step(self):
  tr=self.tr;tr.ingest()
  if not tr.recent:return None
  start=time.monotonic_ns();self.active=[(self.cursor+i)%self.total for i in range(self.n)];self.cursor=(self.cursor+self.n)%self.total
  new=list(tr.recent);batches=[];original_seed=tr.seed
  for e in self.active:
   rng=self.rngs[e];keys=[new[int(rng.integers(len(new)))] for _ in range(4)]+[tr.replay[int(rng.integers(len(tr.replay)))] for _ in range(4)];rng.shuffle(keys)
   for k in keys:
    if len(k)==3:tr.recent[k]['sample_count']+=1
   mi=len(tr.micros);tr.micros.extend([keys[:4],keys[4:]])
   tr.seed=original_seed+e*100000000+self.counts[e]*8-mi*4
   batches.append([tr.prepare(mi),tr.prepare(mi+1)])
  tr.seed=original_seed;losses=[]
  for micro in range(2):
   loss=self.loss([b[micro] for b in batches]);(loss.sum()/2).backward();losses.append(loss.detach())
  lrs=self.optimize();done=event();done.record(tr.stream);done.synchronize();end=time.monotonic_ns();tr.step_id+=1;tr.mi=len(tr.micros);tr.hits+=self.n*8
  ps=[p for b in batches for p in b]
  return dict(step=tr.step_id,start_ns=start,finish_ns=end,compute_finish_ns=end,seconds=(end-start)/1e9,loss=float(torch.stack(losses).mean()),expert_losses=torch.stack(losses).mean(0).cpu().tolist(),experts=self.n,expert_ids=list(self.active),expert_steps=self.n,optimizer_steps=list(self.counts),ema_steps=list(self.counts),effective_batch_per_expert=8,peak_cuda_bytes=torch.cuda.max_memory_allocated(),h2d_ms=sum(a.elapsed_time(b) for p in ps for a,b,_ in p['records']),class_counts={c:dict(hits=self.n*4,misses=0) for c in ['new','replay']},prefix_compute_ms={'new':0.,'replay':0.},hits=self.n*8,misses=0,learning_rates=lrs)

def install(tr,module):
 tr.bank=Bank(tr,8);module.snapshot=lambda t:t.bank.snapshot();module.restore=lambda t,s:t.bank.restore(s)
 tr.next_step=types.MethodType(lambda self:self.bank.step(),tr);return tr.bank
