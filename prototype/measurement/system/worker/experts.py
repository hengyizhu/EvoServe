"""Independent action adapters batched by vmap; shared frozen backbone, E x batch8."""
import os,time,math,types,dataclasses
from cache_engine import torch,np,event,save,OUT,cat_obs,tree_map,tree_flatten
class SuffixLoss(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,fields,actions,noise,t,mask,kv):
  from cache_engine import model_module
  obs=model_module.Observation(**fields);x=t[:,None,None]*noise+(1-t[:,None,None])*actions
  v=self.model.velocity_from_suffix(self.model.run_suffix(obs,x,t,kv,mask));return (v-(noise-actions)).square().mean()
class Bank:
 def __init__(self,tr,n):
  self.tr=tr;self.n=n;self.wrapper=SuffixLoss(tr.model);self.names=['model.'+name for name,_ in tr.pairs];self.dtypes=[p.dtype for _,p in tr.pairs]
  self.masters=[torch.nn.Parameter(p.detach().float().unsqueeze(0).repeat(n,*([1]*p.ndim))) for _,p in tr.pairs];self.ema=[p.detach().clone() for p in self.masters]
  self.optim=torch.optim.AdamW(self.masters,lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,foreach=True)
 def snapshot(self):return dict(masters=[p.detach().cpu().clone() for p in self.masters],ema=[p.detach().cpu().clone() for p in self.ema],optim=tree_map(lambda x:x.detach().cpu() if torch.is_tensor(x) else x,self.optim.state_dict()))
 def restore(self,state):
  self.optim.load_state_dict(state['optim'])
  with torch.no_grad():
   for p,e,x,y in zip(self.masters,self.ema,state['masters'],state['ema']):p.copy_(x);e.copy_(y)
  self.optim.zero_grad(set_to_none=True)
 def loss(self,ps):
  tr=self.tr;fields=[];acs=[];ns=[];ts=[];masks=[];kvs=[]
  for p in ps:
   tr.stream.wait_event(p['done']);assert all(x is not None for x in p['prefixes']),'missing serving prefix; refusing recomputation'
   obs=cat_obs([x[0] for x in p['parts']]);fields.append({f.name:getattr(obs,f.name) for f in dataclasses.fields(obs)})
   a,n,t=[torch.cat([x[k] for x in p['parts']],0) for k in [1,2,3]];acs.append(a);ns.append(n);ts.append(t)
   mask,kv=tree_map(lambda *v:torch.cat(v,0),*p['prefixes']);masks.append(mask);kvs.append(kv)
   for tensor in tree_flatten((p['parts'],p['prefixes']))[0]:
    if torch.is_tensor(tensor) and tensor.is_cuda:tensor.record_stream(tr.stream)
  stack=lambda *xs:torch.stack(xs,0) if torch.is_tensor(xs[0]) else xs[0]
  fields=tree_map(stack,*fields);kv=tree_map(stack,*kvs);mask=torch.stack(masks);params={k:m.to(dtype=d) for k,m,d in zip(self.names,self.masters,self.dtypes)}
  def loss_one(p,f,a,n,t,mask,kv):return torch.func.functional_call(self.wrapper,p,(f,a,n,t,mask,kv),strict=False)
  dims=tree_map(lambda x:0 if torch.is_tensor(x) else None,fields)
  return torch.vmap(loss_one,in_dims=(0,dims,0,0,0,0,0),randomness='error')(params,fields,torch.stack(acs),torch.stack(ns),torch.stack(ts),mask,kv)
 def validate(self):
  tr=self.tr;tr.copy.synchronize();ps=[tr.prepare(0) for _ in range(self.n)]
  values=self.loss(ps);assert values.shape==(self.n,) and bool(torch.isfinite(values).all());values[0].backward()
  own=sum(float(p.grad[0].square().sum()) for p in self.masters if p.grad is not None)
  other=sum(float(p.grad[1:].square().sum()) for p in self.masters if p.grad is not None) if self.n>1 else 0.
  assert own>0 and other==0.,(own,other);self.optim.zero_grad(set_to_none=True);tr.model.zero_grad(set_to_none=True)
  save(OUT/'expert_independence.json',dict(state='passed',experts=self.n,gradient_norm_squared_expert0=own,other_experts_gradient_squared=other,trainable_parameters_per_expert=sum(p.numel() for _,p in tr.pairs),independent_optimizer_moments=True,per_expert_gradient_clipping=True,shared_frozen_backbone=True));del values,ps;torch.cuda.empty_cache()
 def step(self):
  tr=self.tr;tr.ingest()
  if not tr.recent:return None
  start=time.monotonic_ns();new=list(tr.recent);batches=[];class_counts={k:dict(hits=0,misses=0) for k in ['new','replay']}
  for expert in range(self.n):
   keys=[new[int(tr.sample_rng.integers(len(new)))] for _ in range(4)]+[tr.replay[int(tr.sample_rng.integers(len(tr.replay)))] for _ in range(4)];tr.sample_rng.shuffle(keys)
   for k in keys:
    category='new' if len(k)==3 else 'replay';class_counts[category]['hits']+=1
    if category=='new':tr.recent[k]['sample_count']+=1
   mi=len(tr.micros);tr.micros.extend([keys[:4],keys[4:]]);batches.append([tr.prepare(mi),tr.prepare(mi+1)])
  losses=[]
  for micro in range(2):
   loss=self.loss([b[micro] for b in batches]);(loss.sum()/2).backward();losses.append(loss.detach())
  norms=torch.stack([p.grad.reshape(self.n,-1).square().sum(1) for p in self.masters]).sum(0).sqrt();assert bool(torch.isfinite(norms).all());scale=(1./(norms+1e-6)).clamp(max=1.)
  for p in self.masters:p.grad.mul_(scale.reshape(self.n,*([1]*(p.ndim-1))))
  step=tr.step_id+1;lr=tr.learning_rate*(step/50 if step<=50 else .1+.9*.5*(1+math.cos(math.pi*min(1.,(step-50)/(tr.total_steps-50)))));self.optim.param_groups[0]['lr']=lr;self.optim.step()
  with torch.no_grad():torch._foreach_mul_(self.ema,.999);torch._foreach_add_(self.ema,self.masters,alpha=.001)
  self.optim.zero_grad(set_to_none=True);done=event();done.record(tr.stream);done.synchronize();end=time.monotonic_ns();tr.step_id=step;tr.mi=len(tr.micros);tr.hits+=self.n*8
  ps=[p for b in batches for p in b];row=dict(step=step,start_ns=start,finish_ns=end,compute_finish_ns=end,seconds=(end-start)/1e9,loss=float(torch.stack(losses).mean()),expert_losses=torch.stack(losses).mean(0).cpu().tolist(),experts=self.n,expert_steps=self.n,effective_batch_per_expert=8,peak_cuda_bytes=torch.cuda.max_memory_allocated(),h2d_ms=sum(a.elapsed_time(b) for p in ps for a,b,_ in p['records']),class_counts=class_counts,prefix_compute_ms={'new':0.,'replay':0.},hits=self.n*8,misses=0,learning_rate=lr)
  return row
def install(tr,module):
 n=int(os.environ['EXPERT_COUNT']);tr.bank=Bank(tr,n)
 original_snapshot=module.snapshot;original_restore=module.restore
 module.snapshot=lambda t:t.bank.snapshot() if hasattr(t,'bank') else original_snapshot(t)
 module.restore=lambda t,s:t.bank.restore(s) if hasattr(t,'bank') else original_restore(t,s)
 tr.next_step=types.MethodType(lambda self:self.bank.step(),tr)
 return tr.bank
