"""One shared frozen VLM; graph buckets vectorize independent expert suffixes."""
import dataclasses,time
from .backend import GraphModel

class BatchedGraphModel(GraphModel):
 def __init__(self,assets,batch_sizes=(1,2,4,8)):
  super().__init__(assets);t=self.torch;owner=self
  class Suffix(t.nn.Module):
   def __init__(self):super().__init__();self.base=owner.model
   def forward(self,fields,noise,mask,kv):
    obs=owner.mod.Observation(**fields);x=noise
    for j in range(10):
     tt=t.full((1,),1-j/10,device='cuda',dtype=t.float32)
     x=x-.1*self.base.velocity_from_suffix(self.base.run_suffix(obs,x,tt,kv,mask))
    return x
  self.suffix=Suffix();self.buckets={};self.batch_sizes=sorted(set(batch_sizes));assert self.batch_sizes[0]==1
  for size in self.batch_sizes:
   if size==1:continue
   obs,_=self.prepare(self.dummy());static=self.cat_obs([self.obs_map(lambda x:x.to('cuda'),obs) for _ in range(size)])
   bucket=dict(size=size,static=static,noise=t.randn((size,10,32),device='cuda'),params={'base.'+n:t.stack([p.detach()]*size) for n,p in self.pairs.items()},events=[t.cuda.Event(enable_timing=True,external=True) for _ in range(3)],graph=t.cuda.CUDAGraph())
   self.stream.wait_stream(t.cuda.current_stream())
   with t.cuda.stream(self.stream),t.inference_mode():
    reference=self.model.sample_actions(static,num_steps=10,noise=bucket['noise']).clone()
    for _ in range(3):vectorized=self.batch_work(bucket)
    vectorized_reference=vectorized.clone()
   self.stream.synchronize()
   with t.cuda.graph(bucket['graph'],stream=self.stream),t.inference_mode():bucket['result']=self.batch_work(bucket)
   with t.cuda.stream(self.stream),t.inference_mode():bucket['graph'].replay()
   self.stream.synchronize();t.testing.assert_close(vectorized_reference,bucket['result'],rtol=1e-6,atol=1e-6)
   # Vectorized suffix changes GEMM shapes; record numerical differences, never silently correct.
   error=float((reference-bucket['result']).abs().max())
   if not t.isfinite(bucket['result']).all():raise ValueError('nonfinite batched graph')
   bucket['eager_max_abs_error']=error;self.buckets[size]=bucket
 def batch_work(self,b):
  t=self.torch;obs=b['static'];b['events'][0].record();_,mask,kv=self.model.build_prefix_cache(obs);b['prefix']=(mask,kv);b['events'][1].record()
  fields={f.name:self.map(lambda x:x[:,None] if t.is_tensor(x) else x,getattr(obs,f.name)) for f in dataclasses.fields(obs)}
  dims=self.map(lambda x:0 if t.is_tensor(x) else None,fields)
  def suffix(params,fields,noise,mask,kv):return t.func.functional_call(self.suffix,params,(fields,noise,mask,kv),strict=False)
  result=t.vmap(suffix,in_dims=(0,dims,0,0,0),randomness='error')(b['params'],fields,b['noise'][:,None],mask[:,None],self.map(lambda x:x[:,None],kv))[:,0]
  b['events'][2].record();return result
 def infer_batch(self,messages,prefix_hook=None):
  if len(messages)==1:
   m=messages[0];actions,row=self.infer(m['raw'],m['expert'],m['version'],m['seed'],prefix_hook=(lambda p,s,e:prefix_hook(p,s,e)) if prefix_hook else None)
   row.update(batch_size=1,graph_batch_size=1);return actions[None],self.prefix,row
  t=self.torch;start=time.monotonic_ns();n=len(messages);size=min(x for x in self.batch_sizes if x>=n);b=self.buckets[size]
  padded=messages+[messages[-1]]*(size-n);obs=self.cat_obs([self.prepare(m['raw'])[0] for m in padded])
  with t.cuda.stream(self.stream),t.inference_mode():
   for dst,src in zip(self.leaves({f.name:getattr(b['static'],f.name) for f in dataclasses.fields(obs)}),self.leaves({f.name:getattr(obs,f.name) for f in dataclasses.fields(obs)})):
    if t.is_tensor(dst):dst.copy_(src,non_blocking=True)
   for i,m in enumerate(padded):
    for name,dst in b['params'].items():dst[i].copy_(self.slots[(m['expert'],m['version'])][name[5:]])
    b['noise'][i:i+1].copy_(t.randn((1,10,32),device='cuda',generator=t.Generator(device='cuda').manual_seed(m['seed'])))
   b['graph'].replay()
   if prefix_hook:prefix_hook(b['prefix'],b['static'].state,b['events'][1])
   result=self.model.decode_actions(b['result'],b['static'].state).cpu()
  return result[:n].numpy(),b['prefix'],dict(start_ns=start,finish_ns=time.monotonic_ns(),batch_size=n,graph_batch_size=size,prefix_ms=b['events'][0].elapsed_time(b['events'][1]),denoise_ms=b['events'][1].elapsed_time(b['events'][2]),graph_eager_max_abs_error=b['eager_max_abs_error'])
