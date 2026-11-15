import dataclasses,time,torch
from torch.utils._pytree import tree_flatten,tree_unflatten
class PackedObservation:
 """One aligned pinned byte slab, one nonblocking H2D, stable CUDA views."""
 def __init__(self):self.host=None;self.done=None;self.stats={}
 def transfer(self,processed):
  start=time.monotonic_ns();obs=processed
  obs=dataclasses.replace(obs,state=obs.state.to(dtype=torch.float32))
  fields=[f.name for f in dataclasses.fields(obs)];values=[getattr(obs,k) for k in fields];leaves,spec=tree_flatten(values)
  layout=[];offset=0
  for x in leaves:
   if torch.is_tensor(x):
    assert x.device.type=='cpu';offset=(offset+63)//64*64;span=(1+sum((d-1)*st for d,st in zip(x.shape,x.stride()))) if x.numel() else 0;n=span*x.element_size();layout.append((offset,n,x.dtype,tuple(x.shape),tuple(x.stride())));offset+=n
   else:layout.append(None)
  if self.host is None:
   self.layout=layout;self.host=torch.empty(offset,dtype=torch.uint8,pin_memory=True);self.device=torch.empty(offset,dtype=torch.uint8,device='cuda');self.done=torch.cuda.Event()
  else:
   assert layout==self.layout,'Observation structure changed';self.done.synchronize() # previous DMA ownership only, not per tensor
  gpu=[]
  for x,item in zip(leaves,layout):
   if item is None:gpu.append(x);continue
   at,n,dtype,shape,stride=item;self.host[at:at+n].view(dtype).as_strided(shape,stride).copy_(x);gpu.append(self.device[at:at+n].view(dtype).as_strided(shape,stride))
  packed=time.monotonic_ns();self.h2d_start=torch.cuda.Event(enable_timing=True);self.h2d_end=torch.cuda.Event(enable_timing=True);self.h2d_start.record();self.device.copy_(self.host,non_blocking=True);self.h2d_end.record();self.done.record();submitted=time.monotonic_ns()
  self.stats={'host_pack_wall_ms':(packed-start)/1e6,'h2d_enqueue_wall_ms':(submitted-packed)/1e6,'h2d_calls':1,'packed_bytes':offset}
  return type(obs)(**dict(zip(fields,tree_unflatten(gpu,spec))))
