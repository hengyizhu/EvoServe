import sys,types,dataclasses,time
sys.path.insert(0,'/home/hengyi/pi05_review_20261107')
from review_graph_serving import *
from torch.utils._pytree import tree_flatten,tree_unflatten
from rlinf.models.embodiment.openpi.modules.model import Observation
PreviousGraphServing=GraphServing
class PackedObservation:
 """One aligned pinned byte slab, one nonblocking H2D, stable CUDA views."""
 def __init__(self):self.host=None;self.done=None;self.stats={}
 def transfer(self,processed):
  start=time.monotonic_ns();obs=Observation.from_dict(processed)
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
  packed=time.monotonic_ns();self.device.copy_(self.host,non_blocking=True);self.done.record();submitted=time.monotonic_ns()
  self.stats={'host_pack_wall_ms':(packed-start)/1e6,'h2d_enqueue_wall_ms':(submitted-packed)/1e6,'h2d_calls':1,'packed_bytes':offset}
  return type(obs)(**dict(zip(fields,tree_unflatten(gpu,spec))))
class GraphServing(PreviousGraphServing):
 def batch_input(self,i):
  if not hasattr(self,'observation_packer'):
   self.observation_packer=PackedObservation();original=self.model._observation_dict_to_device;self.packed_input_checked=False
   def packed(model,processed):
    result=self.observation_packer.transfer(processed)
    if not self.packed_input_checked:
     reference=original(processed)
     a=tree_flatten([getattr(result,f.name) for f in dataclasses.fields(result)])[0];b=tree_flatten([getattr(reference,f.name) for f in dataclasses.fields(reference)])[0]
     for x,y in zip(a,b):
      if torch.is_tensor(x):torch.testing.assert_close(x,y,rtol=0,atol=0)
     self.packed_input_checked=True
    return result
   self.model._observation_dict_to_device=types.MethodType(packed,self.model)
  return super().batch_input(i)
 def infer(self,i):
  out,row=super().infer(i);row.update(self.observation_packer.stats);return out,row
