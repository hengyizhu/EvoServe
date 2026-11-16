"""Bounded shared-memory ring. Metadata is published only after D2H + host copy.
A slot is reusable only after receiver has cloned it into owned pinned storage.
"""
import json,time,pickle,os,hashlib
from pathlib import Path
import torch
from torch.utils._pytree import tree_flatten,tree_unflatten
class Export:
 def __init__(self,out,values):
  self.out=out;self.meta=out/'ipc';self.meta.mkdir(exist_ok=True);self.path=Path('/dev/shm')/('evoserve_prototype_'+str(os.getpid())+'_'+hashlib.sha256(str(out).encode()).hexdigest()[:12])
  xs,self.spec=tree_flatten(values);self.layout=[];offset=0
  for x in xs:
   assert torch.is_tensor(x);offset=(offset+63)//64*64;n=x.numel()*x.element_size();self.layout.append((offset,n,x.dtype,tuple(x.shape)));offset+=n
  self.size=(offset+63)//64*64;self.ring=torch.from_file(str(self.path),shared=True,size=self.size*16,dtype=torch.uint8);self.stage=torch.empty(self.size,dtype=torch.uint8,pin_memory=True);self.stream=torch.cuda.Stream(priority=0);self.owner={}
  (out/'ipc_layout.pkl').write_bytes(pickle.dumps((str(self.path),self.size,self.layout,self.spec)))
 def publish(self,ident,values,record,ready_event=None):
  slot=ident%16;old=self.owner.get(slot);start=time.monotonic_ns()
  while old is not None and (self.meta/f'{old:08d}.json').exists():
   if time.monotonic_ns()-start>60e9:raise TimeoutError('cache receiver backpressure')
   time.sleep(.001)
  queue_end=time.monotonic_ns()
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
  with torch.cuda.stream(self.stream):
   if ready_event is not None:self.stream.wait_event(ready_event)
   a.record()
   for x,(at,n,dtype,shape) in zip(tree_flatten(values)[0],self.layout):self.stage[at:at+n].view(dtype).view(shape).copy_(x,non_blocking=True)
   b.record()
  b.synchronize();copy_start=time.monotonic_ns();self.ring[slot*self.size:(slot+1)*self.size].copy_(self.stage);copy_end=time.monotonic_ns()
  self.owner[slot]=ident
  record.update(export_queue_wait_ms=(queue_end-start)/1e6,export_host_copy_ms=(copy_end-copy_start)/1e6,export_bytes=self.size,slot=slot,export_d2h_ms=a.elapsed_time(b),export_wall_ms=(time.monotonic_ns()-start)/1e6)
  return record
 def close(self):self.path.unlink(missing_ok=True)
_readers={}
def read_slot(out,slot):
 key=str(out)
 if key not in _readers:
  path,size,layout,spec=pickle.loads((out/'ipc_layout.pkl').read_bytes());ring=torch.from_file(path,shared=True,size=size*16,dtype=torch.uint8);_readers[key]=(size,layout,spec,ring)
 size,layout,spec,ring=_readers[key];host=torch.empty(size,dtype=torch.uint8,pin_memory=True);host.copy_(ring[slot*size:(slot+1)*size]);return tree_unflatten([host[at:at+n].view(dtype).view(shape) for at,n,dtype,shape in layout],spec)
