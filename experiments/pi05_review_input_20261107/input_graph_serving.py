import sys,types,time
sys.path.insert(0,'/home/hengyi/pi05_review_20261107')
from review_graph_serving import *
InstrumentedGraphServing=GraphServing
class GraphServing(InstrumentedGraphServing):
 def __init__(self,batch=1):
  super().__init__(batch);self.input_parts={}
  for name,label in [('input_transform','cpu_transform_wall_ms'),('_observation_dict_to_device','observation_to_device_wall_ms')]:
   original=getattr(self.model,name)
   def wrap(original,label):
    def measured(obj,*args,**kwargs):
     start=time.monotonic_ns();value=original(*args,**kwargs);self.input_parts[label]=(time.monotonic_ns()-start)/1e6;return value
    return measured
   setattr(self.model,name,types.MethodType(wrap(original,label),self.model))
 def infer(self,i):
  self.input_parts={};out,row=super().infer(i);row.update(self.input_parts);row['remaining_input_wall_ms']=row['input_wall_ms']-sum(self.input_parts.values());return out,row
