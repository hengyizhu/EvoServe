import sys
sys.path.insert(0,'/home/hengyi/pi05_optimized_20261107')
from graph_serving import *
BaseGraphServing=GraphServing
class GraphServing(BaseGraphServing):
 def infer(self,i):
  with torch.cuda.stream(self.stream),torch.inference_mode():
   start=stamp();obs,noise=self.batch_input(i);input_done=stamp()
   self.copy_input(obs,noise);copy_submit=stamp();self.graph.replay();replay_submit=stamp()
   out=self.model.decode_actions(self.static_result,obs.state).cpu();self.events[-1].synchronize();end=stamp()
   prefix=self.events[0].elapsed_time(self.events[1]);action=self.events[1].elapsed_time(self.events[-1]);wall=(end-start)/1e6
   return out,{'start_ns':start,'finish_ns':end,'I_PREFIX_ms':prefix,'I_ACTION_ms':action,'denoise_ms':[self.events[j+1].elapsed_time(self.events[j+2]) for j in range(10)],'model_ms':prefix+action,'service_ms':wall,'input_wall_ms':(input_done-start)/1e6,'copy_submit_wall_ms':(copy_submit-input_done)/1e6,'graph_submit_wall_ms':(replay_submit-copy_submit)/1e6,'decode_wait_wall_ms':(end-replay_submit)/1e6,'outside_graph_ms':wall-prefix-action,'finite':bool(torch.isfinite(out).all())}
