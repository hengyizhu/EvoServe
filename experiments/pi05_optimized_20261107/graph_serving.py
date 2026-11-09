import sys,os,time,dataclasses,inspect,textwrap,types
from pathlib import Path
sys.path.insert(0,'/home/hengyi/vla_overnight_20261104')
from core import *
from torch.utils._pytree import tree_leaves
class GraphServing(Serving):
 def __init__(self,batch=1):
  super().__init__();self.model.num_steps=10;self.batch=batch;self.static_obs,self.static_noise=self.batch_input(0);self.events=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(12)]
  self.eager_reference={}
  with torch.inference_mode():
   for i in [0,1,17,63]:
    obs,noise=self.batch_input(i);self.eager_reference[i]=self.model.sample_actions(obs,num_steps=10,noise=noise).clone()
  def substitute(module,name,old,new):
   original=getattr(module,name);code=textwrap.dedent(inspect.getsource(original));assert old in code;code=code.replace(old,new);ns=dict(original.__func__.__globals__);exec(compile(code,'<capture_safe_constants>','exec'),ns);setattr(module,name,types.MethodType(ns[name],module))
  substitute(self.model,'embed_prefix','torch.tensor(ar_mask, device=tokens.device)','torch.zeros(len(ar_mask), dtype=torch.bool, device=tokens.device)')
  substitute(self.model,'embed_suffix','ar_mask[0] = True','ar_mask[:1].fill_(True)')
  for layer in self.model.llm.layers:
   substitute(layer.attn,'forward','torch.tensor(big_neg, dtype=logits.dtype, device=logits.device)','torch.full((), big_neg, dtype=logits.dtype, device=logits.device)')
  self.graph=torch.cuda.CUDAGraph();self.stream.wait_stream(torch.cuda.current_stream());torch.cuda.synchronize()
  with torch.cuda.stream(self.stream),torch.inference_mode():
   for _ in range(3):self.work()
  torch.cuda.synchronize()
  with torch.cuda.graph(self.graph,stream=self.stream),torch.inference_mode():self.static_result=self.work()
  torch.cuda.synchronize()
 def batch_input(self,i):
  c=self.corpus;ids=np.arange(i,i+self.batch)%len(c['states'])
  obs=self.model.env_obs_to_observation({'main_images':c['images'][ids],'wrist_images':c['wrists'][ids],'states':c['states'][ids],'task_descriptions':[str(c['prompts'][j]) for j in ids]})
  noise=torch.randn((self.batch,self.model.action_horizon,self.model.action_dim),device='cuda',generator=torch.Generator(device='cuda').manual_seed(71000+i));return obs,noise
 def work(self):
  self.events[0].record();prepared=self.prefix(self.static_obs);self.events[1].record();x=self.static_noise
  for j in range(10):
   t=torch.full((self.batch,),1-j/10,device='cuda',dtype=torch.float32)
   x=x-.1*self.model.velocity_from_suffix(self.model.run_suffix(prepared[0],x,t,prepared[2],prepared[1]));self.events[j+2].record()
  return x
 def copy_input(self,obs,noise):
  for a,b in zip(tree_leaves([getattr(self.static_obs,f.name) for f in dataclasses.fields(self.static_obs)]),tree_leaves([getattr(obs,f.name) for f in dataclasses.fields(obs)])):
   if torch.is_tensor(a):a.copy_(b)
  self.static_noise.copy_(noise)
 def infer(self,i):
  with torch.cuda.stream(self.stream),torch.inference_mode():
   start=stamp();obs,noise=self.batch_input(i);self.copy_input(obs,noise);self.graph.replay();raw=self.static_result
   out=self.model.decode_actions(raw,obs.state).cpu();self.events[-1].synchronize();end=stamp()
   rows={'start_ns':start,'finish_ns':end,'I_PREFIX_ms':self.events[0].elapsed_time(self.events[1]),'I_ACTION_ms':self.events[1].elapsed_time(self.events[-1]),'denoise_ms':[self.events[j+1].elapsed_time(self.events[j+2]) for j in range(10)],'model_ms':self.events[0].elapsed_time(self.events[-1]),'service_ms':(end-start)/1e6,'finite':bool(torch.isfinite(out).all())}
   return out,rows
 def verify_graph(self):
  rows=[]
  for i in [0,1,17,63]:
   with torch.cuda.stream(self.stream),torch.inference_mode():
    obs,noise=self.batch_input(i);native=self.eager_reference[i];self.copy_input(obs,noise);self.graph.replay();finish()
    delta=float((native-self.static_result).abs().max());torch.testing.assert_close(native,self.static_result,atol=1e-6,rtol=1e-6);rows.append({'input':i,'batch':self.batch,'raw_action_max_abs':delta})
  self.reference=self.infer(0)[0];return rows
