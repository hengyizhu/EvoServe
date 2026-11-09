import traceback,gc,argparse
from core import *

def phase_pair(server,train,istage,tstage,index,trace=None):
 # Solo and pair use exactly the same model state, input and RNG; solo gradients are discarded.
 assert train.micro==0 and train.next_kind=='T_FWD'
 with torch.cuda.stream(server.stream),torch.inference_mode():
  obs,noise=server.input(index)
  prepared=None;prefix_prep_ms=0.
  if istage=='I_ACTION':
   prepared,prep=timed('PREP_I_PREFIX',lambda:server.prefix(obs));prefix_prep_ms=prep['wall_ms']
  finish()
 def inference():
  with torch.inference_mode():
   return server.prefix(obs) if istage=='I_PREFIX' else server.action(prepared,noise)
 prep_ms=0.
 if tstage=='T_BWD':
  with torch.cuda.stream(train.stream):_,r=timed('PREP_T_FWD',train.forward)
  prep_ms=r['wall_ms']
 def training():return train.forward() if tstage=='T_FWD' else train.backward()
 with torch.cuda.stream(server.stream):_,ia=timed(istage,inference)
 with torch.cuda.stream(train.stream):_,ta=timed(tstage,training)
 # End the solo graph, discard its gradients and restore cursor, without changing weights.
 with torch.cuda.stream(train.stream):
  if tstage=='T_FWD':train.backward()
  train.model.zero_grad(set_to_none=True);train.optimizer.zero_grad(set_to_none=True)
  train.micro=0;train.next_kind='T_FWD';finish()
  if tstage=='T_BWD':train.forward();finish()
 ia0=torch.cuda.Event(enable_timing=True);ia1=torch.cuda.Event(enable_timing=True)
 ta0=torch.cuda.Event(enable_timing=True);ta1=torch.cuda.Event(enable_timing=True)
 # All preparation is complete before either measured stream starts.
 begin=stamp()
 def launch_i():
  with torch.cuda.stream(server.stream):
   ia0.record()
   with torch.cuda.nvtx.range(istage):value=inference()
   ia1.record()
   return value
 def launch_t():
  with torch.cuda.stream(train.stream):
   ta0.record()
   with torch.cuda.nvtx.range(tstage):value=training()
   ta1.record()
   return value
 if index%2==0:iv=launch_i();tv=launch_t()
 else:tv=launch_t();iv=launch_i()
 ia1.synchronize();ta1.synchronize();end=stamp()
 inference_pair=ia0.elapsed_time(ia1);training_pair=ta0.elapsed_time(ta1)
 joint=(end-begin)/1e6
 # Finish the actual update, including the remaining accumulation, only after both streams finish.
 with torch.cuda.stream(train.stream):
  if tstage=='T_FWD':train.backward();finish()
 while train.micro!=0 or train.next_kind!='T_FWD':train.segment(False)
 return {'profile':train.tier,'inference_stage':istage,'training_stage':tstage,'repeat':index,
  'start_order':'inference_first' if index%2==0 else 'training_first',
  'inference_alone_ms':ia['cuda_ms'],'training_alone_ms':ta['cuda_ms'],'joint_ms':joint,
  'inference_pair_ms':inference_pair,'training_pair_ms':training_pair,
  'inference_slowdown':inference_pair/ia['cuda_ms'],'pair_gain':(ia['wall_ms']+ta['wall_ms'])/joint,
  'pair_gain_time_basis':'sum of isolated host wall costs divided by joint host wall cost',
  'joint_gpu_span_ms':max(ia0.elapsed_time(ia1),ia0.elapsed_time(ta1),ta0.elapsed_time(ia1),ta0.elapsed_time(ta1)),
  'inference_alone_wall_ms':ia['wall_ms'],'training_alone_wall_ms':ta['wall_ms'],
  'prefix_preparation_ms':prefix_prep_ms,'backward_preparation_ms':prep_ms,'preparation_excluded_from_pair':True,
  'pair_window_start_ns':begin,'pair_window_end_ns':end,
  'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--trace-only',action='store_true');args=ap.parse_args()
 out=ROOT/'gpu3_phases';out.mkdir(exist_ok=True)
 rows=[]
 try:
  save(out/'status.json',{'state':'loading','pid':os.getpid()})
  server=Serving();save(out/'inference_correctness.json',server.verify())
  for i in range(20):server.infer(i)
  for tier in ['L2','L3']:
   train=Training(tier)
   target=out/tier;target.mkdir(exist_ok=True)
   save(out/'status.json',{'state':'correctness','profile':tier})
   train.correctness(target)
   for _ in range(5*8):train.segment(False)
   train.reset()
   for istage in ['I_PREFIX','I_ACTION']:
    for tstage in ['T_FWD','T_BWD']:
     # Five unreported warm pairs, then 100 recorded trials, alternating launch order.
     for index in range(5):phase_pair(server,train,istage,tstage,index)
     for index in range(100):
      save(out/'status.json',{'state':'running','profile':tier,'inference_stage':istage,'training_stage':tstage,'completed':len(rows),'target':800})
      row=phase_pair(server,train,istage,tstage,index);rows.append(row)
      if index%10==9:write_csv(out/'phase_pairs.csv',rows)
     write_csv(out/'phase_pairs.csv',rows)
   # Profiler used only for separate representative intervals, never the trials above.
   try:
    from torch.profiler import profile,ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
     phase_pair(server,train,'I_PREFIX','T_FWD',0)
     phase_pair(server,train,'I_ACTION','T_BWD',1)
    prof.export_chrome_trace(str(out/f'{tier}_representative_trace.json'))
    save(target/'profiler_status.json',{'tool':'torch.profiler CUPTI Chrome trace','nsight_systems':'not found on PATH or standard install locations; no Nsight trace claimed','formal_trials_profiled':False,'state':'complete'})
   except Exception:
    save(target/'profiler_status.json',{'state':'failed','traceback':traceback.format_exc()})
   save(target/'correctness.json',{'production':server.unchanged(full=True),'frozen_training_unchanged':train.check_frozen()})
   del train;gc.collect();torch.cuda.empty_cache()
  save(out/'status.json',{'state':'complete','completed':len(rows)})
 except Exception:
  save(out/'status.json',{'state':'failed','completed':len(rows),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
