from pathlib import Path
import json
import torch
def profile_training(tr,out):
 out=Path(out);out.mkdir(exist_ok=True);tr.ingest()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False) as prof:
  for _ in range(3):
   row=tr.next_step();assert row is not None;prof.step()
 path=out/'training_trace.json';prof.export_chrome_trace(str(path));events=json.loads(path.read_text())['traceEvents'];kernels=[x for x in events if x.get('cat')=='kernel' and x.get('dur') is not None];assert kernels,'no CUDA kernel events captured'
 longest=sorted(kernels,key=lambda x:x['dur'],reverse=True)[:20];(out/'longest_kernel.json').write_text(json.dumps({'profiled_updates':3,'mode':tr.mode,'kernel_count':len(kernels),'longest_kernel_name':longest[0]['name'],'longest_kernel_us':longest[0]['dur'],'top_individual_kernels':[{'name':x['name'],'duration_us':x['dur']} for x in longest],'timing_excluded_from_SLO_windows':True},indent=2))
