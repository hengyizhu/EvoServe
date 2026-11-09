import sys,os,time,json,traceback
from pathlib import Path
sys.path.insert(0,'/home/hengyi/vla_overnight_20261104')
from core import *
OUT=Path('/home/hengyi/pi05_measure_20261107/inference');OUT.mkdir(exist_ok=True)
def main():
 save(OUT/'status.json',{'state':'loading','gpu':GPU,'pid':os.getpid()})
 s=Serving();s.model.num_steps=10
 save(OUT/'correctness.json',s.verify())
 results=[]
 for batch in [1,4,8,16]:
  save(OUT/'status.json',{'state':'measuring','batch':batch,'gpu':GPU})
  def inputs(i):
   ids=np.arange(i,i+batch)%len(s.corpus['states']);c=s.corpus
   obs=s.model.env_obs_to_observation({'main_images':c['images'][ids],'wrist_images':c['wrists'][ids],'states':c['states'][ids],'task_descriptions':[str(c['prompts'][j]) for j in ids]})
   noise=torch.randn((batch,s.model.action_horizon,s.model.action_dim),device='cuda',generator=torch.Generator(device='cuda').manual_seed(71000+i))
   return obs,noise
  def trial(i,split=True):
   start=time.perf_counter();obs,x=inputs(i);ev=[torch.cuda.Event(enable_timing=True) for _ in range(12)]
   ev[0].record();p=s.prefix(obs);ev[1].record()
   for j in range(10):
    tt=torch.full((batch,),1-j/10,device='cuda',dtype=torch.float32)
    x=x-.1*s.model.velocity_from_suffix(s.model.run_suffix(p[0],x,tt,p[2],p[1]));ev[j+2].record()
   out=s.model.decode_actions(x,obs.state).cpu();torch.cuda.synchronize()
   return {'index':i,'batch':batch,'prefix_ms':ev[0].elapsed_time(ev[1]),'denoise_ms':[ev[j+1].elapsed_time(ev[j+2]) for j in range(10)],'model_ms':ev[0].elapsed_time(ev[-1]),'end_to_end_ms':(time.perf_counter()-start)*1000,'finite':bool(torch.isfinite(out).all())}
  try:
   with torch.inference_mode():
    for i in range(10):trial(i)
    torch.cuda.reset_peak_memory_stats();rows=[]
    for i in range(50):
     row=trial(i);rows.append(row)
     with (OUT/f'batch{batch}_raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    peak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
    from torch.utils.flop_counter import FlopCounterMode
    obs,noise=inputs(0)
    with FlopCounterMode(display=False) as fc:p=s.prefix(obs)
    prefix_flops=fc.get_total_flops()
    with FlopCounterMode(display=False) as fc:s.model.velocity_from_suffix(s.model.run_suffix(p[0],noise,torch.ones(batch,device='cuda'),p[2],p[1]))
    suffix_flops=fc.get_total_flops()
    row={'batch':batch,'samples':50,'warmup':10,'prefix_ms':stats([r['prefix_ms'] for r in rows]),'denoise_each_median_ms':np.median([r['denoise_ms'] for r in rows],axis=0).tolist(),'denoise_total_ms':stats([sum(r['denoise_ms']) for r in rows]),'end_to_end_ms':stats([r['end_to_end_ms'] for r in rows]),'model_ms':stats([r['model_ms'] for r in rows]),'peak_bytes':peak,'prefix_flops':prefix_flops,'one_denoise_flops':suffix_flops,'flops_method':'PyTorch FlopCounterMode supported operator accounting; multiply-add=2; excludes unsupported/elementwise ops','physical_gpu':GPU}
    row['prefix_tflops']=prefix_flops/row['prefix_ms']['p50']/1e9
    row['denoise_tflops']=suffix_flops/float(np.median(row['denoise_each_median_ms']))/1e9
    results.append(row);save(OUT/'summary.json',results)
    if batch==1:
     try:
      with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:trial(0)
      prof.export_chrome_trace(str(OUT/'batch1_trace.json'))
     except Exception:save(OUT/'trace_error.json',{'traceback':traceback.format_exc()})
  except torch.cuda.OutOfMemoryError:
   results.append({'batch':batch,'status':'OOM','traceback':traceback.format_exc()});save(OUT/'summary.json',results);torch.cuda.empty_cache()
 save(OUT/'integrity.json',s.unchanged(full=True));save(OUT/'status.json',{'state':'complete','gpu':GPU})
if __name__=='__main__':
 try:main()
 except Exception:save(OUT/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
