import argparse,traceback,gc
from graph_serving import *
R=Path('/home/hengyi/pi05_optimized_20261107');OUT=R/'inference';OUT.mkdir(exist_ok=True)
def main():
 results=[]
 for batch in [1,4,8,16]:
  save(OUT/'status.json',{'state':'capturing','batch':batch,'gpu':GPU,'pid':os.getpid()});start=time.monotonic();s=GraphServing(batch);capture=time.monotonic()-start;save(OUT/f'batch{batch}_correctness.json',s.verify_graph())
  save(OUT/'status.json',{'state':'measuring','batch':batch,'gpu':GPU})
  for i in range(10):s.infer(i)
  torch.cuda.reset_peak_memory_stats();rows=[]
  for i in range(100):
   _,row=s.infer(i);rows.append(row)
   with (OUT/f'batch{batch}_raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  old=next(x for x in json.load(open('/home/hengyi/pi05_measure_20261107/inference/summary.json')) if x['batch']==batch)
  summary={'batch':batch,'physical_gpu':GPU,'backend':'CUDA Graph full prefix + 10 denoise steps; external CUDA events','capture_and_load_seconds':capture,'warmup':10,'measured':100,'prefix_ms':stats([r['I_PREFIX_ms'] for r in rows]),'denoise_ms':stats([r['I_ACTION_ms'] for r in rows]),'denoise_each_median_ms':np.median([r['denoise_ms'] for r in rows],axis=0).tolist(),'end_to_end_ms':stats([r['service_ms'] for r in rows]),'model_ms':stats([r['model_ms'] for r in rows]),'prefix_flops':old['prefix_flops'],'one_denoise_flops':old['one_denoise_flops'],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
  summary['prefix_tflops']=summary['prefix_flops']/summary['prefix_ms']['p50']/1e9;summary['denoise_tflops']=10*summary['one_denoise_flops']/summary['denoise_ms']['p50']/1e9;summary['gap_ms']=250-summary['end_to_end_ms']['p50'];results.append(summary);save(OUT/'summary.json',results)
  save(OUT/f'batch{batch}_integrity.json',s.unchanged(full=True));del s;gc.collect();torch.cuda.empty_cache()
 save(OUT/'status.json',{'state':'complete'})
try:main()
except Exception:save(OUT/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
