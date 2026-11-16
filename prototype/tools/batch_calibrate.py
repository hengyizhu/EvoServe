"""Standalone measured batch latency and graph/slot/cache smoke; no learning claims."""
import argparse,sys,os,time,json,csv,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evoserve.io import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--out',required=True);p.add_argument('--inputs',required=True);p.add_argument('--operators',required=True);a=p.parse_args();os.sched_setaffinity(0,{0,1,2,3});out=Path(a.out);out.mkdir(parents=True)
 from evoserve.batch_backend import BatchedGraphModel
 from evoserve.operators import load_operator
 from evoserve.feature_ipc import Export,read_slot
 import numpy as np
 model=BatchedGraphModel(a.assets,[1,2,4,8]);t=model.torch;model.install(3,'3:v0');samples=pickle.loads(Path(a.inputs).read_bytes())
 rows=[];checks=[]
 for size in [1,2,4,8]:
  for i in range(60):
   messages=[dict(raw=samples[(i*size+j)%len(samples)]['raw'],expert=3,version='3:v0',seed=20261108+i*size+j) for j in range(size)]
   start=time.monotonic_ns();actions,prefix,row=model.infer_batch(messages);elapsed=(time.monotonic_ns()-start)/1e6
   assert actions.shape==(size,10,7) and np.isfinite(actions).all()
   assert all(q.shape[0]==size for q in model.leaves(prefix))
   if i>=10:rows.append(dict(batch=size,iteration=i-10,wall_ms=elapsed,prefix_ms=row['prefix_ms'],denoise_ms=row['denoise_ms'],peak_allocated_bytes=t.cuda.max_memory_allocated(),graph_vs_ordinary_batch_max_abs=row.get('graph_eager_max_abs_error',0)))
  save(out/'progress.json',dict(last_batch=size,measurements=len(rows)))
 # Mixed versions with different actual action weights, frozen common prefix.
 ops=json.loads(Path(a.operators).read_text())['operators'];delta,_=load_operator(ops['3']['checkpoint'],model.pairs,3)
 with t.cuda.stream(model.stream),t.inference_mode():model.slots[(3,'3:teacher')]={n:q.to(device='cuda',dtype=model.pairs[n].dtype) for n,q in delta.items()}
 model.stream.synchronize()
 for size in [2,4,8]:
  messages=[dict(raw=samples[j]['raw'],expert=3,version='3:teacher' if j%2 else '3:v0',seed=20261108+j) for j in range(size)]
  actions,prefix,row=model.infer_batch(messages);b=model.buckets[size]
  with t.cuda.stream(model.stream),t.inference_mode():
   eager=model.batch_work(b);decoded=model.model.decode_actions(eager,b['static'].state).cpu().numpy()
  np.testing.assert_allclose(actions,decoded,rtol=1e-6,atol=1e-6)
  source=out/f'cache_probe_b{size}';source.mkdir();sample_prefix=model.map(lambda x:x[:1],prefix);export=Export(source,sample_prefix)
  try:
   for i in range(size):
    expected=model.map(lambda x:x[i:i+1],prefix);record=export.publish(i,expected,dict(id=i,request_id=messages[i]['seed']));save(source/'ipc'/f'{i:08d}.json',record)
    copied=read_slot(source,record['slot']);(source/'ipc'/f'{i:08d}.json').unlink()
    for want,got in zip(model.leaves(expected),model.leaves(copied)):t.testing.assert_close(want.cpu(),got,rtol=0,atol=0)
  finally:export.close()
  checks.append(dict(batch=size,mixed_versions_graph_matches_eager=True,max_action_error=float(abs(actions-decoded).max()),per_request_export_exact=True))
 with (out/'batch_latency_raw.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for size in [1,2,4,8]:
  group=[r for r in rows if r['batch']==size];summary.append(dict(batch=size,median_ms=float(np.median([r['wall_ms'] for r in group])),p99_ms=float(np.percentile([r['wall_ms'] for r in group],99)),prefix_median_ms=float(np.median([r['prefix_ms'] for r in group])),denoise_median_ms=float(np.median([r['denoise_ms'] for r in group])),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in group),ordinary_batch_numeric_max_abs=max(r['graph_vs_ordinary_batch_max_abs'] for r in group)))
 cap=max([x['batch'] for x in summary if x['p99_ms']<250],default=1)
 save(out/'summary.json',dict(state='passed',batches=summary,mixed_version_checks=checks,selected_max_batch=cap,deadline_ms=250,power_limit_w=225,selection='largest calibrated bucket below deadline; actual capacity independently measured'))
 with (out/'batch_latency.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
if __name__=='__main__':main()
