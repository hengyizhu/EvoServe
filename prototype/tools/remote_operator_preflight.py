"""Actual early KV IPC across GPU0/1; production and operator action parity smoke."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evoserve.io import save,wait_file,read
from evoserve.operators import operator_token,wait_operator_result

def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--operators',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 os.sched_setaffinity(0,{0,1,2,3});out=Path(a.out);out.mkdir(parents=True);source=out/'export';source.mkdir();op=out/'operator1'
 from evoserve.backend import GraphModel
 from evoserve.feature_ipc import Export
 graph=GraphModel(a.assets);graph.install(3,'3:v0');operators={'3':read(a.operators)['operators']['3']};graph.enable_operators(operators)
 raw=graph.dummy();seed=20261108;production,_=graph.infer(raw,3,'3:v0',seed);reference,_=graph.infer_operator(3,seed+7919)
 exporter=Export(source,(*graph.prefix,graph.static.state))
 cfg=dict(assets=a.assets,out=str(op),gpu=1,cpus=[28,29],operators=operators,sources=[str(source)])
 save(out/'operator_config.json',cfg);env=dict(os.environ,CUDA_VISIBLE_DEVICES='1')
 for k in list(env):
  if k.startswith('CUDA_MPS_'):env.pop(k)
 rows=[]
 with (out/'operator.log').open('w') as log:
  child=subprocess.Popen([sys.executable,'-u','-m','evoserve.operator_worker','--config',str(out/'operator_config.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  wait_file(op/'ready.json',[child])
  for i in range(12):
   released=time.monotonic();rid=f'probe:{i}';meta={}
   def early(prefix,state,event):
    rec=dict(id=i,expert=3,seed=seed,request_id=rid,episode='probe',frame=i*5,version='3:v0',deadline=released+.25)
    meta.update(exporter.publish(i,(*prefix,state),rec,ready_event=event));save(source/'ipc'/f'{i:08d}.json',meta)
   actions,timing=graph.infer(raw,3,'3:v0',seed,prefix_hook=early);production_finished=time.monotonic()
   result=wait_operator_result(dict(path=str(op/'results'/(operator_token(str(source),i)+'.json')),request_id=rid),released+2)
   assert result['status']=='complete',result
   graph.np.testing.assert_array_equal(actions,production)
   graph.np.testing.assert_allclose(result['actions'],reference,rtol=1e-5,atol=1e-5)
   rows.append(dict(request=i,production_ms=(production_finished-released)*1000,operator_ms=(result['finished']-released)*1000,operator_started_before_production_finished=result['started']<production_finished,operator_gpu_ms=result['operator_gpu_ms'],max_action_error=float(abs(graph.np.asarray(result['actions'])-reference).max()),export=meta))
   time.sleep(max(0,released+.25-time.monotonic()))
  save(out/'requests.json',rows)
  assert all(r['production_ms']<250 and r['operator_ms']<250 for r in rows[2:]),'cross-GPU deadline smoke failed'
  assert any(r['operator_started_before_production_finished'] for r in rows),'no early-prefix overlap observed'
  save(out/'summary.json',dict(state='passed',production_gpu=0,operator_gpu=1,production_unchanged=True,operator_action_parity=True,overlapped_requests=sum(r['operator_started_before_production_finished'] for r in rows),measured_requests=len(rows),maximum_production_ms=max(r['production_ms'] for r in rows),maximum_operator_ms=max(r['operator_ms'] for r in rows),smoke_only=True))
 finally:
  child.terminate();child.wait(timeout=30);exporter.close()
if __name__=='__main__':main()
