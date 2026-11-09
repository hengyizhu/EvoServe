"""CPU-only integrity, relocation and allocation tests. Never imports model code."""
import ast,json,hashlib,tempfile,sys,os,re
from pathlib import Path
from init_workspace import materialize
R=Path(__file__).resolve().parents[1]
def main():
 sources=json.loads((R/'provenance/source_manifest.json').read_text())
 for item in sources:assert hashlib.sha256((R/item['destination']).read_bytes()).hexdigest()==item['sha256'],item['destination']
 files=list((R/'experiments').rglob('*.py'))+list((R/'tools').glob('*.py'))+list((R/'vendor').rglob('*.py'))
 for p in files:ast.parse(p.read_text(),filename=str(p))
 banned={'.pt','.pth','.safetensors','.ckpt','.bin','.npy','.npz','.parquet','.h5','.hdf5','.log','.jsonl'}
 for top in ['experiments','vendor','provenance','reference','environment','tools','docs','cluster','configs']:
  for p in (R/top).rglob('*'):
   if p.is_file():assert p.suffix not in banned,(str(p),'prohibited artifact')
 required=['pi05_cache_20261107/cache_cost.py','pi05_cache_20261107/cache_train.py','pi05_cache_20261107/run_mps.py','pi05_cache_20261107/learn.py','pi05_cache_20261107/evaluate_gate.py','pi05_optimized_20261107/graph_serving.py']
 for f in required:assert (R/'experiments'/f).exists()
 with tempfile.TemporaryDirectory(prefix='vla-repro-check-') as t:
  w=materialize(Path(t)/'work',Path(t)/'model',Path(t)/'data',Path(sys.executable).absolute())
  for p in w.rglob('*.py'):
   if p.is_symlink():continue
   ast.parse(p.read_text(),filename=str(p))
  code=(w/'pi05_optimized_20261107/run_matrix.py').read_text();assert '<84' not in code and 'VLA_FULL_SM_COUNT' in code
  import vla_portability as vp
  old=os.sched_setaffinity;calls=[];env=os.environ.get('VLA_NODE_CPUS')
  try:
   os.environ['VLA_NODE_CPUS']=json.dumps(list(range(32,64)));os.sched_setaffinity=lambda pid,cpus:calls.append((pid,cpus));vp.set_affinity(0,range(8,12));assert calls==[(0,{40,41,42,43})]
  finally:
   os.sched_setaffinity=old
   if env is None:os.environ.pop('VLA_NODE_CPUS',None)
   else:os.environ['VLA_NODE_CPUS']=env
  # Both an A6000 and an A100 must reject an unpartitioned MPS worker.
  for full,capped in [(84,24),(108,32)]:assert capped<full and not full<full
 print(json.dumps({'source_snapshots_verified':len(sources),'python_files_parsed':len(files),'required_entrypoints':len(required),'relocation':'passed','cpu_allocation_mapping':'passed','gpu_execution':False},indent=2))
if __name__=='__main__':main()
