"""Run one on-policy real-time experiment using physical GPUs 0 and/or 1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

proto=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--k',type=float,default=1)
p.add_argument('--robots-per-gpu',type=int,required=True)
p.add_argument('--experts',type=int,required=True)
p.add_argument('--gpus',type=int,nargs='+',choices=(0,1),required=True)
p.add_argument('--seconds',type=int)
p.add_argument('--calibration',type=Path,default=proto/'configs/a6000_r32_controller.json')
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.k<1 or len(set(a.gpus))!=len(a.gpus) or a.experts<len(a.gpus) or a.robots_per_gpu<1:p.error('invalid GPU/expert/robot layout or k')
machine=json.loads((proto/'configs/machine_a.example.json').read_text())
tasks=[s for s in json.loads((proto/'configs/accelerated.machine_a.json').read_text()) if s['name'] in ('camera','texture')]
ids=list(range(a.experts))
operators={};hashes={}
for e in ids:
    s=tasks[e%2];path=s['checkpoint']
    if path not in hashes:
        with open(path,'rb') as f:hashes[path]=hashlib.file_digest(f,'sha256').hexdigest()
    operators[str(e)]=dict(task_id=s['task_id'],checkpoint=path,sha256=hashes[path])
free={g:int(subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=memory.free','--format=csv,noheader,nounits'])) for g in a.gpus}
robots=[(g+(r%(len(ids[g::len(a.gpus)])))*len(a.gpus)) for r in range(a.robots_per_gpu) for g in range(len(a.gpus))]
calibration=json.loads(a.calibration.read_text())
cfg=dict(assets=machine['assets'],osmesa_lib=machine['osmesa_lib'],prior=str(proto/'configs/learning_prior.json'),
         gpus=a.gpus,operator_gpu=max(free,key=free.get),expert_ids=ids,robot_experts=robots,
         operators=operators,method='ours',paired=True,hg=True,system_metrics=True,feedforward=True,
         request_period_s=.25*a.k,seed=20261108,action_budget=520,
         changes=[dict(expert=e,at_s=0,perturb=tasks[e%2]['perturb']) for e in ids],
         safe_table=calibration["table"],service_ms=calibration["service_ms"],
         seconds=a.seconds or 86400*365,batch_sizes=([1,2,4] if a.robots_per_gpu>=2 else [1]),batch_wait_ms=(10 if a.robots_per_gpu>=2 else 0))
env=dict(os.environ,PYTHONPATH=str(proto),MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='1',GALLIUM_DRIVER='llvmpipe',
         LIBERO_CONFIG_PATH=cfg['assets']+'/libero_config',TOKENIZER_DIR=cfg['assets']+'/assets/.cache/openpi',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with tempfile.NamedTemporaryFile(mode='w',suffix='.json') as f:
    json.dump(cfg,f);f.flush()
    sys.exit(subprocess.call([sys.executable,'-u','-m','evoserve.runtime','--config',f.name,'--out',str(a.out.resolve())],env=env))
