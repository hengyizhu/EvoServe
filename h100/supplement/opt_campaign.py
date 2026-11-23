import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/workspace/EvoServe')
CODE=Path(__file__).resolve().parent
OUT=ROOT/'results/supplement-20261122-200102/training-optimization'
PY=sys.executable
ENV=dict(os.environ,SUITE_HOME=str(ROOT/'.h100'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
         OPENBLAS_NUM_THREADS='1',EVOSERVE_CPU_THREADS='1',PYTHONUNBUFFERED='1',
         PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ('h100','prototype','vendor/RLinf')))


def status(stage,**info):
    q=dict(stage=stage,unix_time=time.time(),**info)
    p=OUT/'status.tmp';p.write_text(json.dumps(q,indent=2));p.replace(OUT/'status.json')
    print(q,flush=True)


def run(mode,n,threads,path):
    label=f'{mode}-m{n}-{path}-t{threads}';dst=OUT/label;dst.mkdir(exist_ok=True)
    if (dst/'trajectory.json').exists() and (dst/'timing.json').exists():return
    status('running',case=label)
    cmd=[PY,str(CODE/'opt_bench.py'),mode,str(n),str(threads),path,str(dst),'--both']
    with (dst/'run.log').open('a') as log:
        subprocess.run(cmd,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    for path in ('eager','packed','graph'):
        for mode in ('cache','recompute'):
            for n in (1,8):
                for threads in (1,4):
                    run(mode,n,threads,path)
    status('standalone matrix complete')

if __name__=='__main__':
    try:main()
    except BaseException as err:
        status('needs attention',detail=str(err))
        raise
