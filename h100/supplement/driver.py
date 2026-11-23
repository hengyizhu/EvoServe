"""Run supplemental tests sequentially after the first suite. Resume from finished cases."""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

from report import put,refresh

CODE=Path(__file__).resolve().parent
ROOT=Path('/workspace/EvoServe')
BASE=ROOT/'results/h100-20261122-200102'
OUT=ROOT/'results/supplement-20261122-200102'
MATRIX=[(f'cache-r{r}-m{n}-{v}','cache_bench.py',[str(r),str(n),v])
        for r in (32,128) for n in (1,8) for v in ('original','prefetch','resident')]
for k in (7,5,4,6):
    if k==7:
        MATRIX.append(('robots-7-inference-m1','robots.py',['7','inference','1']))
        continue
    for mode,n in [('inference',1),('cache',1),('cache',2),('cache',4),('cache',8),('recompute',1)]:
        MATRIX.append((f'robots-{k}-{mode}-m{n}','robots.py',[str(k),mode,str(n)]))
    if k==5:
        MATRIX.append((f'robots-{k}-slack','robots.py',[str(k),'slack','1']))
        MATRIX.append((f'robots-{k}-export','robots.py',[str(k),'export','1']))
    if k==4:
        MATRIX.append(('robots-4-sync-slack','robots.py',['4','slack','1','--alignment','sync']))


def status(stage,**info):
    payload=dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**info)
    put(OUT/'status.json',json.dumps(payload,indent=2))
    print(payload,flush=True)


@contextlib.contextmanager
def transport(env,use_mps):
    env['EVOSERVE_TRANSPORT']='mps' if use_mps else 'streams'
    if not use_mps:
        yield
        return
    with tempfile.TemporaryDirectory(prefix='h100-supplement-mps-') as d:
        env.update(CUDA_MPS_PIPE_DIRECTORY=d,CUDA_MPS_LOG_DIRECTORY=d,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100')
        subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True,timeout=20)
        try:
            yield
        finally:
            subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,env=env,timeout=30)


def case(name,script,args):
    dst=OUT/name;dst.mkdir(exist_ok=True)
    if (dst/'result.json').exists():
        return
    if (dst/'run.log').exists():
        prev=dst/('run-'+str(time.time_ns())+'.log')
        (dst/'run.log').replace(prev)
    env=dict(os.environ)
    env.update(SUITE_HOME=str(ROOT/'.h100'),EVOSERVE_CPU_THREADS='1',OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',
               PYTHONPATH=os.pathsep.join([str(ROOT/'h100'),str(ROOT/'prototype'),str(ROOT/'vendor/RLinf')]))
    cmd=[sys.executable,str(CODE/script),*args,str(dst)]
    status('running',case=name)
    with (dst/'run.log').open('w') as log,transport(env,script=='robots.py' and args[1] in ('cache','slack')):
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc=proc.wait(timeout=2700)
        except BaseException:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
            raise
    if rc or not (dst/'result.json').exists():
        raise RuntimeError(f'{name} exited {rc}; see {dst}/run.log')
    refresh(OUT)


def package():
    complete=refresh(OUT)
    assert set(complete)=={row[0] for row in MATRIX}
    validation=json.loads((OUT/'validation.json').read_text())
    if len(validation)!=8 or not all(x['loss_allclose'] and x['weight_probes_allclose'] for x in validation.values()):
        raise RuntimeError('matched-loss validation needs review')
    for file in CODE.glob('*.py'):
        shutil.copyfile(file,OUT/'source'/file.name)
    archive=OUT.parent/'h100-20261122-200102-complete.tar.gz'
    tmp=archive.with_suffix('.partial')
    with tarfile.open(tmp,'w:gz',compresslevel=4) as tar:
        tar.add(BASE,arcname=BASE.name)
        tar.add(OUT,arcname=OUT.name)
    with tarfile.open(tmp,'r:gz') as tar:
        assert len([m for m in tar if m.name.startswith(OUT.name+'/') and m.name.endswith('/result.json')])==len(MATRIX)
    tmp.replace(archive)
    digest=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
    put(archive.with_suffix('.sha256'),digest+'  '+archive.name+'\n')
    return str(archive),digest


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    with (OUT/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.run(['tmux','has-session','-t','=h100'],capture_output=True).returncode==0:
            status('waiting for original suite')
            while subprocess.run(['tmux','has-session','-t','=h100'],capture_output=True).returncode==0:
                time.sleep(15)
        (OUT/'source').mkdir(exist_ok=True)
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
        commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        put(OUT/'hardware.json',json.dumps(dict(gpu=gpu,commit=commit,base=str(BASE),matrix=MATRIX),indent=2))
        for name,script,args in MATRIX:
            try:
                case(name,script,args)
            except BaseException as e:
                status('needs attention',case=name,detail=str(e))
                raise
        status('packaging',complete=len(MATRIX))
        archive,digest=package()
        status('complete',archive=archive,sha256=digest,complete=len(MATRIX))

if __name__=='__main__':
    main()
