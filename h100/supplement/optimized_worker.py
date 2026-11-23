"""Use the numerically verified fastest training path; CUDA stream priorities stay unchanged."""
import ctypes
import json
import os
from pathlib import Path
import time
import traceback

import torch
from model import Work
from fast_train import attach


def configure_training_thread(count):
    # ATen lazily initializes each thread from its process default. Initialize before
    # setting the OpenMP ICV. These two setters only affect this host thread.
    torch.ones(1).add_(1)
    libdir=Path(torch.__file__).resolve().parent/'lib'
    omp=ctypes.CDLL(str(libdir/'libgomp.so.1'))
    omp.omp_set_num_threads(ctypes.c_int(count))
    mkl=ctypes.CDLL(str(libdir/'libtorch_cpu.so'))
    mkl.MKL_Set_Num_Threads_Local(ctypes.c_int(count))
    if torch.get_num_threads()!=count or omp.omp_get_max_threads()!=count:
        raise RuntimeError('training thread count mismatch')
    return dict(torch_intraop=torch.get_num_threads(),omp=omp.omp_get_max_threads(),
                mkl=mkl.MKL_Get_Max_Threads())


def worker(mode,rank,maximum,shared,ready,errors):
    try:
        cfg=json.loads(os.environ['EVOSERVE_FAST_TRAIN'])
        w=Work(mode,rank,maximum)
        counts=configure_training_thread(cfg['threads'])
        if cfg['path']!='eager':attach(w.model,cfg['path'])
        w.step(maximum)
        meta=dict(mode=mode,width=maximum,rank=rank,path=cfg['path'],threads=counts,pid=os.getpid())
        dst=Path(os.environ['EVOSERVE_ROBOT_OUT'])/'training-worker.json'
        dst.write_text(json.dumps(meta,indent=2))
        ready.set()
        deadline=0.
        while not shared['stop'].is_set():
            with shared['state'].get_lock():
                n,duty,pending,protected,enabled=shared['state'][:]
            if not enabled or not n or not duty or (protected and pending) or time.monotonic()<deadline:
                time.sleep(.001)
                continue
            if torch.get_num_threads()!=cfg['threads']:
                raise RuntimeError('training intra-op threads changed during measurement')
            start=time.monotonic()
            w.step(int(n));finish=time.monotonic()
            shared['completed'].put((start,finish,int(n)))
            deadline=finish+(finish-start)*(1/duty-1)
    except BaseException:
        errors.put(traceback.format_exc());ready.set()
