"""Unprofiled step times and per-step full-parameter SHA-256 validation, with identical trajectories."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from model import Work

STEPS=12
WARMUP=3


def items_for(w,n):
    batch=[w.items[(w.cursor+i)%len(w.items)] for i in range(8)]
    w.cursor=(w.cursor+8)%len(w.items)
    return {i:batch for i in range(n)}


def hash_parameters(m):
    # Every bit of every trainable FP32 master and EMA is checked, per expert.
    out=[]
    for eid,e in sorted(m.experts.items()):
        h=hashlib.sha256()
        for key in ('masters','ema'):
            vec=torch.cat([p.detach().reshape(-1) for _,p in sorted(e[key].items())])
            h.update(memoryview(vec.cpu().numpy()).cast('B'))
        out.append(h.hexdigest())
    return out


def launch(args,validate):
    os.environ['EVOSERVE_CPU_THREADS']=str(args.threads)
    torch.manual_seed(20261108)
    w=Work(args.mode,32,args.experts)
    assert torch.get_num_threads()==args.threads
    m=w.model
    step=m.step
    if args.path!='eager':
        from fast_train import attach
        attach(m,args.path)
        step=m.step
    ts=[];cpu=[];checks=[]
    for i in range(WARMUP+STEPS):
        b=items_for(w,args.experts)
        cpu0=time.process_time()
        start=time.perf_counter()
        res=step(b,recompute=args.mode=='recompute')
        elapsed=time.perf_counter()-start
        used=time.process_time()-cpu0
        if i>=WARMUP:
            ts.append(1000*elapsed);cpu.append(used)
        if validate:
            checks.append(dict(losses=res['losses'],weights=hash_parameters(m)))
    if validate:
        return dict(steps=checks,initial_seed=20261108,all_masters_and_ema=True)
    out=dict(ms=ts,median_ms=statistics.median(ts),mean_ms=statistics.mean(ts),
             experts_per_s=1000*args.experts/statistics.mean(ts),
             cpu_pct_core=100*sum(cpu)/(sum(ts)/1000),cpu_pct_8=100*sum(cpu)/(sum(ts)*8/1000),
             threads=torch.get_num_threads(),mode=args.mode,experts=args.experts,path=args.path,
             graph_capture_setup_ms=sum(g.setup_ms for g in getattr(m,'fast_graphs',{}).values()))
    return out


def put(file,data):
    tmp=file.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(file)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['cache','recompute']);p.add_argument('experts',type=int)
    p.add_argument('threads',type=int);p.add_argument('path');p.add_argument('out',type=Path)
    p.add_argument('--validate',action='store_true')
    p.add_argument('--both',action='store_true')
    a=p.parse_args()
    if a.both:
        from cpu_stacks import profile
        for f in sorted(a.out.parent.glob('*-eager-t1/trajectory.json')):
            if not (f.parent/'cpu-stacks.json').exists():
                cfg=json.loads((f.parent/'timing.json').read_text())
                if cfg['experts'] not in (1,8):
                    continue
                print('CPU stack profile',cfg['mode'],cfg['experts'],flush=True)
                profile(f.parent,cfg['mode'],cfg['experts'])
                gc.collect();torch.cuda.empty_cache()
    a.out.mkdir(parents=True,exist_ok=True)
    r=launch(a,a.validate)
    put(a.out/('trajectory.json' if a.validate else 'timing.json'),r)
    if not a.validate:print(a.mode,a.experts,a.path,a.threads,r['median_ms'],flush=True)
    if a.both:
        gc.collect();torch.cuda.empty_cache()
        put(a.out/'trajectory.json',launch(a,True))
