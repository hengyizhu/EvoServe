"""CPU call-stack attribution on production training. Inclusive/native time can include GPU waits."""
import collections
import gc
import json
import os
from pathlib import Path
import re
import tempfile
import time
import torch
from model import Work


def aggregate(events,kind,lo,hi):
    threads=collections.defaultdict(list)
    for e in events:
        if e.get('ph')!='X' or e.get('cat')!=kind:continue
        name=e['name']
        if kind=='python_function' and not re.search(r'\(\d+\):\s',name):continue
        a=max(lo,e['ts']);b=min(hi,e['ts']+e.get('dur',0))
        if b>a:threads[(e['pid'],e['tid'])].append((a,b,name))
    stats=collections.defaultdict(lambda:[0,0.,0.])
    for rows in threads.values():
        stack=[]
        for a,b,name in sorted(rows,key=lambda x:(x[0],-x[1])):
            while stack and b>stack[-1][1]+.001:stack.pop()
            while stack and a>=stack[-1][1]:stack.pop()
            dur=b-a
            vals=stats[name];vals[0]+=1;vals[1]+=dur;vals[2]+=dur
            if stack:stats[stack[-1][2]][2]-=dur
            stack.append((a,b,name))
    return sorted([dict(function=n,calls=v[0],inclusive_ms=v[1]/1000,
                        self_ms=max(0,v[2])/1000) for n,v in stats.items()],key=lambda q:-q['self_ms'])


def profile(dst,mode,n):
    os.environ['EVOSERVE_CPU_THREADS']='1'
    w=Work(mode,32,n)
    for _ in range(3):w.step(n)
    kw=dict(activities=[torch.profiler.ProfilerActivity.CPU],with_stack=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True))
    with torch.profiler.profile(**kw) as prof:
        with torch.profiler.record_function('evo.python.inspect'):
            w.step(n)
    with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
        prof.export_chrome_trace(tmp.name)
        with open(tmp.name,encoding='utf8') as fp:events=json.load(fp)['traceEvents']
    step=next(e for e in events if e.get('cat')=='user_annotation' and e['name']=='evo.python.inspect')
    lo=step['ts'];hi=lo+step['dur']
    py=aggregate(events,'python_function',lo,hi)
    native=aggregate(events,'cpu_op',lo,hi)
    if len(py)<10:
        raise RuntimeError('PyTorch profiler supplied fewer than ten Python frames')
    data=dict(mode=mode,experts=n,threads=1,profiled_steps=1,wall_ms=step['dur']/1000,
              python_top=py[:60],native_top=native[:60],
              sync_calls=[q for q in native if any(k in q['function'].lower() for k in ('item','scalar_dense','synchroniz','copy_','_to_copy','cat','stack'))],
              definition='Python self time excludes nested Python-defined frames and includes C/native work and blocking waits. Times are host wall times in an instrumented step.')
    dest=dst/'cpu-stacks.json'
    tmp=dst/'cpu-stacks.tmp';tmp.write_text(json.dumps(data,indent=2));tmp.replace(dest)
