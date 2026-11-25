"""Five-minute fixed-width cells on the final execution layer, one invocation per repeat."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import traceback

def worker(mode, rank, maximum, shared, ready, errors):
    import torch
    from fast_train import attach
    from model import Work, prefix
    from optimized_worker import configure_training_thread
    try:
        out=Path(os.environ['NIGHT_CELL'])
        cfg=json.loads((out/'config.json').read_text())
        kind=cfg['method']
        work=Work('cache' if kind=='A2' else 'recompute',rank,maximum)
        configure_training_thread(4 if maximum>1 else 1)
        attach(work.model,'graph' if cfg['graph'] else 'packed')
        def step():
            first=work.cursor
            forwards=8 if kind=='A0' else 0
            batch=[work.items[(work.cursor+i)%len(work.items)] for i in range(8)]
            work.cursor=(work.cursor+8)%len(work.items)
            if kind=='A1':
                with torch.no_grad(),torch.cuda.stream(work.stream):
                    for sample in batch:
                        if 'prefix' not in sample:
                            forwards+=1
                            obs,_=work.batch([sample])
                            sample['prefix']=work.model.map(lambda q:q.detach().cpu().pin_memory(),prefix(work.model,obs))
            work.model.step({e:batch for e in range(maximum)},recompute=kind=='A0')
            return first,forwards
        step()
        if kind=='A1':
            for item in work.items:
                item.pop('prefix',None)
        work.cursor=0
        ready.set()
        deadline=0
        profiled=False
        counts=(out/'draws.csv').open('w',buffering=1)
        while not shared['stop'].is_set():
            with shared['state'].get_lock():
                n,duty,pending,protected,enabled=shared['state'][:]
            if not enabled or not n or not duty or protected and pending or time.monotonic()<deadline:
                time.sleep(.001)
                continue
            profile=cfg['group']=='graph' and not profiled and (out/'profile').exists()
            begin=time.monotonic()
            if profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
                    first,forwards=step()
                path=out/'trace.json'
                p.export_chrome_trace(str(path))
                kernels=[e for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel' and 'dur' in e]
                if kernels:
                    longest=max(kernels,key=lambda e:e['dur'])
                    (out/'kernel.json').write_text(json.dumps({'name':longest['name'],'ms':longest['dur']/1000}))
                path.unlink()
                profiled=True
            else:
                first,forwards=step()
            end=time.monotonic()
            counts.write(f'{end},{first},{forwards}\n')
            shared['completed'].put((begin,end,maximum))
            deadline=end+(end-begin)*(1/duty-1)
    except BaseException:
        errors.put(traceback.format_exc())
        ready.set()

def main(out):
    import inspect
    import math
    import textwrap
    import types
    import numpy as np
    import torch
    import load
    from inference_engine import SelectedServer
    from sparse_export import SparseExport, selected_requests
    from robot_controls import QueueSlack
    cfg=json.loads((out/'config.json').read_text())
    os.environ['NIGHT_CELL']=str(out)
    sizes=(1,2,4,8)
    load.Server=lambda shapes:SelectedServer(shapes,os.environ['EVOSERVE_INFERENCE_PLAN'])
    load.worker=worker
    bench=load.Load('cache',cfg['m'],cfg['rank'],sizes)
    try:
        # Preserve the execution layer's periodic stagger and randomize only its phase.
        service={int(k):v for k,v in cfg['service_ms'].items()}
        rate=cfg['load']*10/service[1]
        rng=random.Random(cfg['seed'])
        shift=rng.random()/rate
        def releases(seconds):
            return [shift+i/rate for i in range(math.ceil((seconds-shift)*rate))]
        source=textwrap.dedent(inspect.getsource(load.Load.run))
        pos=source.index('    releases = []')
        source=source[:pos]+'    releases = self.arrivals(seconds)\n'+source[source.index('    total =',pos):]
        pos=source.index('                state[2] = max(')
        source=source[:pos]+'                state[2] = self.slack(releases,dispatched[0],time.monotonic()-origin)'+source[source.index('\n\n',pos):]
        source=source.replace('        dispatch=records,','        dispatch=records, origin=origin, steps=steps,')
        ns=dict(load.Load.run.__globals__)
        exec(compile(source,'<fixed-width layout measurement>','exec'),ns)
        bench.run=types.MethodType(ns['run'],bench)
        bench.arrivals=releases
        bench.slack=QueueSlack(service)
        serve=bench.server.serve
        if cfg['method']=='A2':
            bench.server.sparse_export=SparseExport(bench.server)
        def run(seconds):
            count=len(releases(seconds))
            selected=selected_requests(1,count) if cfg['method']=='A2' else set()
            cursor=0
            def dispatch(n):
                nonlocal cursor
                ids=[j for j in range(n) if cursor+j in selected]
                cursor+=n
                return serve(n,export_indices=ids)
            bench.server.serve=dispatch
            result=bench.run([rate],seconds,n=cfg['m'],feedback=True)
            assert cursor==count
            bench.check()
            return result,selected
        sec=cfg['seconds']
        r,exported=run(sec)
        steps=[(s,f,n) for s,f,n in r['steps'] if 0<=f<sec]
        times=np.array([1000*(f-s) for s,f,n in steps])
        # Work.step alternates two disjoint batches of eight fixed benchmark observations.
        draws=len(steps)*8
        import csv
        with (out/'draws.csv').open() as f:
            counted=[(int(i),int(v)) for end,i,v in csv.reader(f) if 0<=float(end)-r['origin']<sec]
        assert len(counted)==len(steps)
        unique=len({(i+j)%16 for i,_ in counted for j in range(8)})
        vlm=sum(v for _,v in counted)
        phase=r['phases'][0]
        result=dict(**phase,steps=len(steps),train_draws=draws,distinct_samples=unique,
                    training_vlm_sample_forwards=vlm,inference_vlm_sample_forwards=phase['requests'],
                    vlm_per_unique_training_sample=vlm/unique if unique else None,
                    train_vlm_per_draw=vlm/draws if draws else None,
                    exports=len(exported),phase_offset_s=shift,
                    step_mean_ms=float(times.mean()) if len(times) else None,
                    step_sd_ms=float(times.std(ddof=1)) if len(times)>1 else None)
        (out/'result.json').write_text(json.dumps(result))
        if cfg['group']=='graph':
            (out/'profile').touch()
            run(10)
            if not (out/'kernel.json').exists():
                raise RuntimeError('No CUDA kernel recorded in the separate profiling window')
    finally:
        bench.close()

if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
