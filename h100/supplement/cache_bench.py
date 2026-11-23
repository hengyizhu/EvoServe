"""Compare the production cached step, a two-batch pipeline, and a device-resident KV cache."""
import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import gc
import gzip
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import psutil
import torch
from model import Work
import stages
from trace_metrics import analyze

WARMUP=3
REPEATS=12
TRACE_STEPS=2


def batches(work,experts):
    items=[work.items[(work.cursor+i)%len(work.items)] for i in range(8)]
    work.cursor=(work.cursor+8)%len(work.items)
    return {eid:items for eid in range(experts)}


def snapshot_threads(proc):
    return {t.id:t.user_time+t.system_time for t in proc.threads()}


class Runner:
    def __init__(self,rank,n,variant):
        self.work=Work('cache',rank,n)
        self.model=self.work.model
        self.n=n
        self.variant=variant
        m=self.model
        leaves=[q for x in self.work.items for q in m.leaves(x['prefix']) if torch.is_tensor(q)]
        self.cache=dict(samples=len(self.work.items), tensors=len(leaves),
                        all_pinned=all(q.is_pinned() for q in leaves),
                        bytes=sum(q.numel()*q.element_size() for q in leaves))
        if not self.cache['all_pinned']:
            if variant=='prefetch':
                for item in self.work.items:
                    item['prefix']=m.map(lambda x:x.pin_memory(),item['prefix'])
        if variant=='resident':
            for item in self.work.items:
                item['prefix']=m.map(lambda x:x.cuda(non_blocking=True),item['prefix'])
            torch.cuda.synchronize()
        self.pool=ThreadPoolExecutor(max_workers=1) if variant=='prefetch' else None
        self.pending=None
        stages.install(m)

    def submit(self):
        b=batches(self.work,self.n)
        self.pending=self.pool.submit(stages.prepare,self.model,b)

    def step(self,traced=False):
        m=self.model
        if self.pool:
            if self.pending is None:
                self.submit()
            prepared,ready=self.pending.result()
            self.submit()
            return stages.compute(m,prepared,ready,list(range(self.n)),time.monotonic_ns())
        b=batches(self.work,self.n)
        if not traced:
            return m.step(b)
        prepared,ready=stages.prepare(m,b)
        return stages.compute(m,prepared,ready,list(b),time.monotonic_ns())

    def close(self):
        if self.pool:
            self.pending.result()
            self.pool.shutdown(wait=True)
        torch.cuda.synchronize()

    def clean_timing(self):
        proc=psutil.Process()
        stages.install(self.model)
        for _ in range(WARMUP):
            self.step()
        times=[]
        losses=[]
        start_cpu=time.process_time()
        start_threads=snapshot_threads(proc)
        start=time.perf_counter()
        max_threads=len(start_threads)
        for _ in range(REPEATS):
            t=time.perf_counter()
            result=self.step()
            times.append(1000*(time.perf_counter()-t))
            losses.append(result['losses'])
            max_threads=max(max_threads,proc.num_threads())
        elapsed=time.perf_counter()-start
        end_cpu=time.process_time()
        end_threads=snapshot_threads(proc)
        return dict(warmup=WARMUP, repeats=REPEATS, ms=times, median_ms=statistics.median(times),
                    mean_ms=statistics.mean(times),expert_steps_s=self.n*1000/statistics.mean(times),
                    cpu_pct_one_core=100*(end_cpu-start_cpu)/elapsed,
                    cpu_affinity_cores=len(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),
                    torch_interop_threads=torch.get_num_interop_threads(),os_threads_peak_sampled=max_threads,
                    active_os_threads_10ms=sum(t-start_threads.get(i,0)>=0.009 for i,t in end_threads.items()),
                    cpu_worker_prefetch_threads=int(bool(self.pool)),losses=losses)

    def timeline(self,out):
        stages.install(self.model,traced=True)
        # Warm profiler initialization independently to avoid timing one-off setup in a step.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(TRACE_STEPS):
                with torch.profiler.record_function('evo.step'):
                    self.step(traced=True)
        with tempfile.NamedTemporaryFile(suffix='.json') as raw:
            prof.export_chrome_trace(raw.name)
            try:
                rows=analyze(raw.name,out/'timeline.json.gz')
                if len(rows)!=TRACE_STEPS:
                    raise RuntimeError(f'expected {TRACE_STEPS} host steps, got {len(rows)}')
            except BaseException:
                with open(raw.name,'rb') as fin,gzip.open(out/'trace-debug.json.gz','wb') as fout:
                    shutil.copyfileobj(fin,fout)
                raise
        return dict(steps=rows, means={k:statistics.mean(r[k] for r in rows) for k in rows[0]},
                    bare_cache_lookup_ms_per_step=self.model.lookup_ns/1e6/TRACE_STEPS,
                    bare_cache_lookups=self.model.lookup_calls,
                    attribution='Disjoint wall timeline; H2D takes priority over overlapping kernels. Other GPU includes packing. Host cache lookup is a view/reference; the memory read during DMA is counted in H2D.')

    def parameter_probes(self):
        chunks=[]
        with torch.no_grad():
            for expert in self.model.experts.values():
                for weights in (expert['masters'],expert['ema']):
                    for name,param in sorted(weights.items()):
                        flat=param.detach().reshape(-1)
                        chunks.extend(flat[i:i+1] for i in (0,flat.numel()//2,flat.numel()-1))
            return torch.cat(chunks).float().cpu().tolist()

    def light_timing(self):
        # CUDA events have much less instrumentation overhead than a CPU+CUDA trace.
        # A stream span includes host launch gaps, so these are complementary measurements.
        stages.install(self.model)
        events=[]
        @contextlib.contextmanager
        def scope(name):
            if name not in ('forward','backward','optimizer'):
                yield
                return
            begin=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            begin.record()
            try:
                yield
            finally:
                end.record()
                events.append((name,begin,end))
        self.model._scope=scope
        rows=[]
        for _ in range(6):
            events.clear()
            start=time.perf_counter()
            self.step(traced=True)
            row=dict(wall_ms=1000*(time.perf_counter()-start))
            for name,a,b in events:
                row[name]=row.get(name,0)+a.elapsed_time(b)
            rows.append(row)
        return dict(rows=rows,means={k:statistics.mean(r[k] for r in rows) for k in rows[0]})

def main():
    p=argparse.ArgumentParser()
    p.add_argument('rank',type=int);p.add_argument('n',type=int)
    p.add_argument('variant',choices=['original','prefetch','resident'])
    p.add_argument('out',type=Path)
    p.add_argument('--repair-profile',action='store_true')
    args=p.parse_args()
    if not args.repair_profile:
        for path in sorted(args.out.parent.glob('cache-*/result.json')):
            previous=json.loads(path.read_text())
            if len(previous['trace']['steps'])!=TRACE_STEPS:
                print('Rechecking host-only profiler windows',path.parent.name,flush=True)
                subprocess.run([sys.executable,__file__,str(previous['rank']),str(previous['experts']),
                                previous['variant'],str(path.parent),'--repair-profile'],check=True)
    runner=Runner(args.rank,args.n,args.variant)
    try:
        if args.repair_profile:
            out=json.loads((args.out/'result.json').read_text())
            runner.clean_timing()
            runner.light_timing()
            out['trace']=runner.timeline(args.out)
            out['profile_rechecked']=True
            put=args.out/'result.tmp'
            put.write_text(json.dumps(out,indent=2,allow_nan=False))
            put.replace(args.out/'result.json')
            return
        timing=runner.clean_timing()
        out=dict(rank=args.rank,experts=args.n,variant=args.variant,cache=runner.cache,timing=timing)
        out['parameter_probes']=runner.parameter_probes()
        out['light_cuda_event_timing']=runner.light_timing()
        # Clean timing of the original and resident paths uses TrainModel.step verbatim.
        # Collect a separate profiled timeline; its wall time is reported separately.
        out['trace']=runner.timeline(args.out)
        out['all_finite']=all(torch.isfinite(torch.tensor(x)).all().item() for x in timing['losses'])
        if not out['all_finite']:
            raise RuntimeError('non-finite loss')
        (args.out/'result.tmp').write_text(json.dumps(out,indent=2,allow_nan=False))
        (args.out/'result.tmp').replace(args.out/'result.json')
        print(args.variant,args.rank,args.n,timing['median_ms'],flush=True)
    finally:
        runner.close()

if __name__=='__main__':
    main()
