"""Add wall-window CPU sampling and an alternate queue predicate to the pinned load generator."""
import bisect
import inspect
import os
from pathlib import Path
import textwrap
import threading
import time
import types

from load import Load


class CpuWindow:
    def __init__(self):
        v1=Path('/sys/fs/cgroup/cpuacct/cpuacct.usage')
        if v1.exists():
            self.path=v1
            self.read=lambda:int(v1.read_text())/1e9
        else:
            v2=Path('/sys/fs/cgroup/cpu.stat')
            self.path=v2
            self.read=lambda:int(dict(x.split() for x in v2.read_text().splitlines())['usage_usec'])/1e6
        self.error=None
        self.samples=[]

    def arm(self,origin,seconds):
        def run():
            try:
                for i in range(int(seconds)+1):
                    target=origin+i
                    time.sleep(max(0,target-time.monotonic()))
                    usage=self.read()
                    self.samples.append([time.monotonic()-origin,usage])
            except BaseException as exc:
                self.error=exc
        self.thread=threading.Thread(target=run,daemon=True)
        self.thread.start()

    def finish(self,seconds):
        self.thread.join(5)
        if self.error:
            raise self.error
        samples=self.samples
        if len(samples)!=int(seconds)+1:
            raise RuntimeError(f'incomplete CPU window: {len(samples)} samples')
        intervals=[(b[1]-a[1])/(b[0]-a[0]) for a,b in zip(samples,samples[1:])]
        mean=(samples[-1][1]-samples[0][1])/(samples[-1][0]-samples[0][0])
        return dict(mean_cores=mean,peak_cores=max(intervals),normalization_vcpu=8,
                    mean_pct_of_8=100*mean/8,peak_pct_of_8=100*max(intervals)/8,
                    interval_s=1,span_s=samples[-1][0]-samples[0][0],
                    affinity=len(os.sched_getaffinity(0)),source=str(self.path),
                    scope='Pod cgroup CPU time; includes inference, training worker, MPS and other Pod processes',
                    samples_elapsed_s_and_cpu_s=samples)


class QueueSlack:
    def __init__(self,service_ms):
        self.estimates=service_ms
        self.buckets=sorted(service_ms)
        self.calls=0
        self.pending_checks=0
        self.protected_checks=0

    def __call__(self,releases,first,elapsed):
        self.calls+=1
        pending=bisect.bisect_right(releases,elapsed)-first
        if pending<=0:
            return 0
        self.pending_checks+=1
        size=min(pending,self.buckets[-1])
        bucket=self.buckets[bisect.bisect_left(self.buckets,size)]
        waited_ms=1000*max(0,elapsed-releases[first])
        skip=waited_ms+self.estimates[bucket]>0.8*250
        self.protected_checks+=int(skip)
        return int(skip)


def instrument(bench,slack):
    # Derive from the pinned implementation to preserve its queue, batching, feedback,
    # deadlines and scheduled arrivals. Only the sampling hook and optional predicate differ.
    source=textwrap.dedent(inspect.getsource(Load.run))
    anchor='    origin = time.monotonic() + 0.05\n'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+'    if self.cpu_window is not None:\n        self.cpu_window.arm(origin,seconds*len(rates))\n')
    if slack is not None:
        old='''                state[2] = max(
                    0,
                    bisect.bisect_right(releases, time.monotonic() - origin)
                    - dispatched[0],
                )'''
        assert source.count(old)==1
        source=source.replace(old,'                state[2] = self.slack(releases,dispatched[0],time.monotonic()-origin)')
    bench.cpu_window=None
    bench.slack=slack
    namespace=dict(Load.run.__globals__)
    exec(compile(source,'<instrumented suite load>','exec'),namespace)
    bench.run=types.MethodType(namespace['run'],bench)
