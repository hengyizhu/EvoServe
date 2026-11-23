"""Split the pinned repository's unmodified mathematical operations at the copy/compute boundary."""
import inspect
import textwrap
import time
from contextlib import nullcontext

import torch
from evoserve.backend import TrainModel


def _block(source, first, after):
    lo = source.index(first)
    hi = source.index(after, lo)
    return source[lo:hi]


def _scope(lines, label):
    indent = lines[:len(lines)-len(lines.lstrip(' '))]
    return indent + f"with self._scope('{label}'):\n" + textwrap.indent(lines, '    ')


def build_split_step():
    # Using the installed function keeps the original RNG, loss scaling, optimizer,
    # EMA, dtype conversions and implicit synchronization unchanged.
    source = inspect.getsource(TrainModel.step)
    stage = _block(source, '        for eid,items in batches.items():', '        ready=')
    original_copy = "self.map(lambda q:q.to('cuda',non_blocking=True),x['prefix'])"
    assert stage.count(original_copy) == 1
    stage = stage.replace(original_copy, 'self._copy_cached_prefix(x)')
    prep = "def prepare(self,batches,recompute=False):\n        t=self.torch;prepared={}\n"
    prep += stage
    prep += "        ready=t.cuda.Event();ready.record(self.copy)\n        return prepared,ready\n"

    compute = source[source.index('        losses=[]'):]
    forward = _block(compute,'                fields=[]','                (values.sum()/2).backward()')
    backward = '                (values.sum()/2).backward();losses.append(values.detach())\n'
    optimizer = _block(compute,'            for eid in ids:\n                e=self.experts','            end=t.cuda.Event()')
    assert backward in compute
    compute = compute.replace(forward, _scope(forward,'forward'))
    compute = compute.replace(backward, _scope(backward,'backward'))
    compute = compute.replace(optimizer, _scope(optimizer,'optimizer'))
    comp = 'def compute(self,prepared,ready,ids,start,recompute=False):\n        t=self.torch\n'+compute
    namespace = dict(TrainModel.step.__globals__)
    exec(compile(prep+'\n'+comp,'<EvoServe split step>','exec'),namespace)
    return namespace['prepare'],namespace['compute']


prepare,compute = build_split_step()


def install(model, traced=False):
    model._scope = (lambda label: torch.profiler.record_function('evo.'+label)) if traced else (lambda label:nullcontext())
    model.lookup_ns = 0
    model.lookup_calls = 0
    if traced:
        def prefix(item):
            with model._scope('cache_lookup'):
                begin=time.perf_counter_ns()
                value=item['prefix']
                model.lookup_ns += time.perf_counter_ns()-begin
                model.lookup_calls += 1
                return value
        model._prefix=prefix
    else:
        model._prefix=lambda item:item['prefix']
    def copy_cached_prefix(item):
        cache=model._prefix(item)
        with model._scope('cache_transfer'):
            return model.map(lambda q:q.to('cuda',non_blocking=True),cache)
    model._copy_cached_prefix=copy_cached_prefix
