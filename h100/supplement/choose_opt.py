"""Select measured paths only after matching all 15 loss vectors and complete master/EMA weights."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
from opt_campaign import OUT,CODE,ENV,run,status


def compare(mode,n,threads,path):
    dst=OUT/f'{mode}-m{n}-{path}-t{threads}'
    ref=OUT/f'{mode}-m{n}-eager-t1'/'trajectory.json'
    if not (dst/'trajectory.json').exists():return None
    old=json.loads(ref.read_text())['steps'];cur=json.loads((dst/'trajectory.json').read_text())['steps']
    assert len(old)==len(cur)==15
    loss_bad=[];weight_bad=[];max_abs=0.
    for i,(a,b) in enumerate(zip(old,cur,strict=True),1):
        if a['losses']!=b['losses']:loss_bad.append(i)
        max_abs=max(max_abs,max(abs(x-y) for x,y in zip(a['losses'],b['losses'],strict=True)))
        if a['weights']!=b['weights']:weight_bad.append(i)
    q=json.loads((dst/'timing.json').read_text())
    q.update(eligible=not loss_bad and not weight_bad,loss_mismatches=loss_bad,full_weight_mismatches=weight_bad,max_abs_loss_diff=max_abs)
    return q


def attempt(mode,n,threads,path):
    dst=OUT/f'{mode}-m{n}-{path}-t{threads}'
    if (dst/'unsupported.json').exists():return
    try:
        run(mode,n,threads,path)
    except subprocess.CalledProcessError as err:
        log=(dst/'run.log').read_text(errors='replace')
        # Keep candidate-specific incompatibilities as experimental results. Infrastructure
        # errors and baseline failures must be looked at before proceeding.
        if path not in ('graph','fullgraph','exactgraph') or not any(w in log.lower() for w in ('captur','out of memory','cudagraph')):
            raise
        (dst/'unsupported.json').write_text(json.dumps(dict(error=log[-4500:]),indent=2))


def collect():
    rows=[]
    for path in OUT.glob('*/timing.json'):
        r=json.loads(path.read_text());n=r['experts'];mode=r['mode']
        v=compare(mode,n,r['threads'],r['path'])
        if v:rows.append(v)
    return rows


def winners(rows):
    result={}
    for r in rows:
        if not r['eligible']:continue
        key=f"{r['mode']}-m{r['experts']}"
        if key not in result or r['median_ms']<result[key]['median_ms']:
            result[key]={k:r[k] for k in ('mode','experts','path','threads','median_ms','experts_per_s','cpu_pct_core')}
    return result


def write_report():
    rows=collect();sel=winners(rows)
    rejected={p.parent.name:json.loads(p.read_text()) for p in OUT.glob('*/unsupported.json')}
    (OUT/'selection.json').write_text(json.dumps(dict(rank=32,selected=sel,candidates=rows,unsupported=rejected,
        rule='Lowest measured median among candidates with identical per-step loss vectors and identical SHA-256 of every FP32 master and EMA parameter for all 15 updates.'),indent=2))
    lines=['# Training optimizations on H100','',
           'All standalone timings use rank 32. Three warm-ups precede 12 wall-clock measurements. A separate validation trajectory compares every expert loss and hashes every master and EMA parameter after each of the same 15 updates. The selection uses the lowest median among exact matches.',
           '','| Mode | Experts | Path | Train threads | Median ms | Expert steps/s | CPU % of one core | Exact trajectory |',
           '|---|---:|---|---:|---:|---:|---:|---|']
    for r in sorted(rows,key=lambda x:(x['mode'],x['experts'],x['path'],x['threads'])):
        lines.append(f"| {r['mode']} | {r['experts']} | {r['path']} | {r['threads']} | {r['median_ms']:.2f} | {r['experts_per_s']:.2f} | {r['cpu_pct_core']:.1f} | {'yes' if r['eligible'] else 'no'} |")
    lines+=['','Packed input assembly transfers inputs and builds batches on the GPU. Prefix records are packed once into pinned buffers by dtype. Identical input objects shared by multiple experts are assembled once per update. Recompute still runs the VLM prefix for each expert and each microbatch on every step.',
            '','Graph timing includes input/RNG staging and replay. The graph path captures both forward/backward microbatches and on-device recomputation; the existing optimizer and finite-gradient checks run eagerly. Fullgraph also captures gradient clipping, capturable AdamW, and EMA, with tensor-valued learning rates. Step-dependent RNG and learning rate values are supplied before replay. Exactpack/exactgraph retain the original CPU numerical preprocessing and assemble batches on the GPU.',
            '','## CPU stacks','',
            'Top Python-defined functions by self wall time in one profiled production step. Self time includes native calls and waits reached from that function. Nested Python-defined calls are attributed to the nested frame. Timings in the main table are collected separately without profiling.']
    for p in sorted(OUT.glob('*/cpu-stacks.json')):
        r=json.loads(p.read_text())
        lines+=['',f"### {r['mode']}, {r['experts']} expert(s)",'','| Python function | Calls | Self ms | Inclusive ms |','|---|---:|---:|---:|']
        for e in r['python_top'][:10]:
            lines.append(f"| {e['function'].replace('|','/')} | {e['calls']} | {e['self_ms']:.2f} | {e['inclusive_ms']:.2f} |")
        lines+=['','| Native PyTorch operation | Calls | Self ms |','|---|---:|---:|']
        for e in r['native_top'][:10]:lines.append(f"| {e['function']} | {e['calls']} | {e['self_ms']:.2f} |")
    if rejected:
        lines+=['','## Capture attempts requiring fallback','']
        for k in rejected:lines.append('- '+k+' (details in unsupported.json)')
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    return rows,sel


def main():
    # The first sampler may already have completed eager, packed and hybrid graph.
    for mode in ('cache','recompute'):
        for n in (1,8):
            for path in ('eager','packed','graph'):
                for threads in (1,4):
                    attempt(mode,n,threads,path)
            attempt(mode,n,1,'fullgraph')
            full=compare(mode,n,1,'fullgraph')
            if full and full['eligible']:
                attempt(mode,n,4,'fullgraph')
            entries=[r for r in collect() if r['mode']==mode and r['experts']==n]
            if not any(r['eligible'] and r['path'] in ('packed','graph','fullgraph') for r in entries):
                for path in ('exactpack','exactgraph'):
                    for threads in (1,4):attempt(mode,n,threads,path)
    _,selected=write_report()
    # Deployable widths 2 and 4 receive their own comparisons of the validated endpoint winners.
    candidates={('eager',1)}
    candidates.update((selected[f'cache-m{n}']['path'],selected[f'cache-m{n}']['threads']) for n in (1,8))
    for n in (2,4):
        attempt('cache',n,1,'eager')
        for path,threads in sorted(candidates):
            attempt('cache',n,threads,path)
    rows,selected=write_report()
    required={'cache-m1','cache-m2','cache-m4','cache-m8','recompute-m1','recompute-m8'}
    assert required<=set(selected)
    status('optimized paths selected',selected=selected)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status('needs attention',detail=str(exc))
        raise
