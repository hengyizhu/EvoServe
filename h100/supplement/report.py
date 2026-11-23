import json
import statistics
from pathlib import Path
import numpy as np


def put(path,text):
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(text,encoding='utf-8')
    tmp.replace(path)


def robot_feedforward(rows):
    result={}
    for count in (4,5,6,7):
        measured=[r for r in rows if r['robots']==count and r['mode']=='cache']
        tested=sorted(r['requested_experts'] for r in measured)
        if count==7 and not tested:
            result[count]=dict(status='baseline only',tested=[])
            continue
        if tested!=[1,2,4,8]:
            result[count]=dict(status='pending',tested=tested)
            continue
        eligible=[r for r in measured if r['phase']['slo']>=0.995 and r['phase']['p99_ms']<=220.0]
        if not eligible:
            result[count]=dict(status='no qualifying width',max_experts=0,expert_steps_s=None,tested=tested)
            continue
        best=max(eligible,key=lambda r:r['requested_experts'])
        p=best['phase']
        result[count]=dict(status='qualified',max_experts=best['requested_experts'],expert_steps_s=p['expert_steps_s'],
                           p99_ms=p['p99_ms'],slo=p['slo'],mean_duty=p['mean_duty'],tested=tested)
    return result


def refresh(out):
    complete={p.parent.name:json.loads(p.read_text()) for p in sorted(out.glob('*/result.json'))}
    lines=['# H100 supplemental measurements','',
           'Single H100. Standalone cached training uses 3 warmups and 12 timed steps. '
           'H2D and compute are allowed to overlap. Clean timings and instrumented timings are reported separately.','',
           '| Rank | Experts | Cache path | Clean median step ms | Clean mean step ms | Expert steps/s | CPU % (100 = one core) | Intra/inter-op threads | OS threads sampled (active ≥10 ms) |',
           '|---:|---:|---|---:|---:|---:|---:|---|---|']
    cache=[r for r in complete.values() if 'variant' in r]
    variants={'original':0,'prefetch':1,'resident':2}
    cache.sort(key=lambda r:(r['rank'],r['experts'],variants[r['variant']]))
    for r in cache:
        t=r['timing']
        lines.append(f"| {r['rank']} | {r['experts']} | {r['variant']} | {t['median_ms']:.2f} | {t['mean_ms']:.2f} | {t['expert_steps_s']:.2f} | {t['cpu_pct_one_core']:.1f} | {t['torch_threads']} / {t['torch_interop_threads']} | {t['os_threads_peak_sampled']} ({t['active_os_threads_10ms']}) |")
    lines.extend(['','Profiled wall-clock attribution in ms per step, averaged over 2 independent step windows in the same trace. '
                  'Columns through Other sum to Profiled wall. H2D includes observation, action, and prefix transfers. Cache lookup measures host references to pinned tensors. '
                  'DMA reads from host memory are included in H2D. Forward includes device-side input/parameter assembly inside each forward region. Other includes Python, host waits, and unclassified packing kernels. '
                  'DMA/compute overlap is reported separately. Instrumentation affects elapsed time.',
                  '','| Rank | Experts | Path | Host lookup | H2D | Forward | Backward | Optimizer + EMA | Other | Profiled wall | H2D/compute overlap |',
                  '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for r in cache:
        q=r['trace']['means']
        fields=[q['cache_lookup'],q['h2d'],q['forward'],q['backward'],q['optimizer'],
                q['other_wall']+q['other_gpu'],q['wall_ms'],q['h2d_compute_overlap_ms']]
        lines.append(f"| {r['rank']} | {r['experts']} | {r['variant']} | "+' | '.join(f'{v:.3f}' for v in fields)+' |')
    lines.extend(['','| Rank | Experts | Original pinned cache | Dataset size MiB | Prefetch speedup | Resident speedup | Largest absolute loss difference (prefetch / resident) |',
                  '|---:|---:|---|---:|---:|---:|---|'])
    verification={}
    for r in cache:
        if r['variant']!='original':continue
        rank,n=r['rank'],r['experts']
        speeds=[]; errors=[]
        for variant in ('prefetch','resident'):
            other=next((x for x in cache if (x['rank'],x['experts'],x['variant'])==(rank,n,variant)),None)
            if not other:
                speeds.append('pending');errors.append('pending')
                continue
            a=np.array(r['timing']['losses'])
            b=np.array(other['timing']['losses'])
            diff=float(abs(a-b).max())
            ma=np.asarray(r['parameter_probes'])
            mb=np.asarray(other['parameter_probes'])
            verification[f'{rank}-{n}-{variant}']=dict(max_abs_loss_diff=diff,loss_allclose=bool(np.allclose(a,b,rtol=2e-4,atol=2e-5)),max_abs_weight_probe_diff=float(abs(ma-mb).max()),weight_probes_allclose=bool(np.allclose(ma,mb,rtol=2e-4,atol=2e-5)))
            speeds.append(f"{r['timing']['median_ms']/other['timing']['median_ms']:.3f}x");errors.append(f'{diff:.3g}')
        lines.append(f"| {rank} | {n} | {r['cache']['all_pinned']} | {r['cache']['bytes']/2**20:.1f} | {speeds[0]} | {speeds[1]} | {' / '.join(errors)} |")
    lines.extend(['','The production host cache is pinned once at initialization. Transfers use non_blocking=True on a dedicated CUDA stream; the production step synchronizes before starting the next batch. Prefetch stages the next batch on that stream using one additional host thread. In the resident reference only the prefix cache stays on device; observations and action labels still follow the original input path.',
                  '','| Rank | Experts | Path | Lightweight wall ms | Forward stream ms | Backward stream ms | Optimizer stream ms | Bare host lookup ms | Profiled KV H2D ms |',
                  '|---:|---:|---|---:|---:|---:|---:|---:|---:|'])
    for r in cache:
        q=r['light_cuda_event_timing']['means']
        lines.append(f"| {r['rank']} | {r['experts']} | {r['variant']} | {q['wall_ms']:.2f} | {q['forward']:.2f} | {q['backward']:.2f} | {q['optimizer']:.2f} | {r['trace']['bare_cache_lookup_ms_per_step']:.4f} | {r['trace']['means']['kv_h2d']:.3f} |")
    lines.extend(['','CUDA stream spans include enqueue gaps. The separate clean run is used for speedups. CPU utilization includes all process threads, relative to one logical core.',
                  '','The optimization matrix, CPU stack rankings, staged numerical verification and timing-selected worker configurations are in [training-optimization/summary.md](training-optimization/summary.md). The 7-robot measurement is a pure-inference baseline. The 4-, 5-, and 6-robot training co-runs use the fastest tested exact trajectory for the corresponding training mode and expert count. Inference uses one intra-op thread in every group.',
                  '','Each robot sends 4 requests/s at evenly spaced times. Every measurement window is 300 seconds. SLO is the percentage finishing within 250 ms. '
                  'Cache co-runs fix the merge width at rank 32 and start at 100% commanded duty. The original 5-second deadline-miss feedback changes only duty, with pending-queue protection active. Feedforward lookup is disabled during these measurements. Duty is the controller command averaged over the wall window. '
                  'Raw controls, dispatch times and actual admission widths are archived. Recompute uses the inference-priority stream and a single training expert.',
                  '','| Robots | Arrival | Rate/s | Mode | Experts | Training path / threads | P99 ms | SLO % | Expert steps/s | Mean commanded duty % | CPU mean % of 8 vCPUs | CPU 1 s peak % of 8 vCPUs |',
                  '|---:|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|'])
    robot=[r for r in complete.values() if 'robots' in r]
    robot.sort(key=lambda r:(r['robots'],r['alignment'],{'inference':0,'export':1,'cache':2,'recompute':3,'slack':4}[r['mode']],r['requested_experts'] or 0))
    for r in robot:
        q=r['phase']
        ep=f"{q['expert_steps_s']:.3f}" if r['mode'] not in ('inference','export') else '—'
        duty=f"{100*q['mean_duty']:.2f}" if r['mode'] not in ('inference','export') else '—'
        c=r['cpu']
        cfg=r.get('training_config')
        worker=f"{cfg['path']} / {cfg['threads']}" if cfg else '—'
        lines.append(f"| {r['robots']} | {r['alignment']} | {r['rate_hz']} | {r['mode']} | {r['requested_experts'] or '—'} | {worker} | {q['p99_ms']:.2f} | {q['slo']*100:.3f} | {ep} | {duty} | {c['mean_pct_of_8']:.2f} | {c['peak_pct_of_8']:.2f} |")
    table=robot_feedforward(robot)
    put(out/'robot_feedforward.json',json.dumps(dict(slo_min=0.995,p99_ms_max=220,table=table),indent=2))
    lines.extend(['','## H100 feedforward table by robot count','',
                  'Largest measured merge width satisfying both SLO ≥99.5% and P99 ≤220 ms. Values are tested before rounding. Zero means no measured width qualified; throughput is unavailable for that selection.',
                  '','| Robots | Max experts | Expert steps/s | Selected P99 ms | Selected SLO % | Status |',
                  '|---:|---:|---:|---:|---:|---|'])
    for count,r in table.items():
        vals=[f"{r[k]*scale:.3f}" if r.get(k) is not None else '—' for k,scale in (('expert_steps_s',1),('p99_ms',1),('slo',100))]
        lines.append(f"| {count} | {r.get('max_experts','—')} | "+' | '.join(vals)+f" | {r['status']} |")
    lines.extend(['','## Time-slack guard','',
                  'While a request is queued, the worker defers a training update when the oldest queued wait plus the predicted service time of the current padded batch exceeds 200 ms. Predictions use the original single-GPU static end-to-end means. Feedback and stream/MPS settings match the old guard. Old-rule choices use staggered measurements at the same robot count.',
                  '','| Robots | Arrival | Chosen experts | Selection basis | P99 ms | SLO % | Expert steps/s | Mean duty % |',
                  '|---:|---|---:|---|---:|---:|---:|---:|'])
    for r in robot:
        if r['mode']=='slack':
            q=r['phase']
            lines.append(f"| {r['robots']} | {r['alignment']} | {r['requested_experts']} | {r['selection']} | {q['p99_ms']:.2f} | {100*q['slo']:.3f} | {q['expert_steps_s']:.3f} | {100*q['mean_duty']:.2f} |")
    lines.extend(['','## Host cache export overhead','',
                  'Every dispatched inference batch exports KV to host RAM through the suite exporter. The latest batch is retained; the timed loop performs no disk writes for KV. CPU percentages use the entire Pod CPU cgroup over the 300-second measurement, divided by eight logical cores. Peak is the maximum of one-second intervals.',
                  '','| Robots | P99 baseline → export ms | Δ P99 ms | SLO baseline → export % | CPU mean baseline → export % | CPU peak baseline → export % | Host export GiB |',
                  '|---:|---|---:|---|---|---|---:|'])
    for r in robot:
        if r['mode']!='export':continue
        pure=next(x for x in robot if x['robots']==r['robots'] and x['mode']=='inference')
        q,p=r['phase'],pure['phase']; c,b=r['cpu'],pure['cpu']
        lines.append(f"| {r['robots']} | {p['p99_ms']:.2f} → {q['p99_ms']:.2f} | {q['p99_ms']-p['p99_ms']:+.2f} | {100*p['slo']:.3f} → {100*q['slo']:.3f} | {b['mean_pct_of_8']:.2f} → {c['mean_pct_of_8']:.2f} | {b['peak_pct_of_8']:.2f} → {c['peak_pct_of_8']:.2f} | {r['export_stats']['total_bytes']/2**30:.2f} |")
    put(out/'summary.md','\n'.join(lines)+'\n')
    put(out/'validation.json',json.dumps(verification,indent=2))
    return complete
