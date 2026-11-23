"""Staggered independent 4 Hz robots using the suite's original inference and training controller."""
import argparse
import collections
import json
import os
from pathlib import Path

import load as load_module
from load import Load
from robot_controls import CpuWindow,QueueSlack,instrument

SECONDS=300
RANK=32
SIZES=(1,2,4,8,16)
BASE=Path('/workspace/EvoServe/results/h100-20261122-200102')


def run(args):
    export=args.mode=='export'
    slack_case=args.mode=='slack'
    mode=None if args.mode in ('inference','export') else ('cache' if slack_case else args.mode)
    cache=mode=='cache'
    cap=args.experts
    selection=None
    if slack_case:
        old=[json.loads((args.out.parent/f'robots-{args.robots}-cache-m{n}'/'result.json').read_text()) for n in (1,2,4,8)]
        eligible=[r['requested_experts'] for r in old if r['phase']['slo']>=.995 and r['phase']['p99_ms']<=220.]
        cap=max(eligible,default=1)
        selection='largest qualifying width, old staggered guard' if eligible else 'diagnostic width 1; no qualifying old width'
    rate=4*args.robots
    sync=1 if args.alignment=='staggered' else args.robots
    options=dict(n=cap,sync=sync,table=None,feedback=cache)
    chosen=None
    if mode:
        import torch
        from optimized_worker import worker
        selections=json.loads((args.out.parent/'training-optimization'/'selection.json').read_text())
        chosen=selections['selected'][f'{mode}-m{cap}']
        os.environ['EVOSERVE_FAST_TRAIN']=json.dumps(chosen)
        os.environ['EVOSERVE_ROBOT_OUT']=str(args.out)
        load_module.worker=worker
    bench=Load(mode,cap,RANK,sizes=SIZES)
    try:
        import torch
        if torch.get_num_threads()!=1:
            raise RuntimeError('inference must have one intra-op thread')
        services={n:json.loads((BASE/f'02-static-{n}/attempt-1/result.json').read_text())['service_ms'] for n in SIZES}
        instrument(bench,QueueSlack(services) if slack_case else None)
        priorities=dict(inference=bench.server.stream.priority,
                        training=0 if mode else None)
        if mode=='recompute' and priorities['inference']>=priorities['training']:
            raise RuntimeError(f'inference stream has insufficient priority: {priorities}')
        kernel_ms=[]
        byte_sizes={}
        exports=collections.Counter()
        retained=[None]
        if not mode:
            original_serve=bench.server.serve
            def timed_serve(count):
                result=original_serve(count,save_cache=export)
                kernel_ms.append(float(result[0])+float(result[1]))
                if export:
                    retained[0]=result[2]
                    bucket=next(n for n in SIZES if n>=count)
                    exports[bucket]+=1
                    if bucket not in byte_sizes:
                        leaves=bench.server.leaves(result[2])
                        if not all(t.device.type=='cpu' for t in leaves):
                            raise RuntimeError('cache export did not reach CPU')
                        byte_sizes[bucket]=sum(t.numel()*t.element_size() for t in leaves)
                return result
            bench.server.serve=timed_serve
        bench.run([rate],5,**options)
        kernel_ms.clear();exports.clear()
        if slack_case:
            bench.slack=QueueSlack(services)
        bench.cpu_window=CpuWindow()
        result=bench.run([rate],SECONDS,**options)
        if torch.get_num_threads()!=1:
            raise RuntimeError('inference thread count changed')
        cpu=bench.cpu_window.finish(SECONDS)
        result['cpu_samples_elapsed_s_and_cpu_s']=cpu.pop('samples_elapsed_s_and_cpu_s')
        phase=result['phases'][0]
        if not mode:
            result['inference_dispatch_gpu_ms']=kernel_ms
        if phase['requests']!=rate*SECONDS:
            raise RuntimeError(f"missing requests: {phase['requests']}")
        active_width_seconds=collections.Counter()
        controls=result['controls']
        for (t,width,duty),end in zip(controls,[x[0] for x in controls[1:]]+[SECONDS]):
            active_width_seconds[int(width)]+=max(0,min(end,SECONDS)-max(0,t))
        if cache and set(active_width_seconds)!={cap}:
            raise RuntimeError(f'unexpected merge widths: {active_width_seconds}')
        payload=dict(robots=args.robots,rate_hz=rate,mode=args.mode,rank=RANK if mode else None,
                     requested_experts=cap if mode else None,seconds=SECONDS,staggered=sync==1,alignment=args.alignment,
                     deadline_ms=250,phase=phase,priorities=priorities,
                     transport=os.environ['EVOSERVE_TRANSPORT'] if mode else 'inference only',
                     feedforward_enabled=False,initial_duty=1.0 if mode else None,
                     inference_gpu_stage_fraction=sum(kernel_ms)/(1000*SECONDS) if not mode else None,
                     inference_dispatch_count=len(kernel_ms) if not mode else None,
                     seconds_by_commanded_width=dict(active_width_seconds) if cache else None,
                     queue_guard=('slack' if slack_case else 'any pending') if cache else None,feedback=cache,
                     selection=selection,cpu=cpu,cache_export=export)
        payload['inference_intraop']=torch.get_num_threads()
        if mode:
            payload['training_config']=chosen
            payload['training_worker']=json.loads((args.out/'training-worker.json').read_text())
        if slack_case:
            payload['slack']=dict(margin_ms=200,expected_service_ms=services,
                watcher_checks=bench.slack.calls,pending_checks=bench.slack.pending_checks,
                over_margin_checks=bench.slack.protected_checks)
        if export:
            payload['export_stats']=dict(batch_counts=exports,bytes_per_bucket=byte_sizes,
                total_bytes=sum(exports[n]*byte_sizes[n] for n in exports),retained_batches=1,
                method='Server.serve(save_cache=True): clone to CPU for every dispatched padded batch')
        (args.out/'windows.json').write_text(json.dumps(result,separators=(',',':')))
        tmp=args.out/'result.tmp'
        tmp.write_text(json.dumps(payload,indent=2,allow_nan=False))
        tmp.replace(args.out/'result.json')
        print(phase,flush=True)
    finally:
        bench.close()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('robots',type=int,choices=(4,5,6,7))
    p.add_argument('mode',choices=('inference','cache','recompute','slack','export'))
    p.add_argument('experts',type=int,choices=(1,2,4,8))
    p.add_argument('out',type=Path)
    p.add_argument('--alignment',choices=('sync','staggered'),default='staggered')
    run(p.parse_args())
