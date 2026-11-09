"""Open-loop, separate-process same-GPU inference/SFT experiment on physical GPU 2."""
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'vla_ladder_20261020'
BENCH = ROOT / 'benchmark'


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def save_summaries(rows):
    for row in rows:
        baseline=next((b for b in rows if b['mode']=='serving_only'
            and all(b[k]==row[k] for k in ('tier','repeat','load'))),None)
        if baseline:
            row['p95_increase_percent']=100*(row['p95_seconds']/baseline['p95_seconds']-1)
    save(BENCH/'summary.json',rows)
    columns=['name','tier','repeat','load','mode','requests_completed','p50_seconds',
        'p95_seconds','p95_increase_percent','training_samples_per_second','training_speed_ratio',
        'optimizer_steps','adequate_training_steps','longest_training_boundary_seconds',
        'max_priority_wait_seconds','max_backlog','deadline_exceeded_30s','unfinished',
        'device_peak_bytes','process_peak_bytes','serving_memory','training_memory']
    with (BENCH/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def worker_environment(cpus):
    os.environ.update(CUDA_VISIBLE_DEVICES='2', MUJOCO_EGL_DEVICE_ID='2',
        TOKENIZER_DIR=str(ROOT/'assets/.cache/openpi'),OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',JAX_PLATFORMS='cpu')
    os.sched_setaffinity(0, cpus)


def memory_stats(torch):
    return {'allocated_peak_bytes':torch.cuda.max_memory_allocated(),
            'reserved_peak_bytes':torch.cuda.max_memory_reserved(),
            'allocated_bytes':torch.cuda.memory_allocated(),
            'reserved_bytes':torch.cuda.memory_reserved()}


def server_worker(conn, requests, responses, gate, pending, training_busy, serving_busy, go, abort):
    worker_environment(set(range(8,12)))
    try:
        import numpy as np
        import torch
        from evaluate import load_model
        from train import fingerprint
        torch.set_num_threads(4)
        model = load_model()
        for p in model.parameters():
            p.requires_grad_(False)
        original_hash = fingerprint(model.named_parameters())
        corpus = np.load(BENCH/'inputs.npz',allow_pickle=False)

        def infer(request_id):
            j = request_id % len(corpus['states'])
            torch.manual_seed(71000 + request_id)
            obs = {'main_images':corpus['images'][j:j+1],
                   'wrist_images':corpus['wrists'][j:j+1],
                   'states':corpus['states'][j:j+1],
                   'task_descriptions':[str(corpus['prompts'][j])]}
            with torch.inference_mode():
                actions,_ = model.predict_action_batch(obs,mode='eval')
                output = actions.cpu()
            assert torch.isfinite(output).all()
            return output

        # NPZ arrays are decoded once, not for every request.
        corpus = {key:corpus[key] for key in corpus.files}
        for i in range(30):
            infer(i)
        reference = infer(0)
        conn.send({'state':'ready','pid':os.getpid(),'memory':memory_stats(torch)})
        while True:
            cmd = conn.recv()
            if cmd['op'] == 'stop':
                assert fingerprint(model.named_parameters()) == original_hash
                conn.send({'state':'stopped','production_unchanged':True})
                break
            if cmd['op'] == 'calibrate':
                rows=[]
                for i in range(cmd.get('count',120)):
                    start=time.monotonic(); infer(i); end=time.monotonic()
                    rows.append({'id':i,'start':start,'end':end,'seconds':end-start})
                conn.send({'state':'calibrated','rows':rows})
                continue
            if cmd['op'] != 'run':
                raise ValueError(cmd)
            torch.cuda.reset_peak_memory_stats()
            conn.send({'state':'run_ready'})
            go.wait()
            count=0
            while count < cmd['count'] and not abort.is_set():
                try:
                    req=requests.get(timeout=1)
                except queue.Empty:
                    continue
                dispatch=time.monotonic()
                if cmd['mode'] == 'priority':
                    with gate:
                        while training_busy.value and not abort.is_set():
                            gate.wait(timeout=0.1)
                        serving_busy.value=True
                acquired=time.monotonic()
                cuda_start=torch.cuda.Event(enable_timing=True)
                cuda_end=torch.cuda.Event(enable_timing=True)
                cuda_start.record()
                infer(req['id'])
                cuda_end.record()
                # Wait only for this stream's completion event, not a device-wide barrier.
                cuda_end.synchronize()
                end=time.monotonic()
                with gate:
                    pending.value-=1
                    serving_busy.value=False
                    left=pending.value
                    gate.notify_all()
                responses.put({**req,'dispatch':dispatch,'acquired':acquired,'end':end,
                    'latency_seconds':end-req['scheduled'],
                    'queue_seconds':dispatch-req['enqueued'],
                    'priority_wait_seconds':acquired-dispatch,
                    'service_seconds':end-acquired,
                    'cuda_event_ms':cuda_start.elapsed_time(cuda_end),
                    'backlog_after':left,'deadline_exceeded':end-req['scheduled']>30})
                count+=1
            # The production version must remain numerically unchanged after every trial.
            same=infer(0)
            difference=(reference-same).abs().max().item()
            assert torch.allclose(reference,same,atol=1e-6,rtol=1e-6),difference
            conn.send({'state':'run_done','completed':count,'memory':memory_stats(torch),
                       'production_probe_max_abs':difference})
    except Exception:
        abort.set()
        conn.send({'state':'error','traceback':traceback.format_exc()})
        raise


def trainer_worker(conn, tier, gate, pending, training_busy, serving_busy, go, stop, abort):
    worker_environment(set(range(12,16)))
    try:
        import numpy as np
        import torch
        from evaluate import load_model
        from train import configure, Samples
        torch.set_num_threads(4)
        cfg=json.loads((ROOT/'experiment.json').read_text())
        model=load_model()
        pairs=configure(model,tier)
        masters=[torch.nn.Parameter(p.detach().float().clone()) for _,p in pairs]
        optimizer=torch.optim.AdamW(masters,lr=cfg['lr'],betas=(.9,.95),eps=1e-8,
                                   weight_decay=cfg['weight_decay'],foreach=False)
        saved=torch.load(OLD/f'training/{tier}/resume.pt',map_location='cpu',weights_only=True,mmap=True)
        assert saved['step']==1000 and saved['tier']==tier
        assert saved['master_names']==[n for n,_ in pairs]
        assert saved['parent']==json.loads((ROOT/'audit/parent_download.json').read_text())
        batches=json.loads((OLD/'sample_sequence.json').read_text())
        samples=Samples(model)
        # Warm the data cache outside timing, retaining the original transformations.
        for e in samples.split['target_train']+samples.split['history_train']:
            samples.get(e['episode_index'],0)

        def reset():
            model.load_state_dict(saved['parameters'],strict=False)
            with torch.no_grad():
                for master,value in zip(masters,saved['masters']):
                    master.copy_(value)
            # Adam's CPU step tensors are mutated in place; clone those scalars
            # so every condition really resets to the same saved optimizer state.
            state = {**saved['optimizer'], 'state': {
                key: {name: value.clone() if name == 'step' else value
                      for name, value in values.items()}
                for key, values in saved['optimizer']['state'].items()}}
            optimizer.load_state_dict(state)
            for master in (masters[0], masters[-1]):
                assert optimizer.state[master]['step'].item()==1000
            optimizer.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True)
            model.train()

        def microbatch(step,micro):
            if micro==0:
                optimizer.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True)
            seed=cfg['seed']+step*8+micro
            torch.manual_seed(seed); np.random.seed(seed)
            observation,actions=samples.get(*batches[step % len(batches)][micro])
            loss=model.compute_loss(observation,actions,train=True).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite benchmark training loss')
            (loss/8).backward()
            if micro==7:
                for (name,p),master in zip(pairs,masters):
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        raise RuntimeError(f'Missing/nonfinite gradient: {name}')
                    master.grad=p.grad.detach().float()
                torch.nn.utils.clip_grad_norm_(masters,cfg['clip_grad'],error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    for (_,p),master in zip(pairs,masters):
                        p.copy_(master)
                        if not torch.isfinite(p).all():
                            raise RuntimeError('Nonfinite updated parameter')
            done=torch.cuda.Event()
            done.record(); done.synchronize()
            return loss.detach().item()

        reset()
        for step in range(8):
            for micro in range(8):
                microbatch(step,micro)
        reset()
        conn.send({'state':'ready','pid':os.getpid(),'tier':tier,'resume_step':1000,
                   'trainable_parameters':sum(p.numel() for _,p in pairs),
                   'memory':memory_stats(torch)})
        while True:
            cmd=conn.recv()
            if cmd['op']=='stop':
                conn.send({'state':'stopped'}); break
            assert cmd['op']=='run'
            reset()
            torch.cuda.reset_peak_memory_stats()
            probe_before=model.action_out_proj.weight.detach().clone()
            conn.send({'state':'run_ready'})
            go.wait()
            step=0; micro=0; rows=[]; segments=[]; update_start=None
            while not stop.is_set() and not abort.is_set():
                waiting=time.monotonic()
                if cmd['mode']=='priority':
                    with gate:
                        while (pending.value or serving_busy.value) and not stop.is_set() and not abort.is_set():
                            gate.wait(timeout=.05)
                        if stop.is_set() or abort.is_set():
                            break
                        training_busy.value=True
                start=time.monotonic()
                if micro==0:
                    update_start=start
                loss=microbatch(step,micro)
                end=time.monotonic()
                if cmd['mode']=='priority':
                    with gate:
                        training_busy.value=False; gate.notify_all()
                segments.append({'step':step,'micro':micro,'start':start,'end':end,
                                 'wait_seconds':start-waiting,'loss':loss})
                micro+=1
                if micro==8:
                    rows.append({'step':step+1,'start':update_start,'end':end,'samples':8})
                    micro=0;step+=1
                    if cmd.get('max_steps') and step>=cmd['max_steps']:
                        break
            changed=not torch.equal(probe_before,model.action_out_proj.weight)
            if step and not changed:
                raise RuntimeError('Optimizer steps did not change action output weights')
            conn.send({'state':'run_done','steps':rows,'segments':segments,
                       'partial_microbatches':micro,'parameters_changed':changed,
                       'memory':memory_stats(torch)})
    except Exception:
        abort.set()
        with gate:
            training_busy.value=False; gate.notify_all()
        conn.send({'state':'error','traceback':traceback.format_exc()})
        raise


class Monitor:
    def __init__(self):
        self.rows=[]; self.stop=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        import pynvml as nv
        nv.nvmlInit(); h=nv.nvmlDeviceGetHandleByIndex(2)
        while not self.stop.is_set():
            m=nv.nvmlDeviceGetMemoryInfo(h); util=nv.nvmlDeviceGetUtilizationRates(h)
            self.rows.append({'time':time.monotonic(),'device_used_bytes':m.used,
                'device_free_bytes':m.free,'gpu_util_percent':util.gpu,
                'graphics_clock_mhz':nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_GRAPHICS),
                'power_mw':nv.nvmlDeviceGetPowerUsage(h),
                'processes':[{'pid':p.pid,'used_bytes':p.usedGpuMemory}
                             for p in nv.nvmlDeviceGetComputeRunningProcesses(h)]})
            self.stop.wait(.2)
        nv.nvmlShutdown()
    def __enter__(self):
        self.thread.start(); return self
    def __exit__(self,*args):
        self.stop.set(); self.thread.join(timeout=5)


def receive(conn,expected,timeout=600):
    if not conn.poll(timeout):
        raise TimeoutError(f'Waiting for {expected}')
    value=conn.recv()
    if value.get('state')!=expected:
        raise RuntimeError(value)
    return value


class Experiment:
    def __init__(self):
        self.ctx=mp.get_context('spawn')
        self.gate=self.ctx.Condition()
        self.pending=self.ctx.Value('i',0)
        self.training_busy=self.ctx.Value('b',False)
        self.serving_busy=self.ctx.Value('b',False)
        self.go=self.ctx.Event();self.stop=self.ctx.Event();self.abort=self.ctx.Event()
        self.requests=self.ctx.Queue();self.responses=self.ctx.Queue()
        self.server=None;self.trainer=None

    def start_server(self):
        self.sc,child=self.ctx.Pipe()
        self.server=self.ctx.Process(target=server_worker,args=(child,self.requests,self.responses,
            self.gate,self.pending,self.training_busy,self.serving_busy,self.go,self.abort))
        self.server.start();return receive(self.sc,'ready')

    def start_trainer(self,tier):
        self.tc,child=self.ctx.Pipe()
        self.trainer=self.ctx.Process(target=trainer_worker,args=(child,tier,self.gate,
            self.pending,self.training_busy,self.serving_busy,self.go,self.stop,self.abort))
        self.trainer.start();return receive(self.tc,'ready')

    def stop_trainer(self):
        if self.trainer:
            self.tc.send({'op':'stop'}); receive(self.tc,'stopped')
            self.trainer.join(30)
            assert self.trainer.exitcode==0
            self.trainer=None

    def clear(self):
        self.go.clear(); self.stop.clear()
        assert not self.abort.is_set()
        with self.gate:
            assert self.pending.value==0 and not self.training_busy.value
            self.serving_busy.value=False

    def training_only(self,tier,steps=60):
        assert self.server is None
        self.start_trainer(tier); self.clear()
        self.tc.send({'op':'run','mode':'training_only','max_steps':steps})
        receive(self.tc,'run_ready')
        with Monitor() as monitor:
            self.go.set(); result=receive(self.tc,'run_done',timeout=1200)
        rows=result['steps']; duration=rows[-1]['end']-rows[0]['start']
        result.update(duration_seconds=duration,samples_per_second=len(rows)*8/duration,
                      seconds_per_step=duration/len(rows),device_samples=monitor.rows)
        self.stop_trainer()
        return result

    def serving_capacity(self):
        self.sc.send({'op':'calibrate','count':120})
        value=receive(self.sc,'calibrated')
        rows=value['rows']; rates=[]
        for start in (0,40,80):
            rates.append(40/(rows[start+39]['end']-rows[start]['start']))
        value.update(window_requests_per_second=rates,capacity_rps=min(rates),
                     stable=max(rates)/min(rates)<1.2)
        return value

    def trial(self,name,mode,arrivals):
        self.clear()
        n=len(arrivals)
        self.sc.send({'op':'run','mode':mode,'count':n})
        receive(self.sc,'run_ready')
        if mode!='serving_only':
            self.tc.send({'op':'run','mode':mode})
            receive(self.tc,'run_ready')
        origin=time.monotonic()+1
        responses=[]; max_backlog=0
        with Monitor() as monitor:
            time.sleep(max(0,origin-time.monotonic()))
            self.go.set()
            for i,offset in enumerate(arrivals):
                target=origin+offset
                delay=target-time.monotonic()
                if delay>0:time.sleep(delay)
                if self.abort.is_set():raise RuntimeError('Benchmark worker failed')
                with self.gate:
                    self.pending.value+=1
                    max_backlog=max(max_backlog,self.pending.value)
                    self.gate.notify_all()
                self.requests.put({'id':i,'scheduled':target,'enqueued':time.monotonic()})
            deadline=time.monotonic()+max(600,10*arrivals[-1])
            while len(responses)<n and time.monotonic()<deadline and not self.abort.is_set():
                try:responses.append(self.responses.get(timeout=1))
                except queue.Empty:pass
            self.stop.set()
            with self.gate:self.gate.notify_all()
            if len(responses)!=n:
                save(BENCH/f'{name}_incomplete.json',{'sent':n,'completed':responses,
                    'unfinished_ids':sorted(set(range(n))-{r['id'] for r in responses})})
                raise RuntimeError(f'Incomplete workload: {len(responses)}/{n}')
            serving=receive(self.sc,'run_done')
            training=receive(self.tc,'run_done') if mode!='serving_only' else None
        end=max(r['end'] for r in responses)
        rows=[] if training is None else [r for r in training['steps'] if r['start']>=origin and r['end']<=end]
        segments=[] if training is None else [r for r in training['segments'] if r['start']>=origin and r['end']<=end]
        overlaps=sum(1 for s in segments for r in responses
                     if s['start'] < r['end'] and s['end'] > r['acquired'])
        if mode=='priority':
            assert overlaps==0, 'Priority gate allowed inference/training overlap'
        import numpy as np
        latencies=[r['latency_seconds'] for r in responses]
        process_peaks={}
        for sample in monitor.rows:
            for process in sample['processes']:
                pid=str(process['pid']); process_peaks[pid]=max(process_peaks.get(pid,0),process['used_bytes'])
        assert str(self.server.pid) in process_peaks, 'Serving process was not observed on physical GPU 2'
        if mode!='serving_only':
            assert str(self.trainer.pid) in process_peaks, 'Training process was not observed on physical GPU 2'
        summary={'name':name,'mode':mode,'physical_gpu':2,'serving_pid':self.server.pid,
            'training_pid':self.trainer.pid if self.trainer else None,
            'requests_sent':n,'requests_completed':len(responses),'unfinished':0,
            'training_segments_overlapping_service':overlaps,
            'deadline_exceeded_30s':sum(r['deadline_exceeded'] for r in responses),
            'p50_seconds':float(np.percentile(latencies,50)),
            'p95_seconds':float(np.percentile(latencies,95)),
            'max_latency_seconds':max(latencies),'max_backlog':max_backlog,
            'max_enqueue_lag_seconds':max(r['enqueued']-r['scheduled'] for r in responses),
            'max_priority_wait_seconds':max(r['priority_wait_seconds'] for r in responses),
            'longest_training_boundary_seconds':max((s['end']-s['start'] for s in segments),default=0),
            'duration_seconds':end-origin,'optimizer_steps':len(rows),
            'training_samples_per_second':len(rows)*8/(end-origin),
            'adequate_training_steps':mode=='serving_only' or len(rows)>=10,
            'serving_memory':serving['memory'],'training_memory':training['memory'] if training else None,
            'device_peak_bytes':max((r['device_used_bytes'] for r in monitor.rows),default=0),
            'process_peak_bytes':process_peaks,'production_probe_max_abs':serving['production_probe_max_abs']}
        save(BENCH/f'{name}.json',{'summary':summary,'requests':responses,
            'training':training,'device_samples':monitor.rows})
        return summary

    def close(self):
        if self.abort.is_set():
            for p in (self.server,self.trainer):
                if p and p.is_alive():p.terminate();p.join(30)
            return
        self.stop_trainer()
        if self.server:
            self.sc.send({'op':'stop'}); result=receive(self.sc,'stopped')
            save(BENCH/'production_integrity.json',result)
            self.server.join(30)
            assert self.server.exitcode==0
            self.server=None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--smoke-free',action='store_true')
    ap.add_argument('--smoke-tier',choices=['L1','L2','L3'],default='L1');args=ap.parse_args()
    os.sched_setaffinity(0,{16,17})
    run=Experiment();summaries=[]
    try:
        if args.smoke or args.smoke_free:
            run.start_server();run.start_trainer(args.smoke_tier)
            mode='free' if args.smoke_free else 'priority'
            result=run.trial(f'smoke_{args.smoke_tier}_{mode}',mode,[i*.4 for i in range(30)])
            assert result['requests_completed']==30 and result['optimizer_steps']>=1
            if mode=='free':
                assert result['training_segments_overlapping_service']>0
                run.stop_trainer()
                baseline=run.trial('smoke_serving_only','serving_only',[i*.4 for i in range(30)])
                assert baseline['optimizer_steps']==0 and baseline['training_pid'] is None
            save(BENCH/f'smoke_{args.smoke_tier}_{mode}_status.json',{'state':'passed','summary':result})
            return
        calibration={}
        for tier in ('L1','L2','L3'):
            save(ROOT/'interference_status.json',{'state':'training_only','tier':tier})
            result=run.training_only(tier)
            save(BENCH/f'training_only_{tier}.json',result)
            calibration[tier]={'samples_per_second':result['samples_per_second'],
                               'seconds_per_step':result['seconds_per_step']}
        run.start_server()
        capacity=run.serving_capacity();save(BENCH/'serving_capacity.json',capacity)
        if not capacity['stable']:
            raise RuntimeError('Serving capacity varied by more than 20% across 40-request windows')
        import numpy as np
        loads={};worst=max(x['seconds_per_step'] for x in calibration.values())
        for label,fraction in [('low',.30),('high',.75)]:
            rate=capacity['capacity_rps']*fraction
            # Same workload across tiers/modes. Size before observing any interference
            # result, allowing 20 L3 optimizer steps at the nominal residual capacity.
            n=max(300,math.ceil(rate*20*worst/(1-fraction)))
            rng=np.random.default_rng(20261020)
            intervals=rng.exponential(1, n-1);intervals*=((n-1)/rate)/intervals.sum()
            arrivals=[0.0,*np.cumsum(intervals).tolist()]
            loads[label]={'fraction':fraction,'rate_rps':rate,'count':n,'arrivals':arrivals}
        save(BENCH/'fixed_workloads.json',{'loads':loads,'calibration':calibration,
            'request_seed':20261020,'policy':'same fixed arrivals for every tier, mode and repeat'})
        for repeat in range(1,4):
            if repeat>=2:
                for name in ('L1_seedA','L1_seedB'):
                    deadline=time.monotonic()+7200
                    while json.loads((ROOT/f'training/{name}/status.json').read_text())['state']!='complete':
                        if time.monotonic()>deadline:raise TimeoutError('Waiting for new-seed training')
                        time.sleep(10)
            for tier in ('L1','L2','L3'):
                for load,spec in loads.items():
                    name=f'{tier}_r{repeat}_{load}_serving_only'
                    save(ROOT/'interference_status.json',{'state':'running','trial':name,'completed':len(summaries)})
                    result=run.trial(name,'serving_only',spec['arrivals'])
                    result.update(tier=tier,repeat=repeat,load=load,training_speed_ratio=0.0)
                    summaries.append(result)
                    save_summaries(summaries)
                run.start_trainer(tier)
                modes=('free','priority') if repeat%2 else ('priority','free')
                for mode in modes:
                    for load,spec in loads.items():
                        name=f'{tier}_r{repeat}_{load}_{mode}'
                        save(ROOT/'interference_status.json',{'state':'running','trial':name,'completed':len(summaries)})
                        result=run.trial(name,mode,spec['arrivals'])
                        result.update(tier=tier,repeat=repeat,load=load,
                            training_speed_ratio=result['training_samples_per_second']/calibration[tier]['samples_per_second'])
                        summaries.append(result)
                        save_summaries(summaries)
                run.stop_trainer()
        save(ROOT/'interference_status.json',{'state':'complete','trials':len(summaries),
            'insufficient_training_trials':[r['name'] for r in summaries if not r['adequate_training_steps']]})
    except Exception:
        run.abort.set()
        save(ROOT/'interference_status.json',{'state':'failed','traceback':traceback.format_exc()})
        raise
    finally:
        try:
            run.close()
        except Exception:
            save(ROOT/'interference_status.json',{'state':'shutdown_failed','traceback':traceback.format_exc()})
            raise


if __name__=='__main__':
    main()
