"""GPU workers. Imported only by explicitly launched post-measurement processes."""
from pathlib import Path
import json
import os
import queue
import threading
import time
import traceback
from .io import RPCServer,save,read,append


def serve(cfg):
    from .backend import GraphModel
    from .feature_ipc import Export
    out=Path(cfg['out']);out.mkdir(exist_ok=True,parents=True)
    model=GraphModel(cfg['assets'])
    for eid in cfg['experts']:model.install(eid,f'{eid}:v0')
    remote=bool(cfg.get('operator_results'))
    if cfg.get('operators') and not remote:model.enable_operators(cfg['operators'])
    operator_export=None;operator_ident=0
    if remote:
        from .operators import operator_token
        source=out/'operator_export';source.mkdir(exist_ok=True)
        operator_export=Export(source,(*model.prefix,model.static.state))
    exporter=Export(out,model.prefix) if cfg['export_cache'] else None
    requests=queue.Queue();running=True;ident=0
    def handle(msg):
        reply={};done=threading.Event();requests.put((msg,reply,done));done.wait()
        if 'error' in reply:raise RuntimeError(reply['error'])
        return reply['value']
    rpc=RPCServer(cfg['socket'],handle)
    save(out/'ready.json',dict(pid=os.getpid(),graph_verified=True,trainable_parameters=20876320,denoising_steps=10,returned_horizon=10,normal_action_chunk=5,cpu_affinity=sorted(os.sched_getaffinity(0))))
    try:
        while running:
            msg,reply,done=requests.get()
            try:
                if msg['op']=='infer':
                    operator_ref=None;early_record=None
                    def early_prefix(prefix,state,ready):
                        nonlocal operator_ident,operator_ref,early_record
                        record=dict(id=operator_ident,expert=msg['expert'],seed=msg['seed'],request_id=msg['request_id'],episode=msg['episode'],frame=msg['frame'],version=msg['version'],deadline=msg.get('release',time.monotonic())+.25)
                        early_record=operator_export.publish(operator_ident,(*prefix,state),record,ready_event=ready)
                        save(source/'ipc'/f'{operator_ident:08d}.json',early_record)
                        token=operator_token(str(source),operator_ident)
                        operator_ref=dict(path=str(Path(cfg['operator_results'])/(token+'.json')),request_id=msg['request_id']);operator_ident+=1
                    actions,row=model.infer(msg['raw'],msg['expert'],msg['version'],msg['seed'],prefix_hook=early_prefix if remote and msg.get('operator') else None)
                    if early_record is not None:row.update({'operator_'+k:v for k,v in early_record.items() if k.startswith('export_')})
                    ref=None
                    if exporter:
                        rec=exporter.publish(ident,model.prefix,dict(episode=msg['episode'],frame=msg['frame'],version=msg['version'],expert=msg['expert'],id=ident))
                        save(out/'ipc'/f'{ident:08d}.json',rec)
                        ref=dict(server=str(out),id=ident);ident+=1
                        row.update({k:v for k,v in rec.items() if k.startswith('export_')})
                    operator_actions=None
                    if msg.get('operator') and not remote:
                        operator_actions,timing=model.infer_operator(msg['expert'],msg['seed']+7919);row.update(timing)
                    row.update(finish_ns=time.monotonic_ns(),request_id=msg['request_id'],expert=msg['expert'],version=msg['version'],operator_requested=bool(msg.get('operator')))
                    append(out/'requests.jsonl',row)
                    reply['value']=dict(actions=actions,operator_actions=operator_actions,operator_ref=operator_ref,cache_ref=ref,timing=row)
                elif msg['op']=='install':
                    model.install(msg['expert'],msg['version'],msg.get('path'));reply['value']=True
                elif msg['op']=='remove':
                    model.remove(msg['expert'],msg['version']);reply['value']=True
                elif msg['op']=='stop':running=False;reply['value']=True
                else:raise ValueError(msg['op'])
            except Exception:reply['error']=traceback.format_exc()
            finally:done.set()
    finally:
        rpc.close()
        if exporter:exporter.close()
        if operator_export:operator_export.close()


def train(cfg):
    from .backend import TrainModel
    from .core import CacheKey,Window,Reservoir
    from .feature_ipc import read_slot
    out=Path(cfg['out']);out.mkdir(exist_ok=True,parents=True)
    model=TrainModel(cfg['assets'],cfg['experts'],cfg['seed'])
    pool=Reservoir(cfg['pool_samples'],cfg['pool_bytes'],cfg['seed'])
    for eid in cfg['experts']:pool.activate(eid,f'{eid}:v0')
    pending={};dropped=set();lock=threading.RLock();stop=threading.Event();errors=[];blocked=set();commands=queue.Queue()
    sources=[Path(p) for p in cfg['service_outs']]
    counters=dict(received=0,accepted=0,dropped=0,version_mismatch=0)
    def receive():
        try:
            indices={str(p):0 for p in sources}
            while not stop.is_set():
                moved=False
                for source in sources:
                    if not (source/'ipc_layout.pkl').exists():continue
                    i=indices[str(source)];meta=source/'ipc'/f'{i:08d}.json'
                    if not meta.exists():continue
                    rec=read(meta);key=(str(source),i)
                    prefix=read_slot(source,rec['slot'])
                    with lock:
                        if key not in dropped:pending[key]=prefix
                        else:dropped.discard(key)
                        counters['received']+=1
                        # A failed consumer cannot grow host memory without bound.
                        while len(pending)>128:
                            pending.pop(next(iter(pending)));counters['dropped']+=1
                    meta.unlink();indices[str(source)]+=1;moved=True
                if not moved:stop.wait(.002)
        except Exception:errors.append(traceback.format_exc());stop.set()
    receiver=threading.Thread(target=receive,daemon=True);receiver.start()
    status=dict(updates=0,expert_steps={str(e):0 for e in cfg['experts']},pool_samples=0,pool_bytes=0)
    def drop(ref):
        if ref is None:return
        key=(ref['server'],ref['id'])
        with lock:
            if key in pending:pending.pop(key)
            else:
                dropped.add(key)
                if len(dropped)>512:dropped.remove(next(iter(dropped)))
    def handle(msg):
        op=msg['op']
        if op=='drop':
            for ref in msg['refs']:drop(ref)
            return True
        if op=='window':
            prefix=None;ref=msg.get('cache_ref')
            if cfg['mode']!='recompute':
                if ref is None:raise ValueError('A2 window has no serving prefix')
                key=(ref['server'],ref['id']);deadline=time.monotonic()+10
                while True:
                    with lock:prefix=pending.pop(key,None)
                    if prefix is not None:break
                    if time.monotonic()>deadline:
                        counters['dropped']+=1;return dict(accepted=False,reason='prefix_expired_or_missing')
                    if errors:raise RuntimeError(errors[-1])
                    time.sleep(.001)
            if msg.get('data_source')=='operator_correction' and msg.get('action_sources')!=['operator']*10:raise ValueError('correction window contains non-operator actions')
            payload=dict(raw=msg['raw'],actions=msg['actions'],prefix=prefix,data_source=msg.get('data_source','self_success'))
            nbytes=sum(x.numel()*x.element_size() for x in model.leaves(prefix) if model.torch.is_tensor(x)) if prefix is not None else 0
            nbytes+=sum(x.nbytes for x in msg['raw'].values() if hasattr(x,'nbytes'))
            window=Window(CacheKey(msg['episode'],msg['frame'],msg['version']),msg['expert'],payload,nbytes,tuple(msg['action_frames']),msg['success'],msg.get('data_source','self_success'))
            accepted=pool.add(window);counters['accepted']+=int(accepted)
            source_key=msg.get('data_source','self_success')+'_accepted';counters[source_key]=counters.get(source_key,0)+int(accepted)
            if not accepted and pool.versions.get(msg['expert'])!=msg['version']:counters['version_mismatch']+=1
            return dict(accepted=accepted)
        if op=='status':return dict(status,**counters,error=errors[-1] if errors else None)
        reply={};done=threading.Event();commands.put((msg,reply,done));done.wait()
        if 'error' in reply:raise RuntimeError(reply['error'])
        return reply['value']
    rpc=RPCServer(cfg['socket'],handle)
    save(out/'ready.json',dict(pid=os.getpid(),independent_optimizers=True,independent_ema=True,rank=32,effective_batch_per_expert=8,pool_sample_limit=cfg['pool_samples'],pool_byte_limit=cfg['pool_bytes']))
    last_finish=0.;last_seconds=.27
    try:
        while not stop.is_set():
            if errors:raise RuntimeError(errors[-1])
            while not commands.empty():
                msg,reply,done=commands.get()
                try:
                    if msg['op']=='snapshot':
                        blocked.add(msg['expert']);path=out/(msg['version'].replace(':','_')+'.pt');model.snapshot(msg['expert'],path)
                        reply['value']=dict(path=str(path),step=model.experts[msg['expert']]['step'])
                    elif msg['op']=='resume':
                        if msg.get('discard'):model.discard(msg['expert'],msg.get('production_path'))
                        pool.activate(msg['expert'],msg['version']);blocked.discard(msg['expert']);reply['value']=True
                    elif msg['op']=='stop':stop.set();reply['value']=True
                    else:raise ValueError(msg['op'])
                except Exception:reply['error']=traceback.format_exc()
                finally:done.set()
            plan=read(cfg['plan'],{})
            duty=plan.get('duty',0.);pause=float('inf') if duty<=0 else last_seconds*(1-duty)/duty
            if time.monotonic()<last_finish+pause or not plan.get('experts'):
                stop.wait(.002);continue
            batches={}
            for eid in plan['experts']:
                if eid in blocked:continue
                sample=pool.sample(eid)
                if sample:batches[eid]=[x.payload for x in sample]
                if len(batches)>=plan.get('m',1):break
            if not batches:stop.wait(.01);continue
            row=model.step(batches,recompute=cfg['mode']=='recompute');append(out/'steps.jsonl',row)
            status.update(updates=status['updates']+1,expert_steps={str(e):model.experts[e]['step'] for e in cfg['experts']},pool_samples=len(pool.items),pool_bytes=pool.bytes)
            save(out/'status.json',dict(status,**counters))
            last_finish=time.monotonic();last_seconds=.8*last_seconds+.2*row['seconds']
    finally:stop.set();receiver.join(timeout=5);rpc.close()


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['serve','train'],required=True);p.add_argument('--config',required=True);a=p.parse_args();cfg=read(a.config)
    os.sched_setaffinity(0,set(cfg['cpus']))
    try:(serve if a.kind=='serve' else train)(cfg)
    except Exception:
        save(Path(cfg['out'])/'failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
