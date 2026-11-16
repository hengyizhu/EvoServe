"""Auxiliary operator GPU; consumes early production KV, never runs its VLM."""
from pathlib import Path
import argparse,os,queue,threading,time,traceback
from .io import read,save,append
from .operators import operator_token

def run(cfg):
    from .backend import GraphModel
    from .feature_ipc import read_slot
    out=Path(cfg['out']);out.mkdir(exist_ok=True,parents=True);results=out/'results';results.mkdir(exist_ok=True)
    model=GraphModel(cfg['assets'],stream_priority=0);model.enable_operators(cfg['operators'])
    stop=threading.Event();pending=queue.PriorityQueue(maxsize=64);errors=[];counter=0
    def result(rec,status,**extra):
        row=dict(status=status,request_id=rec['request_id'],expert=rec['expert'],source=rec['source'],ident=rec['id'],deadline=rec['deadline'],finished=time.monotonic(),operator_gpu=cfg['gpu'],stream_priority=0,**extra)
        save(results/(operator_token(rec['source'],rec['id'])+'.json'),row)
        append(out/'requests.jsonl',{k:v for k,v in row.items() if k!='actions'})
    def receive():
        nonlocal counter
        indices={source:0 for source in cfg['sources']}
        try:
            while not stop.is_set():
                moved=False
                for source,i in list(indices.items()):
                    path=Path(source);meta=path/'ipc'/f'{i:08d}.json'
                    if not meta.exists():continue
                    rec=read(meta);rec['source']=source
                    # An expired correction cannot affect production service or accumulate GPU backlog.
                    if time.monotonic()>=rec['deadline']:
                        result(rec,'expired_before_h2d',operator_gpu_ms=0.);meta.unlink();indices[source]+=1;continue
                    packet=read_slot(path,rec['slot']);meta.unlink();indices[source]+=1;counter+=1
                    try:pending.put_nowait((rec['deadline'],counter,rec,packet))
                    except queue.Full:result(rec,'queue_full',operator_gpu_ms=0.)
                    moved=True
                if not moved:stop.wait(.001)
        except Exception:errors.append(traceback.format_exc());stop.set()
    receiver=threading.Thread(target=receive,daemon=True);receiver.start()
    save(out/'ready.json',dict(pid=os.getpid(),gpu=cfg['gpu'],stream_priority=0,cpu_affinity=sorted(os.sched_getaffinity(0)),operators=list(cfg['operators']),no_runtime_prefix_recomputation=True))
    last_cleanup=time.monotonic()
    try:
        while not stop.is_set():
            if errors:raise RuntimeError(errors[-1])
            try:_,_,rec,packet=pending.get(timeout=.05)
            except queue.Empty:continue
            if time.monotonic()>=rec['deadline']:result(rec,'expired_in_queue',operator_gpu_ms=0.);continue
            started=time.monotonic();actions,timing=model.infer_remote_operator(packet,rec['expert'],rec['seed']+7919)
            result(rec,'complete',started=started,actions=actions.tolist(),**timing)
            if time.monotonic()-last_cleanup>10:
                for p in results.glob('*.json'):
                    if time.time()-p.stat().st_mtime>30:p.unlink(missing_ok=True)
                last_cleanup=time.monotonic()
        if errors:raise RuntimeError(errors[-1])
    finally:stop.set();receiver.join(timeout=5)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=read(a.config);os.sched_setaffinity(0,set(cfg['cpus']))
    try:run(cfg)
    except BaseException:save(Path(cfg['out'])/'failure.json',dict(error=traceback.format_exc()));raise

if __name__=='__main__':main()
