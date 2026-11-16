"""Replay measured request arrival times without running a simulator."""
import json
import pickle
import queue
import threading
import time
from pathlib import Path
from .io import RPCClient,read,append


class ArrivalReplay:
    def __init__(self, run, source):
        self.run=run;source=Path(source)
        self.original_clock=read(source/'clock.json')
        self.rows=[json.loads(line) for line in (source/'arrivals.jsonl').read_text().splitlines()]
        with (source/'replay_corpus.pkl').open('rb') as f:self.corpus=pickle.load(f)
        self.queues={r:queue.Queue() for r in {v['robot'] for v in self.rows}}
        self.errors=[];self.threads=[];self.dispatched=0;self.count_lock=threading.Lock()

    def start(self):
        for robot,q in self.queues.items():
            t=threading.Thread(target=self.worker,args=(robot,q),daemon=True);t.start();self.threads.append(t)
        t=threading.Thread(target=self.arrivals,daemon=True);t.start();self.threads.append(t)

    def arrivals(self):
        try:
            for row in sorted(self.rows,key=lambda x:x['release']):
                release=self.run.origin+row['release']-self.original_clock['origin']
                if release>=self.run.deadline:break
                if self.run.stop.wait(max(0,release-time.monotonic())):return
                key='replay:'+row['request_id']
                self.run.ledger.arrive(key,release,row['expert'],row['robot'])
                self.queues[row['robot']].put((key,release,row))
        except Exception:
            import traceback
            self.errors.append(traceback.format_exc())
        finally:
            for q in self.queues.values():q.put(None)

    def worker(self,robot,q):
        clients={}
        try:
            while not self.run.stop.is_set():
                item=q.get()
                if item is None:break
                key,release,row=item;e=row['expert']
                client=clients.setdefault(e,RPCClient(self.run.sockets/f's{self.run.expert_server[e]}'))
                corpus=[v for (expert,_),v in sorted(self.corpus.items()) if expert==e]
                raw=corpus[row.get('corpus_index',0)%len(corpus)]
                with self.count_lock:self.dispatched+=1
                client.call('infer',release=release,operator=row.get('operator',False),raw=raw,expert=e,version=f'{e}:v0',seed=row.get('seed') or self.run.cfg['seed'],episode=f'replay:{robot}',frame=0,request_id=key)
                finish=time.monotonic();self.run.ledger.finish(key,finish)
                append(self.run.out/'replay_requests.jsonl',dict(request_id=key,release=release,finish=finish,expert=e,robot=robot))
        except Exception:
            import traceback
            self.errors.append(traceback.format_exc())
        finally:
            for c in clients.values():c.close()
