"""Open-loop fixed-rate load, actual serving-cache training, unrestricted MPS feedback."""
import argparse,sys,os,time,json,pickle,threading,queue,math,csv,traceback,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evoserve.runtime import Run
from evoserve.io import read,save,append,RPCClient
from evoserve.core import DeadlineLedger,TwoKnobController

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args();cfg=read(a.config);os.sched_setaffinity(0,{30,31});out=Path(a.out)
 root=Path(cfg['inputs']).parent
 if read(root/'pending_reset_gate.json',{}).get('state')=='pending':
  subprocess.run([sys.executable,str(Path(__file__).with_name('repair_reset_gate.py')),'--root',str(root)],check=True)
  save(root/'status.json',dict(state='prototype_running',item=out.parent.name,time=time.time()))
  save(root/'prototype_active.json',dict(pid=os.getpid(),cmd=[sys.executable,*sys.argv],out=str(out.parent),time=time.time()))
 run=Run(cfg,out);samples=pickle.loads(Path(cfg['inputs']).read_bytes());learn=cfg['method']=='ours';stop=threading.Event();incoming=queue.Queue();errors=[];clients=threading.local();thread_clients=[];lock=threading.Lock();dispatched=0;dispatch_counts={}
 try:
  run.start();server=run.service[0];trainer=run.trainer[0]
  def store(sample,reply,key):
   if not learn:return
   trainer.call('window',expert=3,episode=key,frame=sample['frame'],version='3:v0',raw=sample['raw'],actions=sample['actions'],cache_ref=reply['cache_ref'],action_frames=list(range(sample['frame'],sample['frame']+10)),success=True,data_source='demonstration')
  # Real demonstration windows, warmed through exactly the serving export path.
  warm=[x for x in samples if x['category']=='target'][:4]+[x for x in samples if x['category']=='history'][:4]
  for i,sample in enumerate(warm):
   msg=server.call('infer',raw=sample['raw'],expert=3,version='3:v0',seed=20261108+i,episode='warm',frame=sample['frame'],request_id=f'warm:{i}',release=time.monotonic(),operator=False)
   store(sample,msg,f'demonstration:warm:{i}')
  def ingest():
   try:
    while not stop.is_set() or not incoming.empty():
     try:sample,reply,key=incoming.get(timeout=.1)
     except queue.Empty:continue
     store(sample,reply,key)
   except Exception:errors.append(traceback.format_exc());stop.set()
  receiver=threading.Thread(target=ingest,daemon=True);receiver.start()
  ledger=DeadlineLedger();controller=TwoKnobController(1);origin=time.monotonic()+.5;end=origin+cfg['seconds'];rate=cfg['rate'];n=cfg['clients'];alignment=cfg.get('alignment','staggered');count=round(cfg['seconds']*rate);window=0;inflight=set();actual_rows=[]
  save(out/'measurement_clock.json',dict(origin=origin,end=end,rate=rate,clients=n,per_client_hz=rate/n,alignment=alignment))
  before=trainer.call('status')['expert_steps'].get('3',0)
  save(run.plans[0],dict(experts=[3] if learn else [],m=1,duty=1 if learn else 0))
  def request(i,release):
   nonlocal dispatched
   if not hasattr(clients,'rpc'):
    clients.rpc=RPCClient(run.sockets/'s0')
    with lock:thread_clients.append(clients.rpc)
   sample=samples[i%len(samples)];robot=i%n;expert=3 if learn else cfg['expert_ids'][robot%len(cfg['expert_ids'])];sent=time.monotonic()
   with lock:dispatched+=1;dispatch_counts[robot]=dispatch_counts.get(robot,0)+1
   reply=clients.rpc.call('infer',raw=sample['raw'],expert=expert,version=f'{expert}:v0',seed=20261108+i,episode=f'load:{i}',frame=sample['frame'],request_id=f'load:{i}',release=release,operator=False)
   finished=time.monotonic();ledger.finish(f'load:{i}',finished)
   row=dict(request_id=f'load:{i}',robot=robot,expert=expert,release=release,dispatched=sent,finished=finished,latency_ms=(finished-release)*1000,**{k:reply['timing'][k] for k in ['batch_size','graph_batch_size','queue_wait_ms']})
   append(out/'load_requests.jsonl',row)
   with lock:actual_rows.append(row)
   if learn:incoming.put((sample,reply,f'demonstration:served:{i}'))
  def control(now):
   nonlocal window
   while now>=origin+(window+1)*5+.25:
    lo=origin+window*5;hi=lo+5
    if hi>end:break
    stat=ledger.window(lo,hi,now);change=controller.update(stat)
    append(out/'feedback_windows.csv.jsonl',dict(window=window,**stat,**change));save(run.plans[0],dict(experts=[3] if learn else [],m=1,duty=controller.duty if learn else 0));window+=1
  pool=ThreadPoolExecutor(max_workers=min(64,max(8,n*4)))
  try:
   for i in range(count):
    release=origin+(i/rate if alignment=='staggered' else (i//n)/(rate/n))
    time.sleep(max(0,release-time.monotonic()))
    ledger.arrive(f'load:{i}',release,3 if learn else cfg['expert_ids'][(i%n)%len(cfg['expert_ids'])],i%n)
    for f in list(inflight):
     if f.done():f.result();inflight.remove(f)
    if len(inflight)<max(8,n*4):inflight.add(pool.submit(request,i,release))
    if errors:raise RuntimeError(errors[-1])
    for child in run.children:
     if child.poll() is not None:raise RuntimeError(f'worker {child.pid} exited')
    control(time.monotonic())
   time.sleep(max(0,end+.3-time.monotonic()));control(time.monotonic());after=trainer.call('status')['expert_steps'].get('3',0);save(run.plans[0],dict(experts=[],m=1,duty=0))
   stat=ledger.window(origin,end,time.monotonic());finished=[r for r in actual_rows if origin<=r['release']<end]
   step_path=out/'train0/steps.jsonl'
   step_rows=[json.loads(x) for x in step_path.read_text().splitlines()] if step_path.exists() else []
   updates=sum(origin<=x['finish_ns']/1e9<end for x in step_rows)
   summary=dict(state='complete',method=cfg['method'],seconds=cfg['seconds'],clients=n,equivalent_robots=rate/4,rate=rate,alignment=alignment,batch_sizes=cfg['batch_sizes'],batch_wait_ms=cfg['batch_wait_ms'],training_steps=updates,steps_s=updates/cfg['seconds'],actual_dispatch_rate=dispatched/cfg['seconds'],per_client_actual_hz={str(k):dispatch_counts.get(k,0)/cfg['seconds'] for k in range(n)},training_data='real labeled demonstration windows; serving-produced prefixes; throughput-only',power_limit_w=225,**stat)
   feedback=[json.loads(x) for x in (out/'feedback_windows.csv.jsonl').read_text().splitlines()]
   summary['mean_duty']=sum(x['before']['duty'] for x in feedback)/len(feedback) if feedback and learn else 0.
   flat=[dict(window=x['window'],requests=x['requests'],timeouts=x['timeouts'],slo=x['slo'],p99_ms=x['p99_ms'],duty_before=x['before']['duty'],duty_after=x['after']['duty']) for x in feedback]
   if flat:
    with (out/'feedback_windows.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
   save(out/'summary.json',summary)
  finally:
   stop.set();pool.shutdown(wait=True,cancel_futures=True);receiver.join(timeout=30)
   for future in inflight:
    if not future.cancelled():future.result()
 finally:
  stop.set()
  for c in thread_clients:c.close()
  run.close()
if __name__=='__main__':
 try:main()
 except Exception:
  # The campaign retains stderr, retries once and continues independent items.
  raise
