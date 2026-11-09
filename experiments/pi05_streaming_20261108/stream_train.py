"""Fixed-rate trace arrivals; receiver-side service KV costs, bounded recent pool."""
import os,sys,argparse,time,json,threading,queue,collections,traceback,dataclasses,math
from pathlib import Path
R=Path(__file__).resolve().parent;B=Path('/home/hengyi/pi05_a1_baseline_20261108')
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['A1','A2'],required=True);ap.add_argument('--rate',type=int,choices=[4,16],required=True);ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');args0=ap.parse_args();OUT0=Path(args0.out);OUT0.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(B));sys.argv=[str(B/'worker.py'),'--mode',args0.mode,'--phase','co','--out',str(OUT0)]
import worker as base
from worker import torch,np,save,event,obs_map,cat_obs,tree_map,tree_flatten,stat,append,GPU
ARGS=args0;OUT=OUT0
class Streaming(base.Engine):
 def load_cpu(self):
  split=json.loads(Path('/home/hengyi/vla_followup_20261020/data_split.json').read_text());target_episodes={x['episode_index'] for x in split['target_train']};self.new_sources=[k for k in self.keys if k[0] in target_episodes];self.replay=list(dict.fromkeys((x['episode'],x['start']) for x in split['replay_windows']))
  if ARGS.smoke:
   self.new_sources=self.new_sources[:12];self.replay=self.replay[:8];self.micros=[list(self.new_sources[:4]),list(self.replay[:4])]*3
  self.keys=list(dict.fromkeys(self.new_sources+self.replay));super().load_cpu()
 def init_stream(self):
  self.recent=collections.OrderedDict();self.retired=[];self.arrivals=[];self.incoming=queue.Queue(maxsize=1024);self.stop=threading.Event();self.source_cache={};self.producer_error=None;self.stream_count=0;self.max_recent=8 if ARGS.smoke else 512;self.sample_rng=np.random.default_rng(20261108);self.micros=[];self.mi=0;self.prefetched=None;self.measure_origin=None;self.measure_end=None
  # A2 immutable cache payloads stand in for output from the serving cluster.
  # Producer-side computation below is outside the receiving-card measurement.
  if self.mode=='A2':
   start=time.monotonic()
   for i,key in enumerate(self.new_sources):
    obs,_=self.data[key]
    with torch.no_grad():
     device=obs_map(lambda x:x.to('cuda',non_blocking=True),obs);_,mask,kv=self.model.build_prefix_cache(device);self.store(key,(mask,kv))
    if i%16==15:self.copy.synchronize()
    if i%100==0:save(OUT/'status.json',{'state':'external_service_payload_preparation','samples':i,'total':len(self.new_sources),'gpu':GPU})
   self.copy.synchronize();self.source_cache={key:self.cache.pop(key)[0] for key in self.new_sources};save(OUT/'external_payload_preparation.json',{'seconds':time.monotonic()-start,'unique_source_frames':len(self.new_sources),'excluded_from_receiver_measurement':True,'provenance':'frozen service-path B1 prefix of replayed recorded observations; each new arrival receives a new logical experience id'})
  start=time.monotonic()
  for i,key in enumerate(self.replay):
   obs,_=self.data[key]
   with torch.no_grad():
    device=obs_map(lambda x:x.to('cuda',non_blocking=True),obs);_,mask,kv=self.model.build_prefix_cache(device);self.store(key,(mask,kv))
   if i%16==15:self.copy.synchronize()
  self.copy.synchronize();save(OUT/'replay_warmup.json',{'seconds':time.monotonic()-start,'cached_unique_windows':len(self.replay),'configured_windows':1000 if not ARGS.smoke else 8,'coverage':1.0});self.hits=self.misses=0
 def produce(self,origin):
  try:
   rng=np.random.default_rng(20261109);ident=0
   while not self.stop.is_set():
    nominal=origin+round(ident/ARGS.rate*1e9);delay=(nominal-time.monotonic_ns())/1e9
    if delay>0 and self.stop.wait(delay):break
    key=self.new_sources[int(rng.integers(len(self.new_sources)))];start=time.monotonic_ns();payload=None
    if self.mode=='A2':
     def hostcopy(x):
      h=torch.empty(x.shape,dtype=x.dtype,pin_memory=True);h.copy_(x);return h
     payload=tree_map(hostcopy,self.source_cache[key])
    ready=time.monotonic_ns();record={'id':ident,'source':key,'key':(*key,ident),'nominal_ns':nominal,'receiver_started_ns':start,'receiver_ready_ns':ready,'receiver_host_store_ms':(ready-start)/1e6,'receiver_lag_ms':(ready-nominal)/1e6,'sample_count':0,'payload':payload};self.incoming.put(record,timeout=30);ident+=1
  except Exception:self.producer_error=traceback.format_exc();self.stop.set()
 def ingest(self):
  if self.producer_error:raise RuntimeError(self.producer_error)
  while True:
   try:record=self.incoming.get_nowait()
   except queue.Empty:break
   key=record['key'];record['visible_ns']=time.monotonic_ns();self.data[key]=self.data[record['source']];payload=record.pop('payload');self.recent[key]=record;self.arrivals.append(record)
   if payload is not None:
    done=event();done.record(self.copy);self.cache[key]=(payload,done)
   while len(self.recent)>self.max_recent:
    old,rec=self.recent.popitem(last=False);self.cache.pop(old,None);self.data.pop(old,None);rec['evicted_ns']=time.monotonic_ns();self.retired.append(rec)
 def loss(self,p):
  wa,wb=event(),event();wa.record(self.stream);self.stream.wait_event(p['done']);wb.record(self.stream);p['wait_events']=[(wa,wb)];values=[];p['d2h']=[];p['prefix_events']=[];p['class_counts']={'new':{'hits':0,'misses':0},'replay':{'hits':0,'misses':0}}
  for j,item in enumerate(self.micros[p['mi']]):
   key=tuple(item);category='new' if len(key)==3 else 'replay';val=p['prefixes'][j];obs=p['parts'][j][0]
   if key in self.cache:
    self.hits+=1;p['class_counts'][category]['hits']+=1
    if val is None:
     val,done=self.fetch(key,p['records']);a,b=event(),event();a.record();self.stream.wait_event(done);b.record();p['wait_events'].append((a,b))
   else:
    self.misses+=1;p['class_counts'][category]['misses']+=1;a,b=event(),event();a.record()
    with torch.no_grad():_,mask,kv=self.model.build_prefix_cache(obs)
    b.record();p['prefix_events'].append((a,b,category));val=(mask,kv);p['d2h'].append(self.store(key,val))
   values.append(val)
  self.current_parts.append(p);obs=cat_obs([part[0] for part in p['parts']]);actions,noise,t=[torch.cat([part[k] for part in p['parts']],0) for k in [1,2,3]];x=t[:,None,None]*noise+(1-t[:,None,None])*actions;mask,kv=tree_map(lambda *xs:torch.cat(xs,0),*values)
  for part in p['parts']:
   obs_map(lambda q:(q.record_stream(self.stream) or q),part[0])
   for q in part[1:4]:q.record_stream(self.stream)
  for val in values:
   for q in tree_flatten(val)[0]:q.record_stream(self.stream)
  return (self.model.velocity_from_suffix(self.model.run_suffix(obs,x,t,kv,mask))-(noise-actions)).square().mean()
 def next_step(self):
  self.ingest()
  if not self.recent:return None
  new=list(self.recent);batch=[new[int(self.sample_rng.integers(len(new)))] for _ in range(4)]+[self.replay[int(self.sample_rng.integers(len(self.replay)))] for _ in range(4)];self.sample_rng.shuffle(batch)
  for key in batch:
   if len(key)==3:self.recent[key]['sample_count']+=1
  self.micros.extend([batch[:4],batch[4:]]);self.current_parts=[];row=super().step()
  row['class_counts']={c:{k:sum(p['class_counts'][c][k] for p in self.current_parts) for k in ['hits','misses']} for c in ['new','replay']};row['prefix_compute_ms']={c:sum(a.elapsed_time(b) for p in self.current_parts for a,b,kind in p['prefix_events'] if kind==c) for c in ['new','replay']};row['new_pool_size']=len(self.recent);row['cache_new_count']=sum(len(k)==3 for k in self.cache);row['cache_replay_count']=sum(len(k)==2 for k in self.cache)
  assert row['cache_new_count']<=self.max_recent;assert all(len(k)==2 or k in self.recent for k in self.cache)
  self.current_parts=[];return row
 def cohort(self,end):
  origin=self.measure_origin;full=[r for r in self.retired if r['nominal_ns']>=origin and r['evicted_ns']<=end];observed=[r for r in self.arrivals if origin<=r['nominal_ns']<end]
  return {'complete_lifetime_samples':len(full),'k_complete_lifetime':float(np.mean([r['sample_count'] for r in full])) if full else None,'arrivals_in_measurement':len(observed),'receiver_host_store_ms':stat([r['receiver_host_store_ms'] for r in observed]),'receiver_lag_ms':stat([r['receiver_lag_ms'] for r in observed]),'definition':'k: mean sample selections from arrival through eviction, only cohorts born and evicted within measurement; end-censored arrivals excluded'}

def wait_file(p):
 while not p.exists():time.sleep(.05)
 return json.loads(p.read_text())
def main():
 save(OUT/'status.json',{'state':'loading','gpu':GPU,'mode':ARGS.mode,'rate':ARGS.rate});tr=Streaming(ARGS.mode)
 # Base parity uses the streaming loss hook, so initialize its diagnostics first.
 tr.current_parts=[];tr.parity();tr.current_parts=[];tr.init_stream();save(OUT/'train_ready.json',{'ready':True,'pid':os.getpid()});thread=None;allrows=[]
 try:
  for index in range(20):
   cfg=wait_file(OUT/f'command_{index:03d}.json')
   if cfg['op']=='stop':break
   if cfg['op']=='calibrate':save(OUT/f'train_done_{index:03d}.json',{'complete':True});continue
   origin=cfg['origin_ns'];end=origin+round(cfg['duration_s']*1e9);out=OUT/cfg['name'];out.mkdir(exist_ok=True)
   if thread is None:thread=threading.Thread(target=tr.produce,args=(origin,),daemon=True);thread.start()
   time.sleep(max(0,(origin-time.monotonic_ns())/1e9));rows=[]
   if cfg['name']=='window1':tr.measure_origin=origin
   while time.monotonic_ns()<end:
    row=tr.next_step()
    if row is None:time.sleep(.005);continue
    rows.append(row);append(out/'steps.jsonl',row)
    if tr.step_id%25==0:save(OUT/'status.json',{'state':'running','gpu':GPU,'mode':ARGS.mode,'rate':ARGS.rate,'window':cfg['name'],'step':tr.step_id,'remaining_window_s':max(0,(end-time.monotonic_ns())/1e9)})
   valid=[x for x in rows if origin<=x['start_ns'] and x['finish_ns']<=end];counts={c:{k:sum(x['class_counts'][c][k] for x in valid) for k in ['hits','misses']} for c in ['new','replay']};wall=sum(x['seconds'] for x in valid)*1000;prefix={c:sum(x['prefix_compute_ms'][c] for x in valid) for c in ['new','replay']};summary={'complete_updates':len(valid),'steps_per_s':len(valid)/cfg['duration_s'],'class_counts':counts,'hit_rates':{c:counts[c]['hits']/max(1,sum(counts[c].values())) for c in counts},'prefix_compute_ms':prefix,'prefix_compute_fraction_of_step_wall':{c:prefix[c]/max(wall,1e-12) for c in prefix},'h2d_ms':stat([x['h2d_ms'] for x in valid]),'new_pool_size':len(tr.recent),'cache_replay_count':sum(len(k)==2 for k in tr.cache)};save(out/'train_summary.json',summary)
   if cfg['name'].startswith('window'):allrows.extend(valid);save(OUT/'cohort_summary.json',tr.cohort(end))
   save(OUT/f'train_done_{index:03d}.json',{'complete':True})
  tr.stop.set()
  if thread:thread.join(timeout=10)
  if tr.producer_error:raise RuntimeError(tr.producer_error)
  with (OUT/'arrivals.jsonl').open('w') as f:
   for r in tr.arrivals:f.write(json.dumps(r)+'\n')
  tr.copy.synchronize();save(OUT/'train_status.json',{'state':'complete'});save(OUT/'status.json',{'state':'complete','gpu':GPU,'mode':ARGS.mode,'rate':ARGS.rate})
 finally:tr.stop.set()
if __name__=='__main__':
 try:main()
 except Exception:save(OUT/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
