import os,sys,argparse,time,json,threading,queue,collections,traceback,dataclasses,math
from pathlib import Path
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['A0','A1','A2'],required=True);ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--prepare-only',action='store_true');ARGS=ap.parse_args();OUT=Path(ARGS.out)
sys.argv=[str(R/'cache_engine.py'),'--mode',ARGS.mode,'--phase','solo' if ARGS.prepare_only else 'co','--out',str(OUT),'--family','lora32_heads','--lr','5e-5','--seed','20261108','--sequence','/home/hengyi/Workspace/01_VLA-Serving/Experiments/pi05_a1_baseline_20261108/sequence.json','--split','/home/hengyi/Workspace/01_VLA-Serving/Experiments/pi05_extended_20261109/splits/drawer.json','--perturb','/home/hengyi/Workspace/01_VLA-Serving/Experiments/pi05_extended_20261109/tasks/drawer_lowlight_0.json']
import cache_engine as base
from cache_engine import torch,np,save,event,obs_map,cat_obs,tree_map,tree_flatten,stat,append,GPU
from ipc import read_slot
class Streaming(base.Engine):
 def load_cpu(self):
  source=json.loads((R/'sources.json').read_text());self.new_sources=[tuple(x) for x in source['new']];self.replay=[tuple(x) for x in source['replay']]
  if ARGS.smoke:self.new_sources=self.new_sources[:12];self.replay=self.replay[:8]
  self.raw_inputs={};self.new_source_set=set(self.new_sources);self.keys=list(dict.fromkeys(self.new_sources+self.replay));self.micros=[self.keys[:4],self.replay[:4]]*3;super().load_cpu()
  dest=OUT/'inputs.pt';tmp=dest.with_suffix('.tmp')
  torch.save([{'key':k,'raw':self.raw_inputs[k]} for k in self.new_sources],tmp);tmp.replace(dest);self.raw_inputs.clear()
 def init_stream(self):
  self.recent=collections.OrderedDict();self.retired=[];self.arrivals=[];self.incoming=queue.Queue(maxsize=32);self.stop=threading.Event();self.producer_error=None;self.max_recent=8 if ARGS.smoke else int(os.environ.get('RECENT_CAPACITY','512'));self.sample_rng=np.random.default_rng(20261108);self.micros=[];self.mi=0;self.prefetched=None;self.measure_origin=None;self.measure_end=None
  if self.mode!='A0':
   for i,key in enumerate(self.replay):
    with torch.no_grad():
     obs=obs_map(lambda x:x.to('cuda',non_blocking=True),self.data[key][0]);_,mask,kv=self.model.build_prefix_cache(obs);self.store(key,(mask,kv))
    if i%16==15:self.copy.synchronize()
   self.copy.synchronize()
  save(OUT/'replay_warmup.json',{'cached_unique_windows':len(self.cache),'configured_windows':len(self.replay),'mode':self.mode});self.hits=self.misses=0
 def receive(self):
  try:
   index=0
   while not self.stop.is_set():
    p=OUT/'ipc'/f'{index:08d}.json'
    if not p.exists():self.stop.wait(.002);continue
    record=json.loads(p.read_text())
    if record.get('mode')=='baseline':p.unlink();index+=1;self.receive_index=index;continue
    record['source']=tuple(record['source']);record['key']=(*record['source'],index);record['sample_count']=0;record['payload']=None;t0=time.monotonic_ns()
    if record.get('mode')=='A2':
     payload=read_slot(OUT,record['slot']);record['payload']=payload
    record['receiver_ready_ns']=time.monotonic_ns();record['receiver_host_store_ms']=(record['receiver_ready_ns']-t0)/1e6;record['receiver_lag_ms']=(record['receiver_ready_ns']-record['nominal_ns'])/1e6
    put_start=time.monotonic_ns();self.incoming.put(record,timeout=60);append(OUT/'receiver_events.jsonl',dict(index=index,host_store_ms=record['receiver_host_store_ms'],queue_wait_ms=(time.monotonic_ns()-put_start)/1e6,queue_size=self.incoming.qsize()));p.unlink();index+=1;self.receive_index=index
  except Exception:self.producer_error=traceback.format_exc();save(OUT/'receiver_failure.json',{'traceback':self.producer_error,'receive_index':self.receive_index,'queue_size':self.incoming.qsize()});self.stop.set()
 def ingest(self):
  if self.producer_error:raise RuntimeError(self.producer_error)
  while True:
   try:record=self.incoming.get_nowait()
   except queue.Empty:break
   key=record['key'];record['visible_ns']=time.monotonic_ns();self.data[key]=self.data[record['source']];payload=record.pop('payload');self.recent[key]=record;self.arrivals.append(record)
   if payload is not None:
    done=event();done.record(self.copy);self.cache[key]=(payload,done)
    if not getattr(self,'service_parity_done',False):
     with torch.no_grad():
      obs=obs_map(lambda x:x.to('cuda',non_blocking=True),self.data[key][0]);_,mask,kv=self.model.build_prefix_cache(obs)
      ref=torch.load(OUT/'service_reference.pt',weights_only=False,map_location='cpu');checks={}
      for label,other in [('graph',ref['graph']),('eager',ref['eager']),('ipc',payload)]:
       checks[label]=[{'shape':list(a.shape),'equal':torch.equal(a.cpu(),b),'max_abs':float((a.float().cpu()-b.float()).abs().max())} for a,b in zip(tree_flatten((mask,kv))[0],tree_flatten(other)[0])]
      checks['input']={f.name:all(torch.equal(a.cpu(),b) for a,b in zip(tree_flatten(getattr(obs,f.name))[0],tree_flatten(ref['obs'][f.name])[0]) if torch.is_tensor(a)) for f in dataclasses.fields(obs)}
      checks['tf32']=torch.backends.cuda.matmul.allow_tf32;save(OUT/'cross_process_diagnostics.json',checks)
      for a,b in zip(tree_flatten(ref['graph'])[0],tree_flatten(payload)[0]):torch.testing.assert_close(a,b,rtol=0,atol=0)
      assert all(checks['input'].values()),'serving and training inputs differ'
     self.service_parity_done=True;save(OUT/'service_cache_parity.json',{'transport_bitwise_equal':True,'recompute_bitwise_equal':all(x['equal'] for x in checks['graph']),'source':record['source'],'note':'Cross-MPS30/100 arithmetic differs; matched-SM30 control is bitwise equal.'})
   while len(self.recent)>self.max_recent:
    old,rec=self.recent.popitem(last=False);self.cache.pop(old,None);self.data.pop(old,None);rec['evicted_ns']=time.monotonic_ns();self.retired.append(rec)
 def loss(self,p):
  wa,wb=event(),event();wa.record(self.stream);self.stream.wait_event(p['done']);wb.record(self.stream);p['wait_events']=[(wa,wb)];values=[];p['d2h']=[];p['prefix_events']=[];p['class_counts']={'new':{'hits':0,'misses':0},'replay':{'hits':0,'misses':0}}
  for j,item in enumerate(self.micros[p['mi']]):
   key=tuple(item);category='new' if len(key)==3 else 'replay';val=p['prefixes'][j];obs=p['parts'][j][0]
   if self.mode!='A0' and (val is not None or key in self.cache):
    self.hits+=1;p['class_counts'][category]['hits']+=1
    if val is None:
     val,done=self.fetch(key,p['records']);a,b=event(),event();a.record();self.stream.wait_event(done);b.record();p['wait_events'].append((a,b))
   else:
    self.misses+=1;p['class_counts'][category]['misses']+=1;a,b=event(),event();a.record()
    with torch.no_grad():_,mask,kv=self.model.build_prefix_cache(obs)
    b.record();p['prefix_events'].append((a,b,category));val=(mask,kv);
    if self.mode!='A0':p['d2h'].append(self.store(key,val))
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

STATE_FIELDS=['cache','recent','retired','arrivals','micros','mi','step_id','measure_origin','measure_end','hits','misses']
def snapshot(tr):
 return {'masters':[x.detach().cpu().clone() for x in tr.masters],'ema':[x.detach().cpu().clone() for x in tr.ema],'optim':tree_map(lambda x:x.detach().cpu().clone() if torch.is_tensor(x) else x,tr.optim.state_dict())}
def restore(tr,state):
 tr.optim.load_state_dict(state['optim'])
 with torch.no_grad():
  for (n,p),master,ema,m,e in zip(tr.pairs,tr.masters,tr.ema,state['masters'],state['ema']):master.copy_(m);ema.copy_(e);p.copy_(master)
 tr.model.zero_grad(set_to_none=True);tr.optim.zero_grad(set_to_none=True);tr.stream.synchronize()
def activate(tr,cfg):
 import copy
 limit=cfg.get('drain_to',0);until=time.monotonic()+120
 while tr.receive_index<limit:
  tr.ingest()
  if tr.producer_error:raise RuntimeError(tr.producer_error)
  assert time.monotonic()<until,'IPC drain timed out'
  time.sleep(.002)
 tr.ingest();tr.copy.synchronize()
 if tr.active_variant is not None:
  tr.variants[tr.active_variant]={'fields':{k:getattr(tr,k) for k in STATE_FIELDS},'rng':copy.deepcopy(tr.sample_rng.bit_generator.state),'weights':snapshot(tr)}
 if cfg['mode']=='baseline':tr.active_variant=None;return
 key=cfg['variant'];tr.mode=cfg['mode']
 if key in tr.variants:
  state=tr.variants[key]
  for k,v in state['fields'].items():setattr(tr,k,v)
  tr.sample_rng.bit_generator.state=copy.deepcopy(state['rng']);restore(tr,state['weights'])
 else:
  tr.cache={} if tr.mode=='A0' else dict(tr.replay_cache);tr.recent=collections.OrderedDict();tr.retired=[];tr.arrivals=[];tr.micros=[];tr.mi=0;tr.step_id=0;tr.measure_origin=None;tr.measure_end=None;tr.hits=tr.misses=0;tr.sample_rng=np.random.default_rng(20261108);restore(tr,tr.initial_weights)
 tr.prefetched=None;tr.current_parts=[];tr.active_variant=key

def wait_file(p):
 while not p.exists():time.sleep(.02)
 return json.loads(p.read_text())
def main():
 tr=Streaming(ARGS.mode)
 if ARGS.prepare_only:save(OUT/'status.json',{'state':'complete','samples':len(tr.new_sources),'raw_input_format':2});return
 tr.total_steps=4000;assert sum(p.numel() for _,p in tr.pairs)==20876320
 tr.current_parts=[];tr.parity();tr.current_parts=[]
 if os.environ.get('EXPERT_COUNT'):
  if os.environ.get('DYNAMIC_BANK')=='1':from dynamic_experts import install
  else:from experts import install
  bank=install(tr,sys.modules[__name__])
  for key in set(tr.micros[0]):
   with torch.no_grad():
    _,mask,kv=tr.model.build_prefix_cache(obs_map(lambda x:x.cuda(),tr.data[key][0]));tr.store(key,(mask,kv))
  bank.validate();tr.cache={};tr.hits=tr.misses=0
 tr.init_stream();tr.replay_cache=dict(tr.cache);tr.initial_weights=snapshot(tr);tr.variants={};tr.active_variant=None;tr.receive_index=0
 save(OUT/'train_config.json',{'family':'lora32_heads','parameters':20876320,'lr':5e-5,'ema':.999,'rank':32,'microbatch':4,'accumulation':2,'trainable_modules':[n for n,p in tr.pairs]})
 thread=threading.Thread(target=tr.receive,daemon=True);thread.start();save(OUT/'train_ready.json',{'ready':True})
 try:
  for index in range(200):
   cfg=wait_file(OUT/f'command_{index:03d}.json')
   if cfg['op']=='stop':break
   if cfg['op']=='activate':
    activate(tr,cfg);save(OUT/f'train_done_{index:03d}.json',{'complete':True,'variant':tr.active_variant,'cache_samples':len(tr.cache),'step':tr.step_id});continue
   if cfg['op']=='profile':
    from kernel_profile import profile_training
    profile_training(tr,OUT/cfg['name']);save(OUT/f'train_done_{index:03d}.json',{'complete':True});continue
   if cfg['op']=='calibrate':save(OUT/f'train_done_{index:03d}.json',{'complete':True});continue
   if os.environ.get('SLO_FEEDBACK')=='1':
    if os.environ.get('DYNAMIC_BANK')=='1':
     from adaptive_feedback import Feedback
     tr.feedback=Feedback(OUT)
    else:
     from feedback import Feedback
     if not hasattr(tr,'feedback'):tr.feedback=Feedback(OUT)
   if cfg.get('reset_expert_states'):
    restore(tr,tr.initial_weights);tr.step_id=0
   origin=cfg['origin_ns'];end=origin+round(cfg['duration_s']*1e9);out=OUT/cfg['name'];out.mkdir(exist_ok=True);cap=cfg.get('rate_cap',0);next_step=origin
   time.sleep(max(0,(origin-time.monotonic_ns())/1e9));rows=[]
   if cfg.get('round')==1 and cfg['mode']!='baseline':tr.measure_origin=origin
   if os.environ.get('ASYNC_TRAIN')=='1' and cfg['mode']=='A2':
    from async_runner import run_window
    rows=run_window(tr,cfg,out)
   else:
    while time.monotonic_ns()<end:
     if cfg['mode']=='baseline':time.sleep(min(.05,max(0,(end-time.monotonic_ns())/1e9)));continue
     tr.ingest()
     if cfg.get('inference_only'):
      time.sleep(.002);continue
     if cap and time.monotonic_ns()<next_step:time.sleep(min(.002,(next_step-time.monotonic_ns())/1e9));continue
     if os.environ.get('SLO_FEEDBACK')=='1' and not tr.feedback.ready(out):
      time.sleep(.002);continue
     if os.environ.get('DYNAMIC_BANK')=='1':
      tr.bank.n=tr.feedback.experts if os.environ.get('SLO_FEEDBACK')=='1' else int(os.environ.get('INITIAL_EXPERTS','1'))
     row=tr.next_step()
     if row is not None and os.environ.get('DYNAMIC_BANK')=='1':
      row['duty']=tr.feedback.duty if os.environ.get('SLO_FEEDBACK')=='1' else 1.
      row['control_window_end_ns']=tr.feedback.last_window if os.environ.get('SLO_FEEDBACK')=='1' else 0
     if row is None:time.sleep(.002);continue
     if cap:next_step=max(next_step+round(1e9/cap),row['start_ns']+round(1e9/cap))
     rows.append(row);append(out/'steps.jsonl',row)
     if os.environ.get('SLO_FEEDBACK')=='1':
      tr.feedback.after_step(row,out)

   valid=[x for x in rows if origin<=x['start_ns'] and x['finish_ns']<=end];counts={c:{k:sum(x['class_counts'][c][k] for x in valid) for k in ['hits','misses']} for c in ['new','replay']};wall=sum(x['seconds'] for x in valid)*1000;prefix={c:sum(x['prefix_compute_ms'][c] for x in valid) for c in ['new','replay']}
   save(out/'train_summary.json',{'complete_updates':len(valid),'steps_per_s':len(valid)/cfg['duration_s'],'class_counts':counts,'hit_rates':{c:counts[c]['hits']/max(1,sum(counts[c].values())) for c in counts},'prefix_compute_ms':prefix,'prefix_compute_fraction_of_step_wall':{c:prefix[c]/max(wall,1e-12) for c in prefix},'h2d_ms':stat([x['h2d_ms'] for x in valid]),'new_pool_size':len(tr.recent),'peak_cuda_bytes':max([x['peak_cuda_bytes'] for x in valid],default=0),'rate_cap':cap})
   if cfg.get('round') and cfg['mode']!='baseline':save(out/'cohort_summary.json',tr.cohort(end))
   # The server may still finish queued requests after the training window ends.
   # Drain host arrivals without training so the bounded queue/ring cannot deadlock.
   drain_start=time.monotonic_ns();drain_count=tr.receive_index;server_done=OUT/f'server_done_{index:03d}.json'
   while True:
    tr.ingest()
    if server_done.exists():
     server_info=json.loads(server_done.read_text())
     if tr.receive_index>=server_info.get('next_id',tr.receive_index):break
    if time.monotonic_ns()-drain_start>180e9:raise TimeoutError('post-window server drain timeout')
    time.sleep(.002)
   save(out/'post_window_drain.json',{'seconds':(time.monotonic_ns()-drain_start)/1e9,'received':tr.receive_index-drain_count,'pending_queue':tr.incoming.qsize(),'training_updates_outside_window':0})
   save(OUT/f'train_done_{index:03d}.json',{'complete':True})
  tr.stop.set();thread.join(timeout=5)
  if tr.producer_error:raise RuntimeError(tr.producer_error)
  tr.copy.synchronize();save(OUT/'train_status.json',{'state':'complete'})
 finally:tr.stop.set()
if __name__=='__main__':
 try:main()
 except Exception:save(OUT/'train_status.json',{'state':'failed','traceback':traceback.format_exc()});raise
