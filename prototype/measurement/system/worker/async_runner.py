"""Bounded two-update pipeline; completion polling, host-pinned stats and next-batch DMA.
CUDA event spans include serving interference; GIL probes sample explicit acquisition latency.
"""
import time,math,collections,threading,json
import torch
from gil_probe import sample as gil_sample
from cache_engine import event,tree_map,tree_flatten,append,save
class Runner:
 def __init__(self,tr):
  self.tr=tr;self.pending=collections.deque();self.next_batch=None;self.stats_stream=torch.cuda.Stream(priority=0);self.depth=2;self.completed=0
  # Same AdamW math/hyperparameters and master precision, batched tensor operations.
  for group in tr.optim.param_groups:group['foreach']=True
  self.slot=0
 def prefetch(self):
  tr=self.tr;tr.ingest()
  if not tr.recent:return None
  begin=time.monotonic_ns();new=list(tr.recent);batch=[new[int(tr.sample_rng.integers(len(new)))] for _ in range(4)]+[tr.replay[int(tr.sample_rng.integers(len(tr.replay)))] for _ in range(4)];tr.sample_rng.shuffle(batch)
  sample_records=[tr.recent[k] for k in batch if len(k)==3]
  mi=len(tr.micros);tr.micros.extend([batch[:4],batch[4:]])
  ps=[tr.prepare(mi),tr.prepare(mi+1)]
  for p,keys in zip(ps,[batch[:4],batch[4:]]):p['pinned_prefix_owners']=[tr.cache[k][0] for k in keys]
  assert all(x is not None for p in ps for x in p['prefixes']),'A2 async benchmark requires resident prefetched prefixes'
  return dict(parts=ps,mi=mi,host_prepare_ms=(time.monotonic_ns()-begin)/1e6,draw_ns=begin,keys=batch,sample_records=sample_records)
 def submit(self,batch):
  tr=self.tr
  for rec in batch['sample_records']:rec['sample_count']+=1
  start=time.monotonic_ns();cpu=time.thread_time_ns();gil=gil_sample();begin,end=event(),event();opt_a,opt_b=event(),event();forward=[];backward=[];losses=[];h0,m0=tr.hits,tr.misses
  with torch.cuda.stream(tr.stream):
   begin.record();tr.current_parts=[]
   for p in batch['parts']:
    fa,fb,ba,bb=event(),event(),event(),event();fa.record();loss=tr.loss(p);fb.record();ba.record();(loss/2).backward();bb.record();forward.append((fa,fb));backward.append((ba,bb));losses.append(loss.detach())
   opt_a.record()
   for (_,q),master in zip(tr.pairs,tr.masters):master.grad=q.grad.detach().float()
   norm=torch.nn.utils.clip_grad_norm_(tr.masters,1.,error_if_nonfinite=False,foreach=True)
   step=tr.step_id+1;lr=tr.learning_rate*(step/50 if step<=50 else .1+.9*.5*(1+math.cos(math.pi*min(1.,(step-50)/(tr.total_steps-50)))))
   tr.optim.param_groups[0]['lr']=lr;tr.optim.step()
   with torch.no_grad():
    torch._foreach_copy_([p for _,p in tr.pairs],tr.masters);torch._foreach_mul_(tr.ema,.999);torch._foreach_add_(tr.ema,tr.masters,alpha=.001)
   tr.optim.zero_grad(set_to_none=True);tr.model.zero_grad(set_to_none=True);opt_b.record();values=torch.stack([torch.stack(losses).mean(),norm]);end.record()
  host=torch.empty((2,),dtype=torch.float32,pin_memory=True);ready=event()
  with torch.cuda.stream(self.stats_stream):
   self.stats_stream.wait_event(end);host.copy_(values,non_blocking=True);values.record_stream(self.stats_stream);ready.record()
  tr.step_id=step;tr.mi=batch['mi']+2
  item=dict(step=step,start_ns=start,begin=begin,end=end,ready=ready,host=host,device_stats=values,batch=batch,forward=forward,backward=backward,opt=(opt_a,opt_b),hits=tr.hits-h0,misses=tr.misses-m0,learning_rate=lr,gil_acquire_sample_ms=gil,host_launch_wall_ms=(time.monotonic_ns()-start)/1e6,host_thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6,pending_at_submission=len(self.pending),class_counts={c:{k:sum(p['class_counts'][c][k] for p in batch['parts']) for k in ['hits','misses']} for c in ['new','replay']})
  self.pending.append(item)
 def harvest(self):
  rows=[]
  while self.pending and self.pending[0]['ready'].query():
   x=self.pending.popleft();finish=time.monotonic_ns();loss,norm=x['host'].tolist();assert math.isfinite(loss) and math.isfinite(norm),'nonfinite async update'
   ps=x['batch']['parts'];row={k:x[k] for k in ['step','start_ns','hits','misses','learning_rate','gil_acquire_sample_ms','host_launch_wall_ms','host_thread_cpu_ms','pending_at_submission','class_counts']}
   row.update(finish_ns=finish,compute_finish_ns=finish,seconds=(finish-x['start_ns'])/1e9,loss=loss,gradient_norm=norm,cuda_step_span_ms=x['begin'].elapsed_time(x['end']),forward_including_h2d_wait_ms=sum(a.elapsed_time(b) for a,b in x['forward']),backward_ms=sum(a.elapsed_time(b) for a,b in x['backward']),optimizer_ema_ms=x['opt'][0].elapsed_time(x['opt'][1]),h2d_ms=sum(a.elapsed_time(b) for p in ps for a,b,_ in p['records']),prefix_h2d_ms=sum(a.elapsed_time(b) for p in ps for a,b,k in p['records'] if k=='prefix'),input_h2d_ms=sum(a.elapsed_time(b) for p in ps for a,b,k in p['records'] if k=='input'),compute_wait_for_h2d_ms=sum(a.elapsed_time(b) for p in ps for a,b in p['wait_events']),host_prefetch_prepare_ms=x['batch']['host_prepare_ms'],prefix_compute_ms={'new':0.,'replay':0.},peak_cuda_bytes=torch.cuda.max_memory_allocated(),completion_time_kind='host_event_query_upper_bound',data_sampled_ns=x['batch']['draw_ns'],training_stream=int(self.tr.stream.cuda_stream),training_stream_priority=self.tr.stream.priority)
   rows.append(row);self.completed+=1
  return rows
 def run(self,cfg,out):
  tr=self.tr;origin=cfg['origin_ns'];end=origin+round(cfg['duration_s']*1e9);rows=[];polls=0;sleeps=0;cpu_start=time.thread_time_ns();thread_start=time.monotonic_ns()
  if not hasattr(tr,'async_config_written'):
   save(out.parent/'async_config.json',dict(pipeline_depth=self.depth,training_stream=int(tr.stream.cuda_stream),copy_stream=int(tr.copy.cuda_stream),stats_stream=int(self.stats_stream.cuda_stream),priorities=[tr.stream.priority,tr.copy.priority,self.stats_stream.priority],training_thread_id=threading.get_native_id(),event_poll_sleep_s=.0005,optimizer_foreach=True,per_step_device_synchronize=False,prefetch='next complete batch, two microbatches, from already received samples',gil_metric='sampled explicit PyGILState_Ensure acquisition latency; not total GIL waiting time',stats_transfer='separate pinned-host async D2H stream'))
   tr.async_config_written=True
  while time.monotonic_ns()<end or self.pending:
   tr.ingest();fresh=self.harvest()
   for row in fresh:rows.append(row);append(out/'steps.jsonl',row)
   now=time.monotonic_ns()
   if now>=end:
    self.next_batch=None
   elif self.next_batch is None:
    self.next_batch=self.prefetch()
   if now<end and len(self.pending)<self.depth and self.next_batch is not None:
    self.submit(self.next_batch);self.next_batch=None
    # Launch next-batch DMA immediately while the just-submitted update is still on the GPU.
    if time.monotonic_ns()<end:self.next_batch=self.prefetch()
   else:time.sleep(.0005);sleeps+=1
   polls+=1
  save(out/'async_window.json',dict(polls=polls,sleeps=sleeps,thread_cpu_ms=(time.thread_time_ns()-cpu_start)/1e6,host_span_ms=(time.monotonic_ns()-thread_start)/1e6,pending_at_end=len(self.pending),gil_sampling='one explicit acquisition probe per update'))
  return rows
def run_window(tr,cfg,out):
 if not hasattr(tr,'async_runner'):tr.async_runner=Runner(tr)
 return tr.async_runner.run(cfg,out)
