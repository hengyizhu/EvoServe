import argparse,math,queue,traceback,gc
from core import *

def schedule(period,phase,duration):
 rows=[]
 for client,offset in enumerate([0,0 if phase=='sync' else period/2]):
  k=0
  while offset+k*period<duration-1e-8:
   rows.append({'client_id':client,'offset':offset+k*period});k+=1
 rows.sort(key=lambda r:(r['offset'],r['client_id']))
 return [{**r,'request_id':i} for i,r in enumerate(rows)]

def edf_admit(now,cost,inference_cost,margin,known):
 clock=now+cost
 for release,deadline in sorted(known,key=lambda x:(x[1],x[0])):
  clock=max(clock,release)+inference_cost
  if clock+margin>deadline:return False
 return True

def run_window(server,trainer,policy,period,phase,duration,calibration,run_id,uuid,monitor=True,client_count=2,deadline_period=None):
 spec=schedule(period,phase,duration) if client_count==2 else [{'client_id':0,'offset':k*period,'request_id':k} for k in range(math.ceil(duration/period))]
 q=queue.Queue();stop=threading.Event();origin=stamp()+200_000_000;end=origin+int(duration*1e9)
 for r in spec:r.update(nominal_release_ns=origin+round(r['offset']*1e9),deadline_ns=origin+round((r['offset']+(deadline_period or period))*1e9))
 def generator():
  for r in spec:
   if stop.wait(max(0,(r['nominal_release_ns']-stamp())/1e9)):break
   item=dict(r);item['actual_enqueue_ns']=stamp();q.put(item)
 gen=threading.Thread(target=generator,daemon=True);gen.start()
 pending=[];completed=set();requests=[];segments=[];policy_ns=0;last_train_end=None
 torch.cuda.reset_peak_memory_stats()
 with Monitor() as memory:
  while stamp()<end+60_000_000_000:
   while True:
    try:pending.append(q.get_nowait())
    except queue.Empty:break
   now=stamp()
   if pending:
    pending.sort(key=lambda r:(r['deadline_ns'],r['request_id']));r=pending.pop(0)
    _,timing=server.infer(r['request_id']);r.update(timing)
    r['status']='completed_late' if r['finish_ns']>r['deadline_ns'] else 'completed'
    r.update(run_id=run_id,gpu_uuid=uuid,policy=policy,workload=phase)
    requests.append(r);completed.add(r['request_id']);continue
   if now>=end and len(completed)==len(spec):break
   if now<origin or now>=end or policy=='S0':time.sleep(.001);continue
   tick=stamp();split=policy in ('S3','S4')
   kind=trainer.next_kind if split else ('T_MICRO_OPT' if trainer.micro==7 else 'T_MICRO')
   allowed=True
   if policy in ('S2','S4'):
    # Same two-period finite lookahead and common inference/margin estimates in S2/S4.
    known=[((r['nominal_release_ns']-origin)/1e9,(r['deadline_ns']-origin)/1e9) for r in spec if r['request_id'] not in completed and r['nominal_release_ns']<=now+round(2*period*1e9)]
    allowed=edf_admit((now-origin)/1e9,calibration['costs_s'][kind],calibration['inference_mean_s'],calibration['margin_s'],known)
   policy_ns+=stamp()-tick
   if allowed:
    row=trainer.segment(split);row.update(run_id=run_id,policy=policy)
    row['in_measurement_window']=row['cpu_submit_ns']>=origin and row['gpu_done_ns']<=end
    segments.append(row);last_train_end=row['gpu_done_ns']
   else:time.sleep(.001)
 stop.set();gen.join(2)
 for r in spec:
  if r['request_id'] not in completed:
   # Preserve enqueue time even for requests left in the queue at the drain cap.
   while True:
    try:pending.append(q.get_nowait())
    except queue.Empty:break
   actual=next((x.get('actual_enqueue_ns') for x in pending if x['request_id']==r['request_id']),None)
   requests.append({**r,'actual_enqueue_ns':actual,'start_ns':None,'finish_ns':None,'status':'unfinished','run_id':run_id,'gpu_uuid':uuid,'policy':policy,'workload':phase})
 late=sum(r['status']!='completed' for r in requests)
 latencies=[(r['finish_ns']-r['nominal_release_ns'])/1e6 for r in requests if r['finish_ns'] is not None]
 updates=sum(r['optimizer_completed'] for r in segments if r['in_measurement_window'])
 samples=updates*8/duration
 summary={'run_id':run_id,'gpu_uuid':uuid,'profile':trainer.tier,'policy':policy,'phase_pattern':phase,'n_requests':len(spec),
  'p50_ms':float(np.percentile(latencies,50)) if latencies else None,'p95_ms':float(np.percentile(latencies,95)) if latencies else None,'p99_ms':float(np.percentile(latencies,99)) if latencies else None,
  'deadline_miss_rate':late/len(spec),'unfinished':len(spec)-len(completed),'completed_updates':updates,'effective_samples_per_s':samples,
  'train_retention':samples/calibration['training_samples_per_s'],'peak_device_gib':max(r['device_bytes'] for r in memory.rows)/2**30,
  'peak_process_gib':max(r['process_bytes'] for r in memory.rows)/2**30,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
  'policy_overhead_ms':policy_ns/1e6,'period_s':period,'measurement_seconds':duration,'origin_ns':origin,'window_end_ns':end,
  'drain_seconds':max(0,(max([r['finish_ns'] for r in requests if r['finish_ns'] is not None]+[end])-end)/1e9),
  'drain_completed_updates':sum(r['optimizer_completed'] for r in segments if not r['in_measurement_window']),
  'last_partial_microbatch_count':trainer.micro,'last_partial_stage':trainer.next_kind,
  'max_enqueue_lag_ms':max([(r['actual_enqueue_ns']-r['nominal_release_ns'])/1e6 for r in requests if r.get('actual_enqueue_ns') is not None]+[0])}
 assert len(requests)==len(spec) and len({r['request_id'] for r in requests})==len(spec)
 return summary,requests,segments,memory.rows

def calibrate(server,trainer,out):
 save(out/'status.json',{'state':'correctness','profile':trainer.tier})
 equivalence=server.verify();save(out/'inference_correctness.json',equivalence)
 trainer.correctness(out)
 save(out/'status.json',{'state':'calibrating','profile':trainer.tier})
 trainer.reset()
 for _ in range(5*8):trainer.segment(False)
 trainrows=[];trainer.reset()
 for _ in range(30*8):trainrows.append(trainer.segment(False))
 wall=(trainrows[-1]['gpu_done_ns']-trainrows[0]['cpu_submit_ns'])/1e9
 costs={k:stats([r['wall_ms']/1000 for r in trainrows if r['segment_kind']==k]) for k in ['T_MICRO','T_MICRO_OPT']}
 splitrows=[];trainer.reset()
 while trainer.iteration<30:splitrows.append(trainer.segment(True))
 for k in ['T_FWD','T_BWD','T_OPT']:costs[k]=stats([r['wall_ms']/1000 for r in splitrows if r['segment_kind']==k])
 trainer.reset()
 for i in range(20):server.infer(i)
 inferrows=[server.infer(i)[1] for i in range(120)]
 inference=stats([r['service_ms']/1000 for r in inferrows])
 result={'inference':inference,'inference_mean_s':inference['mean'],'margin_s':max(.010,inference['p95']-inference['mean']),
  'costs':costs,'costs_s':{k:v['p95'] for k,v in costs.items()},'training_samples_per_s':30*8/wall,
  'training_updates':30,'infer_requests':120,'profile':trainer.tier,'physical_gpu':GPU,
  'residency':'Independent inference model, training model and initialized optimizer remain resident for all policies, including S0.'}
 save(out/'calibration.json',result);write_csv(out/'calibration_inference.csv',inferrows);write_csv(out/'calibration_segments.csv',trainrows+splitrows)
 return result

def execute_condition(server,trainer,policy,period,phase,calibration,run_id,uuid,out,load,repeat):
 trainer.reset()
 run_window(server,trainer,policy,period,phase,20,calibration,run_id+'_warmup',uuid)
 trainer.reset()
 summary,requests,segments,memory=run_window(server,trainer,policy,period,phase,180,calibration,run_id,uuid)
 summary.update(load=load,repeat=repeat,production_check=server.unchanged())
 dest=out/'runs'/run_id;dest.mkdir(parents=True,exist_ok=True)
 write_csv(dest/'requests.csv',requests);write_csv(dest/'train_segments.csv',segments,fields=['run_id','policy','iteration_id','microbatch_id','segment_kind','cpu_submit_ns','gpu_done_ns','wall_ms','cuda_ms','optimizer_completed','in_measurement_window','memory_allocated','memory_reserved'])
 write_csv(dest/'memory.csv',memory);save(dest/'summary.json',summary)
 return summary

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tier',required=True,choices=['L2','L3']);args=ap.parse_args()
 out=ROOT/f'gpu{GPU}_{args.tier}';out.mkdir(exist_ok=True);server=trainer=None
 try:
  save(out/'status.json',{'state':'loading','pid':os.getpid(),'profile':args.tier})
  server=Serving();trainer=Training(args.tier)
  import pynvml as nv
  nv.nvmlInit();uuid=nv.nvmlDeviceGetUUID(nv.nvmlDeviceGetHandleByIndex(GPU));nv.nvmlShutdown()
  calibration=calibrate(server,trainer,out)
  trainer.reset()
  rt,req,seg,mem=run_window(server,trainer,'S0',.250,'single_client',30,calibration,'real_250ms',uuid,client_count=1)
  rt['original_configuration_realtime_feasible_in_this_sample']=rt['deadline_miss_rate']==0
  save(out/'real_250ms.json',rt);write_csv(out/'real_250ms_requests.csv',req)
  save(out/'status.json',{'state':'waiting_common_calibration','completed':0})
  deadline=time.monotonic()+3600
  while not (ROOT/'frozen_workloads.json').exists():
   if (ROOT/'STOP').exists():raise RuntimeError('Coordinator stopped preflight')
   if time.monotonic()>deadline:raise TimeoutError('Common calibration not available')
   time.sleep(2)
  config=json.loads((ROOT/'frozen_workloads.json').read_text())
  # Both admission policies use the same conservative shared cost estimates, frozen before matrix.
  calibration.update(inference_mean_s=config['common_c_s'],margin_s=config['common_margin_s'])
  results=[]
  for repeat in range(1,4):
   policies=['S0','S1','S2','S3','S4'];policies=policies[repeat-1:]+policies[:repeat-1]
   for load in [.4,.7]:
    period=config['periods'][str(load)]
    for phase in ['sync','staggered']:
     for policy in policies:
      run_id=f'{args.tier}_gpu{GPU}_r{repeat}_{load}_{phase}_{policy}'
      save(out/'status.json',{'state':'running','run_id':run_id,'completed':len(results),'total':60})
      row=execute_condition(server,trainer,policy,period,phase,calibration,run_id,uuid,out,load,repeat)
      results.append(row);write_csv(out/'summary.csv',results);save(out/'summary.json',results)
  integrity={'production':server.unchanged(full=True),'frozen_training_unchanged':trainer.check_frozen(),'completed':len(results)}
  save(out/'correctness.json',integrity)
  if GPU==2:
   trainer.reset();del trainer;gc.collect();torch.cuda.empty_cache();trainer=Training('L2')
   extra=out/'L2_same_card';extra.mkdir(exist_ok=True)
   extra_cal=calibrate(server,trainer,extra);extra_cal.update(inference_mean_s=config['common_c_s'],margin_s=config['common_margin_s'])
   for repeat in range(1,4):
    for policy in ['S0','S1','S2','S3','S4']:
     run_id=f'L2_gpu2_extra_r{repeat}_0.7_staggered_{policy}'
     save(out/'status.json',{'state':'same_card_L2','run_id':run_id,'completed':len(results),'total':75})
     row=execute_condition(server,trainer,policy,config['periods']['0.7'],'staggered',extra_cal,run_id,uuid,extra,.7,repeat)
     results.append(row);write_csv(out/'summary.csv',results);save(out/'summary.json',results)
   save(extra/'correctness.json',{'production':server.unchanged(full=True),'frozen_training_unchanged':trainer.check_frozen()})
  save(out/'status.json',{'state':'complete','completed':len(results)})
 except Exception:
  save(out/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
