import os,sys,time,json,pathlib,argparse,traceback,importlib.util,subprocess
R=pathlib.Path('/home/hengyi/pi05_review_20261107')
sys.path.insert(0,'/home/hengyi/pi05_optimized_20261107')
ap=argparse.ArgumentParser();ap.add_argument('--role',choices=['server','train','segmented'],required=True);ap.add_argument('--vision',action='store_true');ap.add_argument('--out',required=True);args=ap.parse_args();OUT=pathlib.Path(args.out);OUT.mkdir(parents=True,exist_ok=True)
if os.environ.get('PI_MPS')=='1':
 os.environ['PI_MPS_UUID']=subprocess.check_output(['nvidia-smi','-i',os.environ['VLA_GPU'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();os.environ['CUDA_VISIBLE_DEVICES']=os.environ['PI_MPS_UUID'];spec=importlib.util.spec_from_file_location('core',R/'core_mps.py');module=importlib.util.module_from_spec(spec);sys.modules['core']=module;spec.loader.exec_module(module)
from review_graph_serving import GraphServing,torch,np,save,stats,stamp,GPU,write_csv
sys.path.insert(0,str(R))
sys.path.insert(0,'/home/hengyi/pi05_cache_20261107')
from cache_train import CacheTrain
os.sched_setaffinity(0,set(range(GPU*8+(4 if args.role=='train' else 0),GPU*8+(8 if args.role=='train' else 4))))
torch.set_num_threads(4)
def pin_all_threads():
 cpus=set(range(GPU*8+(4 if args.role=='train' else 0),GPU*8+(8 if args.role=='train' else 4)))
 for tid in pathlib.Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(tid.name),cpus)
  except ProcessLookupError:pass
 return {t.name:sorted(os.sched_getaffinity(int(t.name))) for t in pathlib.Path('/proc/self/task').iterdir()}
pin_all_threads()
def wait_file(p):
 while not p.exists():time.sleep(.02)
 return json.loads(p.read_text())
def schedule(cfg):
 rows=[];origin=cfg['origin_ns'];period=cfg['period_s'];n=cfg['clients']
 for client in range(n):
  phase=0 if cfg['pattern']=='sync' else client*period/n
  for j in range(round(cfg['duration_s']/period)):
   release=origin+round((j*period+phase)*1e9);rows.append({'client_id':client,'nominal_release_ns':release,'deadline_ns':release+round(cfg['deadline_s']*1e9)})
 rows.sort(key=lambda x:(x['nominal_release_ns'],x['client_id']));return [dict(x,request_id=i) for i,x in enumerate(rows)]
def summarize(rows,cfg,cal):
 complete=[r for r in rows if r.get('finish_ns') is not None];unfinished=len(rows)-len(complete);end=cfg['origin_ns']+round(cfg['duration_s']*1e9);during=[r for r in complete if r['start_ns']>=cfg['origin_ns'] and r['finish_ns']<=end];lat=[(r['finish_ns']-r['nominal_release_ns'])/1e6 for r in complete]
 phases={k:stats([r[k] for r in during]) if during else None for k in ['I_PREFIX_ms','I_ACTION_ms','service_ms']}
 return {'requests':len(rows),'completed':len(complete),'unfinished':unfinished,'p99_completed_ms':float(np.percentile(lat,99)) if lat else None,'p99_all_ms':float(np.percentile(lat,99)) if lat and not unfinished else None,'p99_note':'Completed requests only when unfinished>0; not stationary latency under overload.','slo_rate':sum(r['finish_ns']<=r['deadline_ns'] for r in complete)/len(rows),'late_completed':sum(r['finish_ns']>r['deadline_ns'] for r in complete),'phase_samples_within_window':len(during),'phases':phases,'prefix_slowdown_median':phases['I_PREFIX_ms']['p50']/cal['I_PREFIX_ms']['p50'] if during else None,'denoise_slowdown_median':phases['I_ACTION_ms']['p50']/cal['I_ACTION_ms']['p50'] if during else None,'clients':cfg['clients'],'pattern':cfg['pattern'],'period_s':cfg['period_s'],'deadline_s':cfg['deadline_s'],'duration_s':cfg['duration_s'],'drain_cap_s':cfg['drain_s'],'physical_gpu':GPU,'inference_batch':1,'backend':'CUDA Graph','phase_measurement':'Only requests starting and completing inside active training window; drain excluded.'}
def server_window(s,cfg,out,cal,tr=None,cost=None):
 plan=schedule(cfg);rows=[];segments=[];idx=0;origin=cfg['origin_ns'];end=origin+round(cfg['duration_s']*1e9);cap=end+round(cfg['drain_s']*1e9);rejects=0
 if tr:
  from replay import edf_admit
 while (idx<len(plan) or (tr is not None and stamp()<end)) and stamp()<cap:
  now=stamp();req=plan[idx] if idx<len(plan) else None
  if req is not None and req['nominal_release_ns']<=now:
   _,timing=s.infer(req['request_id']);row={**req,**timing};rows.append(row);idx+=1
   with (out/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  elif tr and origin<=now<end:
   known=[((x['nominal_release_ns']-origin)/1e9,(x['deadline_ns']-origin)/1e9) for x in plan[idx:] if x['nominal_release_ns']<=now+round(2*cfg['period_s']*1e9)]
   if edf_admit((now-origin)/1e9,cost[tr.next_kind],cal['service_ms']['mean']/1000,.010,known):segments.append(tr.segment())
   else:rejects+=1;time.sleep(.0005)
  elif req is not None:time.sleep(min(.001,max(0,(req['nominal_release_ns']-now)/1e9)))
  else:break
 for req in plan[idx:]:
  row={**req,'start_ns':None,'finish_ns':None,'status':'unfinished_at_drain_cap'};rows.append(row)
  with (out/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 summary=summarize(rows,cfg,cal)
 if tr:
  updates=sum(x['optimizer_completed'] for x in segments if x['cpu_submit_ns']>=origin and x['gpu_done_ns']<=end);summary.update(complete_updates=updates,steps_per_s=updates/cfg['duration_s'],admission_rejections=rejects);write_csv(out/'segments.csv',segments,fields=['iteration_id','microbatch_id','segment_kind','cpu_submit_ns','gpu_done_ns','optimizer_completed','wall_ms','cuda_ms'])
  while tr.micro or tr.next_kind!='T_FWD':tr.segment()
 save(out/'server_summary.json',summary)
def main():
 props=torch.cuda.get_device_properties(0);save(OUT/f'{args.role}_device.json',{'physical_gpu':GPU,'visible_sm_count':props.multi_processor_count,'mps':os.environ.get('PI_MPS'),'active_percentage':os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')})
 s=tr=None;cost=None
 if args.role!='train':
  s=GraphServing(1);save(OUT/'graph_correctness.json',s.verify_graph())
  for i in range(10):s.infer(i)
 if args.role!='server':
  tr=CacheTrain();tr.build_host();tr.cached=os.environ.get("CACHE_ENABLED")=="1"
  for _ in range(3):
   while not tr.segment()['optimizer_completed']:pass
  if args.role=='segmented':
   seg=[]
   for _ in range(5):
    while True:
     row=tr.segment();seg.append(row)
     if row['optimizer_completed']:break
   cost={k:float(np.percentile([x['wall_ms']/1000 for x in seg if x['segment_kind']==k],95)) for k in ['T_FWD','T_BWD','T_OPT']};save(OUT/'training_calibration.json',{'cost_p95_s':cost,'margin_s':.010,'lookahead_periods':2,'microbatch':4,'accumulation':2})
 save(OUT/f'{args.role}_affinity.json',pin_all_threads())
 save(OUT/f'{args.role}_ready.json',{'ready':True,'pid':os.getpid()});cal=None
 for index in range(100):
  cmd=wait_file(OUT/f'command_{index:03d}.json')
  if cmd['op']=='stop':break
  if cmd['op']=='calibrate':
   if s:
    rows=[s.infer(i)[1] for i in range(100)];cal={k:stats([r[k] for r in rows]) for k in ['I_PREFIX_ms','I_ACTION_ms','service_ms']};save(OUT/'isolated_calibration.json',cal);save(OUT/'isolated_calibration_raw.json',rows)
   save(OUT/f'{args.role}_done_{index:03d}.json',{'complete':True});continue
  cfg=cmd;out=OUT/cfg['name'];out.mkdir(exist_ok=True)
  if args.role=='train':
   origin=cfg['origin_ns'];end=origin+round(cfg['duration_s']*1e9);time.sleep(max(0,(origin-stamp())/1e9));rows=[]
   while stamp()<end:rows.append(tr.segment())
   updates=sum(x['optimizer_completed'] for x in rows if x['cpu_submit_ns']>=origin and x['gpu_done_ns']<=end);write_csv(out/'segments.csv',rows);save(out/'train_summary.json',{'complete_updates':updates,'steps_per_s':updates/cfg['duration_s'],'microbatch':4,'accumulation':2})
   while tr.micro or tr.next_kind!='T_FWD':tr.segment()
  else:server_window(s,cfg,out,cal,tr,cost)
  save(OUT/f'{args.role}_done_{index:03d}.json',{'complete':True})
 save(OUT/f'{args.role}_status.json',{'state':'complete'})
try:main()
except Exception:save(OUT/f'{args.role}_status.json',{'state':'failed','traceback':traceback.format_exc()});raise
