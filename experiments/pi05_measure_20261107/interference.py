import os,sys,json,time,traceback,argparse
from pathlib import Path
ROOT=Path('/home/hengyi/pi05_measure_20261107')
p=argparse.ArgumentParser();p.add_argument('--role',choices=['server','train','segmented'],required=True);p.add_argument('--vision',action='store_true');p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
if os.environ.get('PI_MPS')=='1':
 import subprocess,importlib.util
 os.environ['PI_MPS_UUID']=subprocess.check_output(['nvidia-smi','-i',os.environ['VLA_GPU'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 os.environ['CUDA_VISIBLE_DEVICES']=os.environ['PI_MPS_UUID']
 spec=importlib.util.spec_from_file_location('core',ROOT/'core_mps.py');module=importlib.util.module_from_spec(spec);sys.modules['core']=module;spec.loader.exec_module(module)
from training import *
os.sched_setaffinity(0,set(range(GPU*8+(4 if args.role=='train' else 0),GPU*8+(8 if args.role=='train' else 4))))
torch.set_num_threads(4)
def wait_start():
 while not (out/'start.json').exists():time.sleep(.05)
 return json.loads((out/'start.json').read_text())
def summarize_requests(rows):
 lat=[(r['finish_ns']-r['nominal_release_ns'])/1e6 for r in rows];return {'requests':len(rows),'p99_ms':float(np.percentile(lat,99)),'p50_ms':float(np.percentile(lat,50)),'slo_rate':sum(r['finish_ns']<=r['deadline_ns'] for r in rows)/len(rows),'max_ms':max(lat)}
def main():
 props=torch.cuda.get_device_properties(0);save(out/f'{args.role}_device.json',{'name':props.name,'visible_sm_count':props.multi_processor_count,'physical_gpu':GPU,'mps':os.environ.get('PI_MPS')=='1','requested_percentage':os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')})
 if args.role=='server':
  s=Serving();s.model.num_steps=10
  for i in range(10):s.infer(i)
  save(out/'server_ready.json',{'ready':True});cfg=wait_start();start=cfg['start_ns'];rows=[]
  for i in range(240):
   release=start+i*500_000_000;time.sleep(max(0,(release-stamp())/1e9));_,r=s.infer(i)
   rows.append({**r,'request_id':i,'client_id':i%2,'nominal_release_ns':release,'deadline_ns':release+1_000_000_000})
   with (out/'requests.jsonl').open('a') as f:f.write(json.dumps(rows[-1])+'\n')
  save(out/'server_summary.json',summarize_requests(rows));save(out/'server_done.json',{'complete':True})
 elif args.role=='train':
  tr=BenchTrain(args.vision)
  for _ in range(2):
   while True:
    r=tr.segment()
    if r['optimizer_completed']:break
  save(out/'train_ready.json',{'ready':True});cfg=wait_start();start=cfg['start_ns'];end=start+120_000_000_000;time.sleep(max(0,(start-stamp())/1e9));updates=0;rows=[]
  while stamp()<end:
   r=tr.segment();rows.append(r)
   if r['cpu_submit_ns']>=start and r['gpu_done_ns']<=end:updates+=r['optimizer_completed']
  write_csv(out/'segments.csv',rows);save(out/'train_summary.json',{'complete_updates':updates,'steps_per_s':updates/120,'duration_s':120,'last_partial_micro':tr.micro,'last_partial_stage':tr.next_kind});save(out/'train_done.json',{'complete':True})
 else:
  from replay import run_window
  s=Serving();s.model.num_steps=10;tr=BenchTrain(args.vision)
  for i in range(10):s.infer(i)
  for _ in range(2):
   while not tr.segment()['optimizer_completed']:pass
  rows=[]
  while tr.iteration<5:rows.append(tr.segment())
  infer=[s.infer(i)[1]['service_ms']/1000 for i in range(20)]
  cal={'costs_s':{k:float(np.percentile([r['wall_ms']/1000 for r in rows if r['segment_kind']==k],95)) for k in ['T_FWD','T_BWD','T_OPT']},'inference_mean_s':float(np.mean(infer)),'margin_s':max(.01,float(np.percentile(infer,95)-np.mean(infer))),'training_samples_per_s':3*8/((rows[-1]['gpu_done_ns']-rows[0]['cpu_submit_ns'])/1e9)}
  save(out/'calibration.json',cal)
  summary,req,seg,mem=run_window(s,tr,'S4',1.,'staggered',120,cal,'pi05_10step_'+tr.tier,str(GPU))
  summary.update(steps_per_s=summary['completed_updates']/120,slo_rate=1-summary['deadline_miss_rate']);save(out/'summary.json',summary);write_csv(out/'requests.csv',req);write_csv(out/'segments.csv',seg);write_csv(out/'memory.csv',mem)
 save(out/f'{args.role}_status.json',{'state':'complete'})
try:main()
except Exception:save(out/f'{args.role}_status.json',{'state':'failed','traceback':traceback.format_exc()});raise
