import os,sys,time,json,subprocess,traceback,argparse
from pathlib import Path
R=Path(__file__).resolve().parent;B=Path('/home/hengyi/pi05_a1_baseline_20261108');PY='/home/hengyi/vla_followup_20261020/venv/bin/python';GPU=int(os.environ['VLA_GPU']);assert GPU in [0,1]
ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args()
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def wait_checks(paths,children,timeout):
 end=time.monotonic()+timeout
 while not all(p.exists() for p in paths):
  if any(c.poll() is not None for c in children):raise RuntimeError('worker exited before expected acknowledgment')
  if time.monotonic()>end:raise TimeoutError(str(paths))
  time.sleep(.2)
def run(mode,rate,smoke=False):
 dest=R/('preflight' if smoke else 'runs')/f'{mode}_r{rate}';dest.mkdir(parents=True,exist_ok=False);children=[];daemon=False
 uuid=subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();pipe=dest/'pipe';logs=dest/'mps_logs';pipe.mkdir();logs.mkdir();env=dict(os.environ,PI_MPS='1',CUDA_VISIBLE_DEVICES=uuid,CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(logs));status_file=R/('preflight_status.json' if smoke else 'status.json')
 try:
  save(status_file,{'state':'running','mode':mode,'rate':rate,'gpu':GPU,'smoke':smoke});p=subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,capture_output=True,text=True);save(dest/'mps_start.json',{'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr});assert p.returncode==0;daemon=True
  for role in ['server','train']:
   e=dict(env,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='30' if role=='train' else '100');argv=[PY,'-u',str(B/'server.py'),'--role','server','--out',str(dest)] if role=='server' else [PY,'-u',str(R/'stream_train.py'),'--mode',mode,'--rate',str(rate),'--out',str(dest)]+(['--smoke'] if smoke else [])
   with (dest/f'{role}.log').open('w') as f:child=subprocess.Popen(argv,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append(child)
  save(dest/'pids.json',{'server':children[0].pid,'train':children[1].pid});wait_checks([dest/'server_ready.json',dest/'train_ready.json'],children,7200)
  save(dest/'command_000.json',{'op':'calibrate'});wait_checks([dest/'server_done_000.json',dest/'train_done_000.json'],children,300)
  warmup=6 if smoke else int(512/rate)+30;duration=6 if smoke else 300;first=time.monotonic_ns()+2_000_000_000
  configs=[{'name':'warmup','duration_s':warmup,'origin_ns':first}]+[{'name':f'window{i+1}','duration_s':duration,'origin_ns':first+round((warmup+i*duration)*1e9)} for i in range(4)]
  # Predeclared contiguous deadlines: no pause between the four measured windows.
  for i,item in enumerate(configs,1):save(dest/f'command_{i:03d}.json',dict(op='run',clients=1,pattern='sync',period_s=.25,deadline_s=.25,drain_s=5,**item))
  for i,item in enumerate(configs,1):
   wait_checks([dest/f'server_done_{i:03d}.json',dest/f'train_done_{i:03d}.json'],children,item['duration_s']+180);a=json.loads((dest/item['name']/'server_summary.json').read_text());b=json.loads((dest/item['name']/'train_summary.json').read_text());save(dest/item['name']/'summary.json',dict(a,training=b));assert b['hit_rates']['replay']==1.0,b
   if mode=='A2':assert b['hit_rates']['new']==1.0,b
   if not smoke:
    import report;report.render()
  save(dest/'command_006.json',{'op':'stop'})
  for c in children:
   if c.wait(timeout=60):raise RuntimeError('worker shutdown failed')
  save(dest/'complete.json',{'state':'complete','rate':rate,'mode':mode,'gpu':GPU,'smoke':smoke})
 finally:
  for c in children:
   if c.poll() is None:
    c.terminate()
    try:c.wait(timeout=15)
    except subprocess.TimeoutExpired:c.kill();c.wait()
  if daemon:subprocess.run(['nvidia-cuda-mps-control'],env=env,input='quit\n',text=True,capture_output=True,timeout=20)
def main():
 if args.preflight_only:
  for mode in ['A1','A2']:run(mode,16,True)
  save(R/'preflight_status.json',{'state':'complete','gpu':GPU});return
 assert json.loads((R/'preflight_status.json').read_text())['state']=='complete'
 for rate in [4,16]:
  for mode in ['A1','A2']:run(mode,rate)
 save(R/'status.json',{'state':'complete','gpu':GPU})
 import report;report.render()
if __name__=='__main__':
 try:main()
 except Exception:save(R/('preflight_status.json' if args.preflight_only else 'status.json'),{'state':'failed','traceback':traceback.format_exc()});raise
