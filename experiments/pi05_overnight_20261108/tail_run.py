import os,sys,time,json,subprocess,threading,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;OUT=R/'tail';OUT.mkdir(exist_ok=True);B=Path('/home/hengyi/pi05_a1_baseline_20261108');PY='/home/hengyi/vla_followup_20261020/venv/bin/python';gpu=int(os.environ['VLA_GPU']);samples=[];stop=threading.Event()
def save(p,d):
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def observe():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);cards={int(x.split(',')[0]):[int(v.strip()) for v in x.split(',')[1:]] for x in s.splitlines()};p=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 uu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True);mapping={int(x.split(',')[0]):x.split(',')[1].strip() for x in uu.splitlines()};busy={x.split(',')[0].strip() for x in p.splitlines() if ',' in x}
 return {'time_ns':time.monotonic_ns(),'utc_epoch':time.time(),'cards':cards,'compute_apps':p,'other_cards_idle':all(cards[i][0]<1024 and cards[i][1]<5 and mapping[i] not in busy for i in [2,3])}
def monitor():
 while not stop.is_set():
  try:
   row=observe();samples.append(row)
   with (OUT/'gpu_monitor.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  except Exception:
   samples.append({'time_ns':time.monotonic_ns(),'other_cards_idle':False,'error':traceback.format_exc()})
  stop.wait(1)
def wait(paths,children,timeout):
 end=time.monotonic()+timeout
 while not all(p.exists() for p in paths):
  if any(c.poll() is not None for c in children):raise RuntimeError('tail worker exited early')
  if time.monotonic()>end:raise TimeoutError(str(paths))
  time.sleep(.2)
def main():
 first=observe();save(OUT/'before.json',first);assert first['other_cards_idle'],'GPU2/3 not idle at launch';thread=threading.Thread(target=monitor,daemon=True);thread.start();pipe=OUT/'pipe';logs=OUT/'mps_logs';pipe.mkdir(exist_ok=True);logs.mkdir(exist_ok=True);uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();env=dict(os.environ,PI_MPS='1',CUDA_VISIBLE_DEVICES=uuid,CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(logs));children=[];daemon=False
 try:
  save(OUT/'status.json',{'state':'running','gpu':gpu});s=subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,capture_output=True,text=True);assert s.returncode==0;daemon=True
  for role in ['server','train']:
   argv=[PY,'-u',str(B/'server.py'),'--role','server','--out',str(OUT)] if role=='server' else [PY,'-u',str(R/'tail_train.py'),'--out',str(OUT)];e=dict(env,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100' if role=='server' else '30')
   with (OUT/f'{role}.log').open('w') as f:p=subprocess.Popen(argv,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append(p)
  wait([OUT/'server_ready.json',OUT/'train_ready.json'],children,7200);save(OUT/'command_000.json',{'op':'calibrate'});wait([OUT/'server_done_000.json',OUT/'train_done_000.json'],children,300)
  idx=0
  for cap in [-1,3,2,1]:
   for w in range(5):
    idx+=1;name=f'cap{cap}_window{w+1}';cfg={'op':'run','name':name,'rate_cap':cap,'clients':1,'pattern':'sync','period_s':.25,'deadline_s':.25,'duration_s':60,'drain_s':5,'origin_ns':time.monotonic_ns()+2_000_000_000};save(OUT/f'command_{idx:03d}.json',cfg);wait([OUT/f'server_done_{idx:03d}.json',OUT/f'train_done_{idx:03d}.json'],children,180);data=json.loads((OUT/name/'server_summary.json').read_text());data['training']=json.loads((OUT/name/'train_summary.json').read_text());obs=[x for x in samples if cfg['origin_ns']<=x['time_ns']<=cfg['origin_ns']+65_000_000_000];data['other_cards_idle']=bool(obs) and all(x['other_cards_idle'] for x in obs);data['monitor_samples']=len(obs);save(OUT/name/'summary.json',data)
  save(OUT/f'command_{idx+1:03d}.json',{'op':'stop'})
  for c in children:assert c.wait(timeout=60)==0
  contaminated=[str(p.parent.name) for p in OUT.glob('cap*/summary.json') if not json.loads(p.read_text())['other_cards_idle']];save(OUT/'status.json',{'state':'complete','gpu':gpu,'contaminated_windows':contaminated,'isolation_valid':not contaminated})
 finally:
  stop.set();thread.join(timeout=5)
  for c in children:
   if c.poll() is None:
    c.terminate()
    try:c.wait(timeout=15)
    except subprocess.TimeoutExpired:c.kill();c.wait()
  if daemon:subprocess.run(['nvidia-cuda-mps-control'],env=env,input='quit\n',text=True,capture_output=True,timeout=20)
if __name__=='__main__':
 try:main()
 except Exception:save(OUT/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
