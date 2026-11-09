import os,sys,time,json,pathlib,subprocess,traceback
R=pathlib.Path('/home/hengyi/pi05_optimized_20261107');PY='/home/hengyi/vla_followup_20261020/venv/bin/python';gpu=int(sys.argv[1]);vision=sys.argv[2]=='vision';tier='action_vision_lora' if vision else 'action';OUT=R/'matrix'/tier;OUT.mkdir(parents=True,exist_ok=True)
def save(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def wait_checks(paths,children,timeout=900):
 deadline=time.monotonic()+timeout
 while not all(p.exists() for p in paths):
  if any(c.poll() is not None for c in children):raise RuntimeError('worker exited before expected acknowledgment')
  if time.monotonic()>deadline:raise TimeoutError(str(paths))
  time.sleep(.5)
prerequisites=[R/'training_mb4/status.json'] if gpu==2 else [R/'inference/status.json',R/'training_mb4/status.json']
if gpu==2:
 while not (R/'inference/batch1_correctness.json').exists():time.sleep(3)
 assert all(x['raw_action_max_abs']<=1e-6 for x in json.load(open(R/'inference/batch1_correctness.json')))
for need in prerequisites:
 while True:
  if need.exists():
   d=json.loads(need.read_text())
   if d.get('state')=='complete':break
   if d.get('state')=='failed':save(OUT/'status.json',{'state':'blocked_prerequisite','path':str(need)});sys.exit(1)
  time.sleep(3)
workloads=[(1,'sync')]+[(n,p) for n in [2,4,8] for p in ['sync','staggered']]
for mode in ['pure','free','mps30','mps50','segmented']:
 dest=OUT/mode;dest.mkdir(exist_ok=True);children=[];env={**os.environ,'VLA_GPU':str(gpu)};daemon=False
 try:
  save(OUT/'status.json',{'state':'loading','mode':mode,'gpu':gpu})
  if mode.startswith('mps'):
   uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();pipe=dest/'pipe';logs=dest/'mps_logs';pipe.mkdir(exist_ok=True);logs.mkdir(exist_ok=True);env.update(CUDA_VISIBLE_DEVICES=uuid,CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(logs),PI_MPS='1')
   p=subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,text=True,capture_output=True);save(dest/'mps_start.json',{'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
   if p.returncode:raise RuntimeError('MPS unavailable')
   daemon=True
  roles=['segmented'] if mode=='segmented' else (['server'] if mode=='pure' else ['server','train'])
  for role in roles:
   e=env.copy()
   if daemon:e['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']=mode[3:] if role=='train' else '100'
   f=open(dest/f'{role}.log','a');c=subprocess.Popen([PY,'-u',str(R/'matrix_worker.py'),'--role',role,'--out',str(dest)]+(['--vision'] if vision else []),env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append(c)
  wait_checks([dest/f'{role}_ready.json' for role in roles],children)
  if daemon:
   d=json.load(open(dest/'train_device.json'));assert d['visible_sm_count']<84,d
  save(dest/'command_000.json',{'op':'calibrate'});wait_checks([dest/f'{role}_done_000.json' for role in roles],children)
  for idx,(clients,pattern) in enumerate(workloads,1):
   name=f'n{clients}_{pattern}';cfg={'op':'run','name':name,'clients':clients,'pattern':pattern,'period_s':.250,'deadline_s':.250,'duration_s':30,'drain_s':5,'origin_ns':time.monotonic_ns()+2_000_000_000};save(OUT/'status.json',{'state':'running','mode':mode,'workload':name,'gpu':gpu});save(dest/f'command_{idx:03d}.json',cfg);wait_checks([dest/f'{role}_done_{idx:03d}.json' for role in roles],children,timeout=180)
   s=json.load(open(dest/name/'server_summary.json'))
   if mode=='pure':s.update(complete_updates=0,steps_per_s=0.)
   elif mode!='segmented':s.update(json.load(open(dest/name/'train_summary.json')))
   s.update(mode=mode,tier=tier);save(dest/name/'summary.json',s)
  save(dest/'command_008.json',{'op':'stop'})
  for c in children:
   if c.wait(timeout=60):raise RuntimeError('worker failed at shutdown')
  save(dest/'status.json',{'state':'complete','conditions':7})
 except Exception:save(dest/'failure.json',{'traceback':traceback.format_exc()})
 finally:
  for c in children:
   if c.poll() is None:
    c.terminate()
    try:c.wait(timeout=15)
    except subprocess.TimeoutExpired:c.kill();c.wait()
  if daemon:subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',env=env,text=True,capture_output=True,timeout=20)
save(OUT/'status.json',{'state':'finished','gpu':gpu,'note':'inspect individual failures'})
