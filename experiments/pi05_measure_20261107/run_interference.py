import os,sys,json,time,subprocess,pathlib,traceback
R=pathlib.Path('/home/hengyi/pi05_measure_20261107');PY='/home/hengyi/vla_followup_20261020/venv/bin/python';gpu=int(sys.argv[1]);vision=sys.argv[2]=='vision';root=R/'interference'/('action_vision_lora' if vision else 'action');root.mkdir(parents=True,exist_ok=True)
def save(p,d):p.write_text(json.dumps(d,indent=2))
wait_file=R/('inference' if gpu==3 else 'training')/'status.json'
while True:
 if wait_file.exists() and json.loads(wait_file.read_text()).get('state') in ['complete','failed']:break
 time.sleep(5)
base={**os.environ,'VLA_GPU':str(gpu)}
def launch(role,out,env):
 log=open(out/f'{role}.log','a');return subprocess.Popen([PY,'-u',str(R/'interference.py'),'--role',role,'--out',str(out)]+(['--vision'] if vision else []),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
def wait_ready(children,out):
 deadline=time.monotonic()+600
 while not all((out/f'{role}_ready.json').exists() for role in ['server','train']):
  if any(c.poll() is not None for c in children):raise RuntimeError('worker exited before ready')
  if time.monotonic()>deadline:raise TimeoutError('ready timed out')
  time.sleep(1)
def run(mode):
 out=root/mode;out.mkdir(exist_ok=True);save(root/'status.json',{'state':'running','mode':mode,'gpu':gpu});children=[];daemon=False;env=base.copy()
 try:
  if mode.startswith('mps'):
   uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();pipe=out/'pipe';logs=out/'mps_logs';pipe.mkdir(exist_ok=True);logs.mkdir(exist_ok=True)
   env.update(CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(logs),CUDA_VISIBLE_DEVICES=uuid)
   attempt=subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,capture_output=True,text=True);save(out/'mps_start.json',{'returncode':attempt.returncode,'stdout':attempt.stdout,'stderr':attempt.stderr})
   if attempt.returncode:save(out/'summary.json',{'state':'skipped','reason':'MPS startup failed'});return
   daemon=True;env['PI_MPS']='1'
  if mode=='segmented':
   c=launch('segmented',out,env);children=[c];rc=c.wait(timeout=900)
   if rc:raise RuntimeError(f'segmented exit {rc}')
  else:
   trainenv=env.copy();serverenv=env.copy()
   if daemon:trainenv['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']=mode[3:];serverenv['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']='100'
   children=[launch('server',out,serverenv),launch('train',out,trainenv)];wait_ready(children,out)
   if daemon:
    probe=subprocess.run(['nvidia-cuda-mps-control'],input='get_server_list\n',env=env,text=True,capture_output=True);save(out/'mps_server_list.json',{'stdout':probe.stdout,'stderr':probe.stderr})
    dev=json.loads((out/'train_device.json').read_text())
    if dev['visible_sm_count']>=84:raise RuntimeError('MPS percentage not visibly effective; do not report as valid MPS')
   save(out/'start.json',{'start_ns':time.monotonic_ns()+2_000_000_000,'duration_s':120,'clients':2,'client_period_s':1.,'phase_offset_s':.5,'slo_s':1.,'requests':240})
   for c in children:
    if c.wait(timeout=480):raise RuntimeError('worker failed')
   result={**json.loads((out/'server_summary.json').read_text()),**json.loads((out/'train_summary.json').read_text()),'mode':mode,'gpu':gpu,'state':'complete'};save(out/'summary.json',result)
 except Exception:save(out/'failure.json',{'traceback':traceback.format_exc()})
 finally:
  for c in children:
   if c.poll() is None:
    c.terminate()
    try:c.wait(timeout=15)
    except subprocess.TimeoutExpired:c.kill();c.wait()
  if daemon:subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',env=env,capture_output=True,text=True,timeout=20)
for mode in os.environ.get('PI_MODES','free,segmented,mps30,mps50').split(','):run(mode)
save(root/'status.json',{'state':'finished','gpu':gpu,'note':'inspect each mode summary/failure; finished does not imply every mode succeeded'})
