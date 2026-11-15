"""Real-cache concurrent experiments with common arrival schedule and absolute deadlines."""
from pathlib import Path
import os,json,time,subprocess,pickle,math,argparse,traceback,statistics,csv,signal,hashlib,shutil,threading,ctypes
R=Path(__file__).resolve().parent;W=R/'worker';PY='/home/hengyi/Workspace/01_VLA-Serving/Environment-and-Earlier-Runs/vla_followup_20261020/venv/bin/python'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def observe():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,power.limit,power.draw','--format=csv,noheader,nounits'],text=True);cards={}
 for line in s.splitlines():
  i,u,m,v,pl,pd=[x.strip() for x in line.split(',')];cards[int(i)]=dict(uuid=u,memory_mib=int(m),util=int(v),power_limit_w=float(pl),power_draw_w=float(pd) if pd not in ['[N/A]','N/A'] else None)
 return dict(time=time.time(),ns=time.monotonic_ns(),cards=cards,load=list(os.getloadavg()),cpu_stat=Path('/proc/stat').read_text(),meminfo=Path('/proc/meminfo').read_text(),cpu_pressure=Path('/proc/pressure/cpu').read_text(),memory_pressure=Path('/proc/pressure/memory').read_text())
def quantile(values,q=.99):
 return sorted(values)[max(0,math.ceil(q*len(values))-1)] if values else None
class Experiment:
 def __init__(self,out,layout,policy,mode,smoke=False,gpu=1):
  self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False);self.layout=layout;self.policy=policy;self.mps_policy='mps100' if policy=='feedback' else policy;self.mode=mode;self.smoke=smoke;self.children=[];self.daemons=[];self.mps_dirs={};self.index=0;self.mon=[];self.last=0;self.pairs=[];self.before=observe();save(self.out/'before.json',self.before)
  placements=[(gpu,gpu,0,2)] if layout=='co_one' else ([(gpu,gpu,0,1)] if layout=='single' else ([(0,0,0,2),(1,1,1,2)] if layout=='co' else [(0,1,0,1)]))
  # Admission uses live compute processes, never rolling utilization samples.
  # The other owned card must also remain idle for single-card latency measurements.
  checks=[];deadline=time.monotonic()+300
  while True:
   active={g:subprocess.check_output(['nvidia-smi',f'--id={g}','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip() for g in [0,1]}
   checks.append(dict(time=time.time(),processes=active))
   if not any(active.values()):
    time.sleep(30)
    active={g:subprocess.check_output(['nvidia-smi',f'--id={g}','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip() for g in [0,1]}
    checks.append(dict(time=time.time(),after_settle=True,processes=active))
    if not any(active.values()):break
   if time.monotonic()>deadline:save(self.out/'idle_admission_checks.json',checks);raise TimeoutError('GPU0/1 have compute processes; no workloads interrupted')
   time.sleep(5)
  save(self.out/'idle_admission_checks.json',checks);self.before=observe();save(self.out/'before.json',self.before)
  assert all(self.before['cards'][g]['power_limit_w']==225. for g in [0,1]),'Power limit changed; not altering settings'
  self.placements=placements
 def sample(self):
  if time.monotonic()-self.last<2:return
  self.last=time.monotonic();x=observe();self.mon.append(x)
  with (self.out/'host_gpu_monitor.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 def wait(self,paths,timeout):
  end=time.monotonic()+timeout
  while not all(p.exists() for p in paths):
   self.sample()
   for p in self.children:
    if p.poll() is not None:raise RuntimeError(f'worker{p.pid} exited{p.returncode}; {self.out}')
   if time.monotonic()>end:raise TimeoutError(str([str(p) for p in paths if not p.exists()]))
   time.sleep(.05)
 def env(self,gpu,train=False):
  env=dict(os.environ,VLA_GPU=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4');env.pop('PI_MPS',None);env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None)
  if self.mps_policy.startswith('mps'):
   env.update(PI_MPS='1',CUDA_VISIBLE_DEVICES=self.before['cards'][gpu]['uuid'],CUDA_MPS_PIPE_DIRECTORY=str(self.mps_dirs[gpu]/'pipe'),CUDA_MPS_LOG_DIRECTORY=str(self.mps_dirs[gpu]/'log'),CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(int(self.mps_policy[3:]) if train and self.layout!='split' else 100))
  if self.layout=='split':env.update(TRAIN_CPU_LIST=','.join(map(str,range(8,16))),SERVER_CPU_LIST=','.join(map(str,range(0,8))))
  else:env.update(TRAIN_CPU_LIST=','.join(map(str,range(gpu*8+4,gpu*8+8))),SERVER_CPU_LIST=','.join(map(str,range(gpu*8,gpu*8+4))))
  env['SERVER_TORCH_THREADS']='1';env['SLO_FEEDBACK']='1' if self.policy=='feedback' else '0';env['ASYNC_TRAIN']='1' if self.policy=='streams_opt' else '0';env['RECENT_CAPACITY']='256' if self.layout in ['co','co_one'] else '512';return env
 def spawn(self,cmd,env,log):
  with Path(log).open('w') as f:p=subprocess.Popen([PY,'-u',*cmd],env=env,cwd=W,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  self.children.append(p);save(self.out/'worker_pids.json',[x.pid for x in self.children]);return p
 def start(self):
  if self.mps_policy.startswith('mps'):
   for gpu in sorted({x for p in self.placements for x in p[:2]}):
    base=Path('/tmp')/f'ev{os.getuid()}_{hashlib.sha256(str(self.out).encode()).hexdigest()[:12]}_{gpu}';base.mkdir(mode=0o700);self.mps_dirs[gpu]=base;(base/'pipe').mkdir(mode=0o700);(base/'log').mkdir(mode=0o700);save(self.out/f'mps_paths{gpu}.json',dict(pipe=str(base/'pipe'),log=str(base/'log')));env=self.env(gpu);p=subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,capture_output=True,text=True,timeout=30);save(self.out/f'mps_start{gpu}.json',dict(returncode=p.returncode,stderr=p.stderr,stdout=p.stdout));assert p.returncode==0,(p.returncode,p.stdout,p.stderr);self.daemons.append(env)
  for i,(sg,tg,offset,stride) in enumerate(self.placements):
   out=self.out/f'pair{i}';out.mkdir();(out/'ipc').mkdir();pair=dict(out=out,sg=sg,tg=tg,offset=offset,stride=stride);self.pairs.append(pair)
   if self.policy=='streams_opt' and sg==tg:
    self.spawn([str(W/'single_stream.py'),'--out',str(out)]+(['--smoke'] if self.smoke else []),self.env(tg,True),out/'single_process.log')
   elif self.mode!='baseline':self.spawn([str(W/'interleaved_train.py'),'--mode','A1','--out',str(out)]+(['--smoke'] if self.smoke else []),self.env(tg,True),out/'train.log')

  if self.mode=='baseline':
   for p in self.pairs:os.link(R/'inputs.pt',p['out']/'inputs.pt')
  else:self.wait([p['out']/'train_ready.json' for p in self.pairs],1200)
  for p in self.pairs:
   if not (self.policy=='streams_opt' and p['sg']==p['tg']):self.spawn([str(W/'interleaved_server.py'),'--out',str(p['out']),'--mode','interleaved','--inputs',str(p['out']/'inputs.pt')],self.env(p['sg']),p['out']/'server.log')
  self.wait([p['out']/'server_ready.json' for p in self.pairs],600)
  save(self.out/'affinities.json',{str(p.pid):{t.name:sorted(os.sched_getaffinity(int(t.name))) for t in Path(f'/proc/{p.pid}/task').iterdir()} for p in self.children});self.command({'op':'calibrate'},180);self.calibrations=[read(p['out']/'calibration.json') for p in self.pairs];save(self.out/'calibrations.json',self.calibrations)
 def command(self,cfg,timeout):
  index=self.index
  for p in self.pairs:
   local=dict(cfg,source_offset=p['offset'],source_stride=p['stride'])
   if cfg['op']=='run':local['rate']=cfg['global_rate']/p['stride']
   if cfg['op']=='activate':local['drain_to']=p.get('next_id',0)
   save(p['out']/f'command_{index:03d}.json',local)
  self.wait([p['out']/f'{role}_done_{index:03d}.json' for p in self.pairs for role in (['server'] if self.mode=='baseline' else ['train','server'])],timeout)
  for p in self.pairs:p['next_id']=read(p['out']/f'server_done_{index:03d}.json').get('next_id',p.get('next_id',0))
  self.index+=1
 def window(self,name,duration,rate,round_number):
  cfg=dict(op='run',mode=self.mode,variant=self.mode,name=name,global_rate=rate,clients=4 if self.layout=='single' else 8,duration_s=duration,round=round_number,rate_cap=0,origin_ns=time.monotonic_ns()+2_000_000_000)
  save(self.out/f'{name}_schedule.json',cfg)
  capture=None
  if name=='measurement' and os.environ.get('TRACE_CAPTURE')=='1':
   def mark_capture():
    lib=ctypes.CDLL(os.environ['NVTX_LIBRARY']);lib.trace_begin.argtypes=[ctypes.c_char_p]
    time.sleep(max(0,(cfg['origin_ns']-time.monotonic_ns())/1e9));begin=time.monotonic_ns();lib.trace_begin(b'evoserve_trace_5s');time.sleep(5);lib.trace_end();end=time.monotonic_ns();save(self.out/'trace_range.json',dict(start_monotonic_ns=begin,end_monotonic_ns=end,duration_s=(end-begin)/1e9,requested_s=5))
   capture=threading.Thread(target=mark_capture);capture.start()
  self.command(cfg,duration+180)
  if capture:capture.join()
  return cfg
 def execute(self,target,duration,profile=False):
  self.start();reference=R/'reference_calibration.json'
  if self.smoke:service_ms=self.calibrations[0]['service_ms']['mean']
  elif reference.exists():service_ms=read(reference)['service_ms']
  else:
   service_ms=self.calibrations[0]['service_ms']['mean'];save(reference,dict(service_ms=service_ms,source=str(self.out/'calibrations.json'),definition='mean single request service time, inference alone while trainer idle'))
  rate=target/(service_ms/1000);save(self.out/'config.json',dict(layout=self.layout,policy=self.policy,mode=self.mode,target=target,global_rate=rate,reference_service_ms=service_ms,seconds=duration,smoke=self.smoke,deadline_ms=250,global_new_pool_capacity=512,clients=4 if self.layout=='single' else 8))
  self.command(dict(op='activate',mode=self.mode,variant=self.mode),120)
  warm=5 if self.smoke else (30 if self.mode=='baseline' else 512/rate+5);self.window('warmup',warm,rate,0);cfg=self.window('measurement',duration,rate,1)
  requests=[];training=[];slo=total=done=0
  for p in self.pairs:
   o=p['out']/'measurement';server=read(o/'server_summary.json');tr=read(o/'train_summary.json') if self.mode!='baseline' else dict(complete_updates=0,steps_per_s=0); slo+=server['slo_successes'];total+=server['requests'];done+=server['completed'];training.append(tr)
   if tr['complete_updates'] and self.mode=='A2':assert tr['hit_rates']['new']==1 and tr['hit_rates']['replay']==1
   a=[json.loads(x) for x in (o/'requests.jsonl').read_text().splitlines()];requests+=a
  lat=[(x['finish_ns']-x['nominal_release_ns'])/1e6 for x in requests]+[float('inf')]*(total-done);p99=quantile(lat);start=cfg['origin_ns'];end=start+round(duration*1e9);mon=[x for x in self.mon if start<=x['ns']<=end]
  util=statistics.mean([x['cards'][1]['util'] for x in mon]) if mon and self.layout=='split' else None
  result=dict(state='complete',layout=self.layout,policy=self.policy,mode=self.mode,target=target,requests=total,completed=done,unfinished=total-done,slo_rate=slo/total,p99_ms=p99 if math.isfinite(p99) else None,p99_status='finite' if math.isfinite(p99) else 'infinite_with_unfinished_requests',total_training_steps_per_s=sum(x['steps_per_s'] for x in training),training_summaries=training,dedicated_training_gpu_util_mean=util,feasible=slo/total>=.99,duration_s=duration,global_rate=rate,reference_service_ms=service_ms,smoke=self.smoke,raw_pairs=[str(p['out']) for p in self.pairs],timestamp=time.time())
  result['power_limits_before_w']={str(g):x['power_limit_w'] for g,x in self.before['cards'].items()};result['power_limits_during_w']={str(g):sorted({x['cards'][g]['power_limit_w'] for x in mon}) for g in range(4)};result['external_gpu_monitor']='host_gpu_monitor.jsonl includes GPU2/3, power, hostCPU pressure/load';save(self.out/'result.json',result)
  # Per60-second windows, preserving the same nominal request denominator.
  windowrows=[]
  for i in range(math.ceil(duration/60)):
   lo=start+round(i*60e9);hi=min(end,lo+60_000_000_000);ns=(hi-lo)/1e9;planned=sum(1 for n in range(math.ceil(duration*rate-1e-10)) if lo<=start+round(n/rate*1e9)<hi);rs=[x for x in requests if lo<=x['nominal_release_ns']<hi];ls=[(x['finish_ns']-x['nominal_release_ns'])/1e6 for x in rs]+[float('inf')]*(planned-len(rs));pp=quantile(ls);updates=0
   for p in self.pairs:
    path=p['out']/'measurement/steps.jsonl'
    if path.exists():updates+=sum(start<=x['start_ns'] and lo<=x['finish_ns']<hi for x in map(json.loads,path.read_text().splitlines()))
   windowrows.append(dict(window=i+1,seconds=ns,requests=planned,completed=len(rs),slo_rate=sum(x['finish_ns']<=x['deadline_ns'] for x in rs)/planned,p99_ms=pp if pp is not None and math.isfinite(pp) else None,training_steps_per_s=updates/ns))
  with (self.out/'windows.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(windowrows[0]));w.writeheader();w.writerows(windowrows)
  if profile:
   try:self.command(dict(op='profile',name='profile',mode=self.mode),180)
   except Exception:save(self.out/'profile_failure.json',dict(traceback=traceback.format_exc()))
  return result
 def close(self):
  if self.pairs:
   for p in self.pairs:save(p['out']/f'command_{self.index:03d}.json',{'op':'stop'})
  for p in self.children:
   if p.poll() is None:
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:p.terminate()
   if p.poll() is None:
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  for env in self.daemons:
   try:subprocess.run(['nvidia-cuda-mps-control'],env=env,input='quit\n',text=True,capture_output=True,timeout=20)
   except Exception:pass
  for gpu,base in self.mps_dirs.items():
   if (base/'log').exists():shutil.copytree(base/'log',self.out/f'mps_log{gpu}',dirs_exist_ok=True)
  for p in self.pairs:
   layout=p['out']/'ipc_layout.pkl'
   if layout.exists():Path(pickle.loads(layout.read_bytes())[0]).unlink(missing_ok=True)
   (p['out']/'inputs.pt').unlink(missing_ok=True)
def main():
 def stop(signum,frame):raise RuntimeError(f'terminated by signal{signum}')
 signal.signal(signal.SIGTERM,stop)
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--layout',choices=['single','co','co_one','split'],required=True);ap.add_argument('--policy',choices=['mps30','mps100','streams_opt','feedback','none'],required=True);ap.add_argument('--mode',choices=['A0','A2','baseline'],default='A2');ap.add_argument('--target',type=float,required=True);ap.add_argument('--seconds',type=float,required=True);ap.add_argument('--gpu',type=int,default=1);ap.add_argument('--smoke',action='store_true');ap.add_argument('--profile',action='store_true');a=ap.parse_args();e=None
 try:e=Experiment(a.out,a.layout,a.policy,a.mode,a.smoke,a.gpu);e.execute(a.target,a.seconds,a.profile)
 except Exception:
  out=Path(a.out);out.mkdir(parents=True,exist_ok=True);save(out/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
 finally:
  if e:e.close()
if __name__=='__main__':main()
