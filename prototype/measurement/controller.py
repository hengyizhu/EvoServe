from pathlib import Path
import os,sys,json,time,subprocess,signal,fcntl,traceback,statistics
R=Path(__file__).resolve().parent;P=json.loads((R/'protocol.json').read_text());PY='/home/hengyi/Workspace/01_VLA-Serving/Environment-and-Earlier-Runs/vla_followup_20261020/venv/bin/python'
def read(p):
 try:return json.loads(Path(p).read_text())
 except (OSError,ValueError):return {}
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def report():subprocess.run([PY,str(R/'report.py')],timeout=60)
def status(state,**kw):save(R/'status.json',dict(state=state,time=time.time(),**kw));report()
def env(n=None,trace=False):
 e=dict(os.environ)
 for k in list(e):
  if k.startswith('CUDA_MPS_') or k in ['PI_MPS','EXPERT_COUNT','ASYNC_TRAIN','SLO_FEEDBACK','SINGLE_PROCESS']:e.pop(k,None)
 e.update(VLA_GPU='0',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 if n:e['EXPERT_COUNT']=str(n)
 if trace:e.update(TRACE_CAPTURE='1',NVTX_LIBRARY=P['trace']['nvtx_library'],NSYS_NVTX_PROFILER_REGISTER_ONLY='0')
 return e
def duty(out,x):
 from duty_stats import mean_duty
 return mean_duty(out,x)
def call(cmd,label,e,timeout=2700):
 with (R/'logs'/(label+'.log')).open('a') as f:p=subprocess.Popen(list(map(str,cmd)),cwd=R,env=e,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 save(R/'active_process.json',dict(pid=p.pid,cmd=list(map(str,cmd)),started=time.time()))
 try:rc=p.wait(timeout=timeout)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=90)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  raise TimeoutError(label)
 if rc:raise RuntimeError(f'{label} exit{rc}')
def case(name,n,load,mode='A2',trace=False,policy='feedback'):
 marker=R/'task_state'/(name+'.json')
 if read(marker).get('state') in ['complete','skipped']:return
 for stage,seconds in [('smoke',60),('formal',5 if trace else 600)]:
  stagefile=R/'task_state'/f'{name}_{stage}.json'
  if read(stagefile).get('state')=='complete':continue
  for attempt in [1,2]:
   out=R/'system'/('smokes' if stage=='smoke' else 'runs')/f'{name}_{stage}_{time.time_ns()}';cmd=[PY,'-u',R/'system/measure.py','--layout','single','--policy',policy,'--mode',mode,'--gpu',0,'--target',load,'--out',out,'--seconds',seconds]
   if stage=='smoke':cmd+=['--smoke']
   if not trace:cmd+=['--profile']
   prof=None
   if trace:
    prof=R/'traces'/f'{name}_{stage}_{attempt}_{time.time_ns()}'
    cmd=[P['trace']['nsys'],'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=nvtx','--nvtx-capture=evoserve_trace_5s','--capture-range-end=stop','--kill=none','--force-overwrite=true','--output',prof,*cmd]
   try:
    status('running',case=name,stage=stage,attempt=attempt,raw=str(out));call(cmd,name+'_'+stage+str(attempt),env(n if not trace else None,trace));x=read(out/'result.json');assert x.get('state')=='complete' and x['completed']>0
    kernel=read(out/'pair0/profile/longest_kernel.json') if not trace else {}
    record=dict(kernel_profile=kernel,state='complete',case=name,mode=mode,experts=n,load=load,raw=str(out),result=x,expert_steps_s=n*x['total_training_steps_per_s'],mean_duty=duty(out,x),trace=trace)
    if trace:
     rep=Path(str(prof)+'.nsys-rep');assert rep.exists();db=Path(str(prof)+'.sqlite');call([P['trace']['nsys'],'export','--type=sqlite','--output',db,rep],name+'_'+stage+'_export',env(),600)
     timeline=Path(str(prof)+'_kernels.csv');call([PY,R/'export_trace.py','--sqlite',db,'--run',out,'--out',timeline],name+'_'+stage+'_csv',env(),300);record.update(nsys_report=str(rep),sqlite=str(db),kernel_csv=str(timeline),trace_validation=read(timeline.with_suffix('.json')))
    save(stagefile,record);report();break
   except Exception:
    save(R/'failures'/f'{name}_{stage}_attempt{attempt}.json',dict(traceback=traceback.format_exc(),raw=str(out),time=time.time()));report()
    if attempt==2:save(marker,dict(state='skipped',reason=f'{stage} failed after one retry; continuing'));report();return
 save(marker,read(R/'task_state'/f'{name}_formal.json'));report()
def main():
 with (R/'controller.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(R/'controller_pid.json',dict(pid=os.getpid()))
  for u in [.75,.8]:
   for n in [1,2,4,8]:case(f'expert{n}_u{int(u*100)}',n,u)
  case('trace_A0_mps30_u70',1,.7,'A0',True,'mps30')
  skipped=[p.stem for p in (R/'task_state').glob('*.json') if read(p).get('state')=='skipped']
  status('measurements_finished',skipped=skipped)
  save(R/'measurements_complete.json',dict(time=time.time(),skipped=skipped))
  # Prototype readiness is explicit; never consume a GPU during measurement.
  while not (R/'prototype_ready.json').exists():
   status('waiting_prototype_cpu_checks',skipped=skipped);time.sleep(30)
  ready=read(R/'prototype_ready.json');assert ready['cpu_checks']=='passed'
  call(ready['command'],'prototype',dict(os.environ),timeout=7*24*3600)
  status('finished',skipped=skipped)
if __name__=='__main__':
 (R/'failures').mkdir(exist_ok=True)
 try:main()
 except Exception:status('failed',traceback=traceback.format_exc());raise
