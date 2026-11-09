"""Single-GPU interrupt-free insertion; MPS daemon belongs only to this suite."""
import os,sys,json,time,subprocess,traceback,signal,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;PY='/home/hengyi/vla_followup_20261020/venv/bin/python';GPU=int(os.environ['VLA_GPU']);assert GPU in [0,1]
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def run_worker(mode,phase):
 dest=R/phase/mode;dest.mkdir(parents=True,exist_ok=False)
 save(R/'status.json',{'state':'running','phase':phase,'mode':mode,'gpu':GPU})
 with (dest/'worker.log').open('w') as f:
  p=subprocess.Popen([PY,'-u',str(R/'worker.py'),'--mode',mode,'--phase',phase,'--out',str(dest)],env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  save(dest/'pid.json',{'pid':p.pid})
  if p.wait():raise RuntimeError(f'{phase}/{mode} failed; see worker.log')
def wait_checks(paths,children,timeout):
 end=time.monotonic()+timeout
 while not all(p.exists() for p in paths):
  if any(c.poll() is not None for c in children):raise RuntimeError('worker exited before acknowledgment')
  if time.monotonic()>end:raise TimeoutError(str(paths))
  time.sleep(.2)
def corun(mode):
 dest=R/'co'/mode;dest.mkdir(parents=True,exist_ok=False);children=[];daemon=False
 uuid=subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();pipe=dest/'pipe';pipe.mkdir();logs=dest/'mps_logs';logs.mkdir();env=dict(os.environ,PI_MPS='1',CUDA_VISIBLE_DEVICES=uuid,CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(logs))
 try:
  save(R/'status.json',{'state':'running','phase':'co','mode':mode,'gpu':GPU});p=subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,capture_output=True,text=True);save(dest/'mps_start.json',{'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr});assert p.returncode==0;daemon=True
  for role in ['server','train']:
   e=dict(env,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='30' if role=='train' else '100')
   argv=[PY,'-u',str(R/'server.py'),'--role','server','--out',str(dest)] if role=='server' else [PY,'-u',str(R/'worker.py'),'--phase','co','--mode',mode,'--out',str(dest)]
   with (dest/f'{role}.log').open('w') as f:child=subprocess.Popen(argv,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append(child)
  save(dest/'pids.json',{'server':children[0].pid,'train':children[1].pid});wait_checks([dest/'server_ready.json',dest/'train_ready.json'],children,7200)
  save(dest/'command_000.json',{'op':'calibrate'});wait_checks([dest/'server_done_000.json',dest/'train_done_000.json'],children,300)
  for i in range(1,4):
   cfg={'op':'run','name':f'window{i}','clients':1,'pattern':'sync','period_s':.250,'deadline_s':.250,'duration_s':60,'drain_s':5,'origin_ns':time.monotonic_ns()+2_000_000_000};save(dest/f'command_{i:03d}.json',cfg);wait_checks([dest/f'server_done_{i:03d}.json',dest/f'train_done_{i:03d}.json'],children,180)
   a=json.loads((dest/cfg['name']/'server_summary.json').read_text());b=json.loads((dest/cfg['name']/'train_summary.json').read_text());save(dest/cfg['name']/'summary.json',dict(a,training=b));render()
  save(dest/'command_004.json',{'op':'stop'})
  for c in children:
   if c.wait(timeout=60):raise RuntimeError('worker shutdown failed')
  save(dest/'complete.json',{'state':'complete'})
 finally:
  for c in children:
   if c.poll() is None:
    c.terminate()
    try:c.wait(timeout=15)
    except subprocess.TimeoutExpired:c.kill();c.wait()
  if daemon:subprocess.run(['nvidia-cuda-mps-control'],env=env,input='quit\n',text=True,capture_output=True,timeout=20)
def render():
 import report;report.render()
def main():
 save(R/'status.json',{'state':'preflight','gpu':GPU,'pid':os.getpid()});run_worker('A1','preflight');render()
 for mode in ['A0','A1','A2']:
  run_worker(mode,'solo');render();corun(mode);render()
 save(R/'status.json',{'state':'complete','gpu':GPU});render()
if __name__=='__main__':
 try:main()
 except Exception:save(R/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
