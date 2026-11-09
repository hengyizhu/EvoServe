"""Continue existing trainers and assign independent runs to four GPUs."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from learning_queue import ci
R=Path('/home/hengyi/pi05_cache_20261107')
PY='/home/hengyi/vla_followup_20261020/venv/bin/python'
GPU=int(sys.argv[1]);STATUS=R/f'gpu{GPU}_learning_status.json'
def save(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
def run(script,args,log):
 with open(R/log,'x') as f:
  p=subprocess.Popen([PY,'-u',str(R/script)]+args,env=dict(os.environ,VLA_GPU=str(GPU),CUDA_VISIBLE_DEVICES=str(GPU),MUJOCO_EGL_DEVICE_ID=str(GPU)),stdout=f,stderr=subprocess.STDOUT)
  code=p.wait()
 if code:raise RuntimeError(f'{script} exited {code}; see {log}')
def summarize():
 rows=[]
 for f in sorted((R/'results').glob('*.jsonl')):
  records=[]
  for line in f.read_text().splitlines():
   try:records.append(json.loads(line))
   except json.JSONDecodeError:pass # Other evaluator may be appending its next row.
  for task in sorted({x['task'] for x in records}):
   rr=[x for x in records if x['task']==task];n=len(rr);k=sum(x['success'] for x in rr)
   rows.append(dict(file=str(f),task=task,n=n,successes=k,rate=k/n,wilson95=ci(k,n)))
 save(R/'learning_summary.json',rows)
def main():
 assert json.loads((R/'learning_gate_mirror.json').read_text())['state']=='passed'
 plan=json.loads((R/'parallel_allocation.json').read_text())
 for job in plan['jobs'][str(GPU)]:
  variant,seed=job['variant'],job['seed'];name=f'{variant}_seed{seed}';out=R/'learning'/name
  save(STATUS,dict(state='training',name=name,gpu=GPU,queue_pid=os.getpid(),adopted_pid=job.get('existing_pid')))
  if job.get('existing_pid'):
   while alive(job['existing_pid']):time.sleep(2)
  else:
   assert not out.exists(),f'Refusing duplicate training: {out}'
   run('learn.py',['--variant',variant,'--seed',str(seed),'--brightness','1.0','--mirror'],f'{name}_train.log')
  trained=json.loads((out/'status.json').read_text())
  assert trained['state']=='trained' and trained['steps']==1000,trained
  assert (out/'final.pt').is_file()
  save(STATUS,dict(state='evaluating',name=name,gpu=GPU,queue_pid=os.getpid()))
  run('evaluate_gate.py',['--name',name,'--delta',str(out/'final.pt'),'--brightness','1.0','--mirror'],f'{name}_eval.log')
  rows=[json.loads(x) for x in (R/'results'/f'{name}.jsonl').read_text().splitlines()]
  assert len(rows)==100
  for task in [0,3]:assert sorted(x['index'] for x in rows if x['task']==task)==list(range(50))
  save(out/'evaluation_complete.json',dict(state='complete',episodes=100,gpu=GPU));summarize()
 save(STATUS,dict(state='complete',gpu=GPU))
if __name__=='__main__':
 try:main()
 except Exception:save(STATUS,dict(state='failed',gpu=GPU,traceback=traceback.format_exc()));raise
