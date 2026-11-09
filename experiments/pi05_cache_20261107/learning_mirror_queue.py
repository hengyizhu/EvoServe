import os,sys,json,time,subprocess,traceback,math
from pathlib import Path
R=Path('/home/hengyi/pi05_cache_20261107');PY='/home/hengyi/vla_followup_20261020/venv/bin/python'
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def run(gpu,script,args,log):
 with open(R/log,'w') as f:
  p=subprocess.run([PY,'-u',str(R/script)]+args,env=dict(os.environ,VLA_GPU=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu)),stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'{script} failed; {log}')
def ci(k,n):
 z=1.959963984540054;p=k/n;c=(p+z*z/(2*n))/(1+z*z/n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return [c-h,c+h]
def summarize():
 rows=[]
 for f in (R/'results').glob('*.jsonl'):
  rr=[json.loads(x) for x in f.read_text().splitlines()]
  for task in sorted({x['task'] for x in rr}):
   a=[x for x in rr if x['task']==task];k=sum(x['success'] for x in a);rows.append({'file':str(f),'task':task,'n':len(a),'successes':k,'rate':k/len(a),'wilson95':ci(k,len(a))})
 save(R/'learning_summary.json',rows)
def main():
 gpu=int(sys.argv[1]);status=R/f'gpu{gpu}_learning_status.json'
 while not (R/'parent_mirror_pilot_done.json').exists():time.sleep(2)
 save(status,{'state':'parent_mirror_gate','brightness':1.0,'mirror':True,'indices':[(10+gpu*20),(10+gpu*20)+20]})
 run(gpu,'evaluate_gate.py',['--name',f'parent_mirror_part{gpu}','--task','3','--brightness','1.0','--mirror','--start',str((10+gpu*20)),'--end',str((10+gpu*20)+20)],f'parent_mirror_part{gpu}.log')
 save(R/f'parent_mirror_part{gpu}_done.json',{'state':'complete'})
 if gpu==0:
  while not (R/'parent_mirror_part1_done.json').exists():time.sleep(2)
  rows=[json.loads(x) for g in [0,1] for x in (R/f'results/parent_mirror_part{g}.jsonl').read_text().splitlines()]
  rows=[json.loads(x) for x in (R/'results/parent_mirror_pilot.jsonl').read_text().splitlines()]+rows
  assert sorted(x['index'] for x in rows)==list(range(50))
  (R/'results/parent_mirror.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));summarize();rate=sum(x['success'] for x in rows)/len(rows)
  save(R/'learning_gate_mirror.json',{'state':'passed' if rate<.6 and len(rows)==50 else 'failed','parent_rate':rate,'n':len(rows),'brightness':1.0,'mirror':True,'task':'custom horizontally mirrored drawer, both cameras, brightness unchanged','history':'unmodified task0','clean_parent':{'success':30,'n':50,'rate':.6},'eval_indices':list(range(50)),'denoise_steps':5})
 else:
  while not (R/'learning_gate_mirror.json').exists():time.sleep(2)
 gate=json.load(open(R/'learning_gate_mirror.json'))
 if gate['state']!='passed':save(status,{'state':'gate_failed','gate':gate});return
 # Run primary three variants first; GPU1 then adds seed for c, GPU0 adds seed for a.
 jobs=[('a',20261107),('b',20261107),('a',20261108)] if gpu==0 else [('c',20261107),('c',20261108)]
 for variant,seed in jobs:
  name=f'{variant}_seed{seed}';save(status,{'state':'training','name':name})
  run(gpu,'learn.py',['--variant',variant,'--seed',str(seed),'--brightness','1.0','--mirror'],f'{name}_train.log')
  save(status,{'state':'evaluating','name':name})
  run(gpu,'evaluate_gate.py',['--name',name,'--delta',str(R/'learning'/name/'final.pt'),'--brightness','1.0','--mirror'],f'{name}_eval.log');summarize()
  save(R/'learning'/name/'evaluation_complete.json',{'state':'complete','episodes':100})
 save(status,{'state':'complete'})
if __name__=='__main__':
 try:main()
 except Exception:save(R/f'gpu{sys.argv[1]}_learning_status.json',{'state':'failed','traceback':traceback.format_exc()});raise
