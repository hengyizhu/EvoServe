import os,time,json,subprocess
from pathlib import Path
r=Path('/home/hengyi/pi05_review_20261107');old=Path('/home/hengyi/pi05_cache_20261107');py='/home/hengyi/vla_followup_20261020/venv/bin/python';name='a_seed20261107'
def launch(gpu,task):
 part=f'{name}_task{task}'
 return subprocess.Popen([py,'-u',str(r/'evaluate_review.py'),'--name',part,'--delta',str(old/'learning'/name/'final.pt'),'--mirror','--end','10','--task',str(task)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),VLA_GPU=str(gpu)),stdout=open(r/(part+'_eval.log'),'w'),stderr=subprocess.STDOUT)
p0=launch(2,0)
while Path('/proc/3664502').exists():
 try:
  if Path('/proc/3664502/stat').read_text().split(') ',1)[1].split()[0]=='Z':break
 except FileNotFoundError:break
 time.sleep(1)
assert (r/'results/a_seed20261108_eval_manifest.json').exists()
p3=launch(0,3)
(r/'eval_status.json').write_text(json.dumps({'state':'running','name':name,'parallel_tasks':[0,3]}))
assert p0.wait()==0;assert p3.wait()==0
rr=[];man=[]
for task in [3,0]:
 rr.extend(json.loads(x) for x in (r/f'results/{name}_task{task}.jsonl').read_text().splitlines())
 man.extend(json.loads((r/f'results/{name}_task{task}_eval_manifest.json').read_text()))
assert len(rr)==20
(r/f'results/{name}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rr))
(r/f'results/{name}_eval_manifest.json').write_text(json.dumps(man,indent=2))
(r/'eval_status.json').write_text(json.dumps({'state':'complete','good_checkpoint_evaluation_gpus':{'task3':0,'task0':2}}))
