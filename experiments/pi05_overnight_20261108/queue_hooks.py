import json,time,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;L=Path('/home/hengyi/pi05_lora_lr_20261108');PY='/home/hengyi/vla_followup_20261020/venv/bin/python'
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def save(p,d):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def count(name):
 p=L/'results'/f'{name}.jsonl'
 if not p.exists():return None
 try:rows=[json.loads(x) for x in p.read_text().splitlines()]
 except json.JSONDecodeError:return None
 if sorted(x['index'] for x in rows)!=list(range(50)):return None
 return sum(x['success'] for x in rows)
last_render=0
def gates(add,jobs,selection):
 global last_render
 def reg(jid,args,status,priority):add(jid,'benchmark',args,[0,1],priority,{'status_path':str(status)})
 if selection:
  add('overnight_tail','benchmark',[str(R/'tail_run.py')],[0,1],8,{'status_path':str(R/'tail/status.json'),'requires_idle_gpus':[2,3]})
  for task in ['lowlight','camera_shift','colorcast']:
   gate=read(R/f'{task}_gate.json')
   if task=='lowlight' and not gate:gate={'state':'passed','successes':16,'episodes':50,'perturb':str(R/'tasks/lowlight_0.json'),'reused':'pi05_stability_round2 task_gate'};save(R/f'{task}_gate.json',gate)
   if not gate:
    for ci in range(3):
     name=f'overnight_parent_{task}_{ci}';reg(name,[str(L/'eval_worker.py'),'--name',name,'--task','3','--perturb',str(R/'tasks'/f'{task}_{ci}.json')],L/'results'/f'{name}_status.json',14);k=count(name)
     if jobs.get(name,{}).get('state')!='complete' or k is None:break
     if k<25:gate={'state':'passed','successes':k,'episodes':50,'perturb':str(R/'tasks'/f'{task}_{ci}.json')};save(R/f'{task}_gate.json',gate);break
    else:gate={'state':'failed','reason':'all3declared perturbations parent>=50%'};save(R/f'{task}_gate.json',gate)
   if gate.get('state')=='passed':
    name='curve_'+task;reg(name,[str(R/'train.py'),'--name',name,'--family',selection['family'],'--lr',str(selection['lr']),'--seed','20261108','--steps','2000','--perturb',gate['perturb']],R/'training'/name/'status.json',18)
    for step in [100,250,500,1000,2000]:
     ready=read(R/'training'/name/f'checkpoint_{step}.json');ename=f'{name}_c{step}'
     if ready.get('state')=='ready':reg(ename,[str(L/'eval_worker.py'),'--name',ename,'--task','3','--delta',ready['path'],'--perturb',gate['perturb']],L/'results'/f'{ename}_status.json',17)
 if time.monotonic()-last_render>60:
  subprocess.run([PY,str(R/'report.py')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30);last_render=time.monotonic()
