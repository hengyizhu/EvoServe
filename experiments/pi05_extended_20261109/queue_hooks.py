import json,time,subprocess,math
from pathlib import Path
R=Path(__file__).resolve().parent;OLD=Path('/home/hengyi/pi05_overnight_20261108');L=Path('/home/hengyi/pi05_lora_lr_20261108');PY='/home/hengyi/vla_followup_20261020/venv/bin/python'
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def save(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def scores(name,root=R):
 p=root/'results'/f'{name}.jsonl'
 if not p.exists():return {}
 try:rows=[json.loads(x) for x in p.read_text().splitlines()]
 except json.JSONDecodeError:return {}
 out={}
 for task in set(x['task'] for x in rows):
  a=[x for x in rows if x['task']==task]
  if sorted(x['index'] for x in a)==list(range(50)):out[task]=sum(x['success'] for x in a)
 return out
last_render=0
def gates(add,jobs):
 global last_render
 def reg(name,args,status,priority,training=False):
  meta={'status_path':str(status),'round':'extended_20261109'}
  if training:meta.update(exclusive_group='cached_training',min_host_available_gib=130)
  add(name,'benchmark',args,[0,1,2,3],priority,meta)
 def evaluate(name,delta,perturb,tid=3,only=None,priority=0):
  args=[str(R/'eval_worker.py'),'--name',name,'--adapt-task',str(tid),'--perturb',str(perturb)]
  if delta:args+=['--delta',str(delta)]
  if only is not None:args+=['--task',str(only)]
  reg(name,args,R/'results'/f'{name}_status.json',priority)
 def train(name,family,seed,steps,task,perturb,checkpoints,priority,verify=None):
  args=[str(R/'train_cached.py'),'--name',name,'--family',family,'--lr','1e-5' if family=='full' else '5e-5','--seed',str(seed),'--steps',str(steps),'--split',str(R/'splits'/f'{task}.json'),'--sequence',str(R/'sequences'/f'{task}_{seed}_{steps}.json'),'--perturb',str(perturb),'--checkpoints',','.join(map(str,checkpoints))]
  if verify:args+=['--verify-2000',str(verify)]
  reg(name,args,R/'training'/name/'status.json',priority,True)
 low=R/'tasks/drawer_lowlight_0.json'
 evaluate('extended_full_s08_history',OLD/'training/curve_lowlight/ema_2000.pt',low,only=0)
 for family in ['full','lora32_heads']:
  for seed in [20261108,20261109,20261110]:
   if family=='full' and seed==20261108:continue
   name=f'extended_{family}_s{seed}';train(name,family,seed,2000,'drawer',low,[2000],1 if family=='full' else 2)
   ck=read(R/'training'/name/'checkpoint_2000.json')
   if ck.get('state')=='ready':evaluate(name+'_c2000',ck['path'],low)
 decisions={}
 for family in ['full','lora32_heads']:
  values=[]
  for seed in [20261108,20261109,20261110]:
   if family=='full' and seed==20261108:
    data={3:scores('curve_lowlight_c2000',L).get(3),0:scores('extended_full_s08_history').get(0)}
   else:data=scores(f'extended_{family}_s{seed}_c2000')
   if data.get(3) is None or data.get(0) is None:continue
   values.append({'seed':seed,'adaptation':data[3],'history':data[0]})
  decision={'state':'pending','scores':values}
  if len(values)==3:
   passed=all(v['adaptation']>=29 and v['history']>=39 for v in values) and max(v['adaptation'] for v in values)-min(v['adaptation'] for v in values)<=12.5
   decision['state']='stable' if passed else 'not_stable'
  decisions[family]=decision
 save(R/'decisions.json',decisions)
 # Curves use the previous full-action configuration; no retrospective selection.
 for task in ['book','microwave']:
  gate=read(R/f'{task}_gate.json');tid={'book':5,'microwave':9}[task]
  if not gate:
   for candidate in range(3):
    en=f'extended_parent_{task}_{candidate}';pc=R/'tasks'/f'{task}_{candidate}.json';evaluate(en,None,pc,tid,tid,3)
    k=scores(en).get(tid)
    if jobs.get(en,{}).get('state')!='complete' or k is None:break
    if k<25:gate={'state':'passed','successes':k,'episodes':50,'task_id':tid,'perturb':str(pc)};save(R/f'{task}_gate.json',gate);break
   else:gate={'state':'failed','reason':'all declared candidates parent>=50%'};save(R/f'{task}_gate.json',gate)
  if gate.get('state')=='passed':
   name=f'extended_curve_{task}';points=[100,250,500,1000,2000,3000,4000];train(name,'full',20261108,4000,task,gate['perturb'],points,6)
   for step in points:
    ck=read(R/'training'/name/f'checkpoint_{step}.json')
    if ck.get('state')=='ready':evaluate(f'{name}_c{step}',ck['path'],gate['perturb'],tid,tid,5)
 for task,previous in [('drawer_lowlight','curve_lowlight'),('drawer_shift','curve_camera_shift')]:
  name=f'extended_curve_{task}';pc=R/'tasks'/f'{task}_0.json';train(name,'full',20261108,4000,'drawer',pc,[3000,4000],4 if task=='drawer_lowlight' else 9,OLD/'training'/previous/'ema_2000.pt')
  for step in [3000,4000]:
   ck=read(R/'training'/name/f'checkpoint_{step}.json')
   if ck.get('state')=='ready':evaluate(f'{name}_c{step}',ck['path'],pc,3,3,5 if task=='drawer_lowlight' else 9)
 if time.monotonic()-last_render>60:
  subprocess.run([PY,str(R/'report.py')],timeout=30,check=True,stdout=subprocess.DEVNULL);last_render=time.monotonic()
