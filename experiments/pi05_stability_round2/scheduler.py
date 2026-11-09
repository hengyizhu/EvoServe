"""Persistent, non-preemptive queue. Never signals or reclaims foreign GPU jobs."""
import os,sys,json,time,math,subprocess,fcntl,traceback,shutil
from pathlib import Path
R=Path('/home/hengyi/pi05_stability_round2');PY='/home/hengyi/vla_followup_20261020/venv/bin/python'
def save(p,d):
 p=Path(p);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False));tmp.replace(p)
def read(p):
 try:return json.loads(Path(p).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def rows(p):
 if not Path(p).exists():return []
 return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def inventory():
 output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);cards={}
 for line in output.splitlines():
  i,u,m,v=[x.strip() for x in line.split(',')];cards[int(i)]={'uuid':u,'memory_mib':int(m),'util_pct':int(v),'compute_pids':[]}
 output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 for line in output.splitlines():
  if ',' not in line:continue
  uuid,pid=[x.strip() for x in line.split(',')]
  for card in cards.values():
   if card['uuid']==uuid:card['compute_pids'].append(int(pid))
 return cards
def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
def judge_stability(values,parent_successes):
 required=max(20,math.ceil(parent_successes+12.5))
 rates=[x[3]/50 for x in values]
 passed=all(x[3]>=required and x[0]>=39 for x in values) and max(rates)-min(rates)<=.25+1e-12
 return {'state':'stable' if passed else 'not_stable','results':values,'adaptation_required_successes':required,'adaptation_range_pp':100*(max(rates)-min(rates))}
def main():
 lock=(R/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg=read(R/'protocol.json');jobs=read(R/'jobs.json') or {};active={};idle_since={};last_report=0
 def add(jid,kind,args,gpus,priority,meta=None):
  if jid not in jobs:jobs[jid]={'id':jid,'kind':kind,'args':args,'gpus':[0,1],'priority':priority,'state':'pending','meta':meta or {}}
 add('preflight_learning','preflight',['preflight.py','--kind','learning'],[0,1,3],0)
 add('preflight_transfer','preflight',['preflight.py','--kind','transfer'],[2],0)
 # A restarted scheduler adopts live direct workers; it does not duplicate their outputs.
 for jid,j in jobs.items():
  if j['state']=='running':
   if alive(j['pid']):active[jid]=None
   elif j['kind'] in ['screen','eval'] and (R/'results'/f'{jid}_eval_manifest.json').exists():j['state']='complete';j['completion_evidence']='finished evaluation manifest recovered on restart'
   else:j['state']='failed';j['reason']='scheduler restarted after worker disappeared; inspect output before retry'
 def succeeded(jid):return jobs.get(jid,{}).get('state')=='complete'
 def eval_success(name):
  a=rows(R/'results'/f'{name}.jsonl')
  if len(a)!=100:return None
  for t in [3,0]:
   if sorted(x['index'] for x in a if x['task']==t)!=list(range(50)):return None
  return {t:sum(x['success'] for x in a if x['task']==t) for t in [3,0]}
 def gates():
  if succeeded('preflight_transfer'):add('transfer','transfer',['run_transfer.py','1','action'],[0,1],2)
  if succeeded('transfer'):jobs['preflight_learning']['gpus']=[0,1]
  if succeeded('preflight_learning') and not read(R/'task_gate.json'):
   for ci,brightness in enumerate(cfg['task_screening']['candidate_brightness']):
    prefix=f'parent_light_{ci}'
    ids=[]
    for part in range(5):
     jid=f'{prefix}_part{part}';ids.append(jid);add(jid,'screen',['eval_worker.py','--name',jid,'--task','3','--brightness',str(brightness),'--start',str(part*10),'--end',str(part*10+10)],[0,1,2,3],1,{'candidate':ci,'brightness':brightness})
    if not all(succeeded(j) for j in ids):break
    data=[x for jid in ids for x in rows(R/'results'/f'{jid}.jsonl')];assert sorted(x['index'] for x in data)==list(range(50));k=sum(x['success'] for x in data);save(R/f'screen_{ci}.json',{'brightness':brightness,'successes':k,'episodes':50,'rate':k/50})
    if k<25:
     save(R/'task_gate.json',{'state':'passed','brightness':brightness,'parent_successes':k,'episodes':50,'parent_rate':k/50,'task':'fixed low-exposure drawer, both cameras; no mirroring','threshold':'parent<50%','required_adaptation_successes':max(20,math.ceil(k+12.5))});break
   else:save(R/'task_gate.json',{'state':'failed','reason':'none of the predeclared natural exposure candidates had parent success<50%'})
  gate=read(R/'task_gate.json')
  if gate.get('state')!='passed':return
  for family in cfg['learning']['families']:
   if family=='lora32' and cfg['learning'].get('lora_condition')=='only_if_full_a_not_stable' and read(R/'stability_decisions.json').get('full',{}).get('a',{}).get('state')!='not_stable':continue
   for variant in cfg['learning'].get('variants_by_family',{}).get(family,['a','b']):
    for seed in cfg['learning']['seeds']:
     name=f'{family}_{variant}_s{seed}';add(name,'train',['train_worker.py','--family',family,'--variant',variant,'--seed',str(seed),'--name',name],[0,1,2,3],5 if family=='lora32' else 20,{'family':family,'variant':variant,'seed':seed,'name':name,'stage':1})
     for step in cfg['learning'].get('eval_steps',[500,1000]):
      ready=read(R/'training'/name/f'checkpoint_{step}.json')
      if ready.get('state')=='ready':
       ename=f'{name}_c{step}';add(ename,'eval',['eval_worker.py','--name',ename,'--delta',ready['path'],'--brightness',str(gate['brightness'])],[0,1,2,3],(4 if family=='lora32' else 10) if step==1000 else 30,{'family':family,'variant':variant,'seed':seed,'step':step,'stage':1})
  # Fixed final EMA checkpoint only, never per-seed best-checkpoint selection.
  decisions={}
  for family in cfg['learning']['families']:
   decisions[family]={}
   for variant in ['a','b']:
    vals=[eval_success(f'{family}_{variant}_s{s}_c1000') for s in cfg['learning']['seeds']]
    if any(x is None for x in vals):decisions[family][variant]={'state':'pending'};continue
    decisions[family][variant]=judge_stability(vals,gate['parent_successes'])
  if cfg['learning'].get('lora_condition')=='only_if_full_a_not_stable':
   stable=decisions['full']['a']['state']=='stable'
   for job in jobs.values():
    if job.get('meta',{}).get('family')=='lora32':
     if stable and job['state']=='pending':job.update(state='deferred',reason='Full+a stable on all three seeds; LoRA deferred by user')
     elif decisions['full']['a']['state']=='not_stable' and job['state']=='deferred':job['state']='pending'
   if stable:
    for variant in ['a','b']:decisions['lora32'][variant]={'state':'deferred','reason':'full+a passed stability gate'}
  if cfg['learning'].get('lora_condition')=='explicit_b_only':decisions['lora32']['a']={'state':'deferred','reason':'User requested LoRA+b only; a remains deferred'}
  save(R/'stability_decisions.json',decisions)
  combined_names=[f'full_{v}_s{s}_c1000' for v in ['a','b'] for s in cfg['learning']['seeds']]+[f'e2e_full_{mode}_s{s}_c1000' for mode in ['cache','recompute'] for s in cfg['stage3']['seeds']]
  if all(eval_success(name) is not None for name in combined_names):save(R/'combined_results_ready.json',{'state':'ready','updated':time.time(),'results':{name:eval_success(name) for name in combined_names},'reporting':'combined learning report; no separate early cache/recompute report'})
  selection=read(R/'selection.json')
  if not selection:
   if decisions['full']['a']['state']=='stable':selection={'state':'selected','family':'full','step':1000,'based_on':'a, all3seeds, fixed final EMA checkpoint; b judged independently'}
   elif decisions['full']['a']['state']=='not_stable' and decisions['lora32']['a']['state']=='stable':selection={'state':'selected','family':'lora32','step':1000,'based_on':'a, all3seeds; full did not pass; b judged independently'}
   elif all(decisions[f]['a']['state']=='not_stable' for f in ['full','lora32']):selection={'state':'no_stable_config','reason':'neither family passed the agreed final-checkpoint a criteria'}
   if selection:save(R/'selection.json',selection)
  if selection.get('state')=='selected':
   for cache in [False,True]:
    for seed in cfg['stage3']['seeds']:
     mode='cache' if cache else 'recompute';name=f'e2e_{selection["family"]}_{mode}_s{seed}';argv=['train_worker.py','--family',selection['family'],'--variant','b','--seed',str(seed),'--name',name,'--stage','3']+(['--cache'] if cache else [])
     add(name,'train',argv,[3],15,{'stage':3,'family':selection['family'],'mode':mode,'seed':seed})
     ready=read(R/'training'/name/'checkpoint_1000.json')
     if ready.get('state')=='ready':add(name+'_c1000','eval',['eval_worker.py','--name',name+'_c1000','--delta',ready['path'],'--brightness',str(gate['brightness'])],[3],9,{'stage':3,'mode':mode,'seed':seed,'family':selection['family']})
 while True:
  # Completion uses worker exit status, not nvidia-smi utilization alone.
  for jid,p in list(active.items()):
   j=jobs[jid];code=p.poll() if p is not None else (None if alive(j['pid']) else 0)
   if code is None:continue
   if p is None:
    if j['kind']=='train':ok=read(R/'training'/j['meta']['name']/'status.json').get('state')=='complete' if 'name' in j['meta'] else read(R/'training'/jid/'status.json').get('state')=='complete'
    elif j['kind'] in ['screen','eval']:ok=(R/'results'/f'{jid}_eval_manifest.json').exists()
    elif j['kind']=='benchmark':ok=read(j['meta']['status_path']).get('state')=='complete'
    elif j['kind']=='transfer':ok=read(R/'matrix_replicates/action/status.json').get('state')=='complete'
    else:ok=read(R/f'{jid}.json').get('state')=='passed'
    code=0 if ok else 1
   j.update(state='complete' if code==0 else 'failed',returncode=code,finished=time.time());del active[jid]
  gates()
  if (Path('/home/hengyi/pi05_lora_lr_20261108')/'protocol.json').exists():
   import importlib.util
   if 'lr_hook' not in locals():
    spec=importlib.util.spec_from_file_location('lr_sweep_queue_hooks','/home/hengyi/pi05_lora_lr_20261108/queue_hooks.py');lr_hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(lr_hook)
   lr_hook.gates(add,jobs)
  if Path('/home/hengyi/pi05_extended_20261109/protocol.json').exists():
   import importlib.util
   if 'extended_hook' not in locals():
    spec=importlib.util.spec_from_file_location('extended_queue_hooks','/home/hengyi/pi05_extended_20261109/queue_hooks.py');extended_hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(extended_hook)
   extended_hook.gates(add,jobs)
  for request in read(R/'external_jobs.json') or []:
   if any(read(Path(dep['path'])).get('state')!=dep['state'] for dep in request.get('requires',[])):continue
   add(request['id'],request['kind'],request['args'],request['gpus'],request['priority'],request.get('meta'))
  # User authorized four GPUs only after the latency-sensitive run finishes.
  policy=read(R/'gpu_policy.json')
  tail_finished=(jobs.get('overnight_tail',{}).get('state')=='complete' and read(Path('/home/hengyi/pi05_overnight_20261108/tail/status.json')).get('state')=='complete')
  allowed_gpus=[0,1,2,3] if policy.get('four_gpu_after_tail') and tail_finished else [0,1]
  for queued in jobs.values():
   if queued['state']=='pending':queued['gpus']=[0,1] if queued['id']=='overnight_tail' else allowed_gpus.copy()
  cards=inventory();occupied={jobs[j]['gpu'] for j in active}
  for gpu,card in cards.items():
   idle=(gpu not in occupied and not card['compute_pids'] and card['memory_mib']<1024 and card['util_pct']<5)
   if idle:idle_since.setdefault(gpu,time.monotonic())
   else:idle_since.pop(gpu,None)
  for gpu in sorted(g for g in cards if g in allowed_gpus):
   if gpu in occupied or gpu not in idle_since or time.monotonic()-idle_since[gpu]<90:continue
   if shutil.disk_usage(R).free<24*2**30:continue
   eligible=[j for j in jobs.values() if j['state']=='pending' and gpu in j['gpus'] and all(g in idle_since and time.monotonic()-idle_since[g]>=90 for g in j.get('meta',{}).get('requires_idle_gpus',[])) and (gpu!=2 or j['kind'] in ['preflight','transfer'] or succeeded('transfer'))]
   host_available_gib=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))/1024**2
   active_groups={jobs[j].get('meta',{}).get('exclusive_group') for j in active}
   eligible=[j for j in eligible if host_available_gib>=j.get('meta',{}).get('min_host_available_gib',0) and (not j.get('meta',{}).get('exclusive_group') or j['meta']['exclusive_group'] not in active_groups)]
   if not eligible:continue
   j=min(eligible,key=lambda x:(x['priority'],x['id']));env=dict(os.environ,VLA_GPU=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
   if j['kind']=='transfer':j['args'][1]=str(gpu)
   log=R/'logs'/f'{j["id"]}.log';log.parent.mkdir(exist_ok=True)
   with log.open('a') as f:p=subprocess.Popen([PY,'-u',str(R/j['args'][0])]+j['args'][1:],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   j.update(state='running',pid=p.pid,gpu=gpu,started=time.time(),log=str(log));active[j['id']]=p;occupied.add(gpu);idle_since.pop(gpu,None)
  save(R/'jobs.json',jobs);save(R/'queue_status.json',{'state':'running' if active else 'waiting_for_idle_gpu','pid':os.getpid(),'updated':time.time(),'cards':cards,'allowed_gpus':allowed_gpus,'four_gpu_after_tail':bool(policy.get('four_gpu_after_tail')),'active':list(active),'pending':sum(j['state']=='pending' for j in jobs.values()),'completed':sum(j['state']=='complete' for j in jobs.values()),'failed':[j for j in jobs if jobs[j]['state']=='failed'],'disk_free_gib':shutil.disk_usage(R).free/2**30})
  if time.monotonic()-last_report>20:
   try:
    import report;report.render()
   except Exception:save(R/'report_error.json',{'traceback':traceback.format_exc()})
   last_report=time.monotonic()
  if not active and not any(j['state']=='pending' for j in jobs.values()):
   terminal=read(R/'task_gate.json').get('state')=='failed' or (read(R/'selection.json') and all(j['state'] in ['complete','failed','cancelled','deferred'] for j in jobs.values())) or any(j['state']=='failed' for j in jobs.values())
   if terminal:
    final_state='complete' if all(j['state'] in ['complete','cancelled','deferred'] for j in jobs.values()) else 'finished_with_failures'
    if read(R/'task_gate.json').get('state')=='failed':final_state='screening_failed'
    elif read(R/'selection.json').get('state')=='no_stable_config':final_state='complete_no_stable_config'
    save(R/'queue_final.json',{'state':final_state,'selection':read(R/'selection.json'),'task_gate':read(R/'task_gate.json')})
    status=read(R/'queue_status.json');status['state']=final_state;save(R/'queue_status.json',status)
    import report;report.render()
    break
  time.sleep(15)
if __name__=='__main__':
 try:main()
 except BlockingIOError:raise SystemExit('scheduler already running')
 except Exception:save(R/'scheduler_failure.json',{'traceback':traceback.format_exc()});raise
