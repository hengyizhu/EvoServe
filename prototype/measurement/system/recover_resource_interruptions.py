"""Finish measurements interrupted by GPU occupancy; preserve the original audit trail.

Waits for the original queue to exit, then only resumes resource-interrupted tasks.
GPU contention is a deferred admission, not a programming-error retry. Actual
execution errors still retry once and then skip. No other process is terminated.
"""
import argparse,datetime,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from zoneinfo import ZoneInfo
import adaptive_campaign as campaign
from measure import save,read,observe

def resource_error(out):
 p=Path(out)/'failure.json'
 text=p.read_text() if p.exists() else ''
 return any(s in text for s in ['GPU1 no longer idle','GPU0/1 have compute processes','Required GPU(s)','External compute process on GPU0'])

def main(root,resume=False):
 root=Path(root);last_heartbeat=0
 def heartbeat(label):
  nonlocal last_heartbeat
  if time.monotonic()-last_heartbeat<1800:return
  last_heartbeat=time.monotonic();x=observe();x['recovery_status']=label
  x['gpu_processes']={str(g):subprocess.check_output(['nvidia-smi',f'--id={g}','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip() for g in [0,1,2,3]}
  with (root/'recovery_heartbeats.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 save(root/'recovery_plan.json',dict(state='resuming' if resume else 'waiting_for_original_queue',pid=os.getpid(),created=time.time(),resource_policy='Only GPU0 must be idle; GPU1/2/3 are monitored',priority=['inference_only','fixed1','mps30_1','boundary80_m1','boundary80_m2','boundary80_m4','boundary80_m8']))
 if not resume:
  while read(root/'queue_status.json')['state']!='finished':heartbeat('waiting_for_original_queue');time.sleep(10)
 time.sleep(2);shutil.copy2(root/'queue_status.json',root/f'queue_before_resume_{int(time.time())}.json');state=read(root/'queue_status.json')
 if not any(i['name']=='inference_only' for i in state['items']):state['items'].append(dict(name='inference_only',configuration='inference_only',state='pending'))
 previous_pid=state['pid'];state.update(state='running_recovery',pid=os.getpid(),previous_supervisor_pid=previous_pid,recovery_started=time.time(),external_load_policy=campaign.EXTERNAL_POLICY);state.pop('finished',None)
 def persist():
  state['updated']=datetime.datetime.now(ZoneInfo('America/New_York')).isoformat();save(root/'queue_status.json',state);campaign.update_report(root)
 todo=[i for i in state['items'] if i['state'] in ['pending','deferred_resource','waiting_resource'] or (i['state']=='failed' and resource_error(i.get('out','')))]
 order=['inference_only','fixed1','mps30_1','boundary80_m1','boundary80_m2','boundary80_m4','boundary80_m8'];todo.sort(key=lambda i:order.index(i['name']) if i['name'] in order else 999)
 for item in todo:
  item.setdefault('attempt_history',[]).append({k:v for k,v in item.items() if k!='attempt_history'})
  item['state']='deferred_resource'
 persist();save(root/'recovery_plan.json',dict(state='running',pid=os.getpid(),tasks=[i['name'] for i in todo]))
 for item in todo:
  errors=0;attempt=0;state['current']=item['name']
  while errors<2:
   item['state']='waiting_resource';persist();idle_since=None
   while True:
    active={str(g):subprocess.check_output(['nvidia-smi',f'--id={g}','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip() for g in [0]}
    now=time.monotonic();idle_since=(idle_since or now) if not any(active.values()) else None
    with (root/'resource_admission.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),task=item['name'],processes=active,idle_seconds=now-idle_since if idle_since else 0))+'\n')
    heartbeat('waiting_resource')
    if idle_since is not None and now-idle_since>=30:break
    time.sleep(5)
   while True:
    attempt+=1;out=root/f"{item['name']}_recovery{attempt}"
    if not out.exists():break
   cfg=item['name'] if item['name'] in ['mps30_1','inference_only'] else 'fixed1'
   cmd=[sys.executable,'-u',str(root/'code/adaptive_campaign.py'),'run','--root',str(out),'--configuration',cfg,'--phase-seconds','600']
   if item['name'].startswith('boundary80_m'):cmd+=['--constant-load','.8','--initial',item['name'].split('_m')[-1]]
   item.update(state='running',out=str(out),recovery_attempt=attempt,started=time.time());item.pop('error',None);item.pop('finished',None);persist()
   with (root/f"{item['name']}_recovery{attempt}.log").open('w') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   item['pid']=p.pid;persist()
   while p.poll() is None:heartbeat('measuring');time.sleep(5)
   item['finished']=time.time()
   if p.returncode==0 and (out/'result.json').exists():item['state']='complete';persist();break
   item.setdefault('attempt_history',[]).append(dict(out=str(out),returncode=p.returncode,resource_interruption=resource_error(out),finished=time.time()))
   if resource_error(out):item.update(state='deferred_resource',error='GPU occupancy interrupted the exclusive measurement; raw data preserved, awaiting a new idle interval');persist();continue
   errors+=1;item.update(state='retrying' if errors<2 else 'failed',error=f'exit {p.returncode}; {out}/failure.json');persist()
 state.update(state='finished',current=None,finished=time.time());persist();save(root/'recovery_plan.json',dict(state='finished',pid=os.getpid(),finished=time.time()))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 try:main(a.root,a.resume)
 except Exception:save(Path(a.root)/'recovery_failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
