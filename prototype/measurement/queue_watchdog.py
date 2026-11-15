"""Recover a dead controller only at an idle worker boundary. Never stop experiments."""
from pathlib import Path
import os, json, time, subprocess, fcntl, traceback
R=Path(__file__).resolve().parent
EXCLUDED={'controller.py','queue_watchdog.py','queue_heartbeat.py','report.py'}
def read(p):
 try:return json.loads(Path(p).read_text())
 except (OSError,ValueError):return {}
def save(p,x):
 p=Path(p);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def script_processes():
 found={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid() or p.joinpath('stat').read_text().split(') ')[1].split()[0]=='Z':continue
   args=p.joinpath('cmdline').read_bytes().split(b'\0')
   # Match script arguments, not shell commands mentioning this directory.
   scripts=[a.decode() for a in args[1:] if a.startswith((str(R)+'/').encode()) and a.endswith(b'.py')]
   if scripts:found[int(p.name)]=scripts[0]
  except (OSError,ValueError,IndexError):continue
 # The prototype lives in the repository, outside this experiment directory.
 # Honor the recorded live command without matching unrelated shell mentions.
 active=read(R/'active_process.json');pid=active.get('pid')
 if pid:
  try:
   args=[x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
   expected=active.get('cmd',[])
   if args==expected and 'prototype/run_campaign.py' in ' '.join(args):found[pid]=args[1]
  except (OSError,ValueError):pass
 return found
def decision(controller_alive,workers,terminal):
 if controller_alive:return 'controller_alive'
 if workers:return 'waiting_for_existing_workers'
 if terminal:return 'finished'
 return 'restart_controller'
def reconcile():
 # A complete 10-minute run can survive its controller. Recover only this
 # queue's schema, never synthesize a successful trace before CSV validation.
 a=read(R/'active_process.json');cmd=a.get('cmd',[])
 if '--out' not in cmd or '--smoke' in cmd:return
 out=Path(cmd[cmd.index('--out')+1]);result=read(out/'result.json')
 if out.parent!=R/'system/runs' or result.get('state')!='complete':return
 name=out.name.rsplit('_formal_',1)[0]
 if name not in [f'expert{n}_u{u}' for n in [1,2,4,8] for u in [75,80]]:return
 import controller
 n=int(name.split('_')[0][6:])
 kernel=read(out/'pair0/profile/longest_kernel.json')
 record=dict(kernel_profile=kernel,state='complete',case=name,mode='A2',experts=n,load=result['target'],raw=str(out),result=result,expert_steps_s=n*result['total_training_steps_per_s'],mean_duty=controller.duty(out,result),trace=False,recovered=True)
 save(R/'task_state'/f'{name}_formal.json',record)
def main():
 with (R/'watchdog.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  save(R/'watchdog_pid.json',dict(pid=os.getpid()))
  while True:
   try:
    ps=script_processes();controllers=[p for p,s in ps.items() if s==str(R/'controller.py')];workers={p:s for p,s in ps.items() if Path(s).name not in EXCLUDED}
    state=read(R/'status.json');action=decision(bool(controllers),workers,state.get('state','').startswith('finished'))
    record=dict(time=time.time(),action=action,controller_pids=controllers,workers=workers,poll_seconds=60)
    if action=='restart_controller':
     reconcile()
     with (R/'logs/controller.log').open('a') as f:
      p=subprocess.Popen(['/usr/bin/python3','-u',str(R/'controller.py')],cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     record['restarted_pid']=p.pid
     with (R/'watchdog_restarts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    save(R/'watchdog_status.json',record)
   except Exception:save(R/'watchdog_status.json',dict(time=time.time(),error=traceback.format_exc()))
   time.sleep(60)
if __name__=='__main__':main()
