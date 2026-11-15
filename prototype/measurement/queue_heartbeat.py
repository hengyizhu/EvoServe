"""Independent heartbeat survives an experiment controller exiting."""
from pathlib import Path
import json,os,time,subprocess,fcntl,datetime
from zoneinfo import ZoneInfo
R=Path(__file__).resolve().parent
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def alive(pid):
 p=Path(f'/proc/{pid}/stat')
 return bool(pid and p.exists() and p.read_text().split(') ')[1].split()[0]!='Z')
with (R/'heartbeat.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (R/'heartbeat_pid.json').write_text(json.dumps(dict(pid=os.getpid())))
 while True:
  try:
   raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20);cards=[]
   for line in raw.strip().splitlines():
    i,m,u=map(int,line.split(','));assignment=read(R/'status.json').get('running',{}).get(str(i),{});assigned=alive(assignment.get('pid'));cards.append(dict(gpu=i,memory_mib=m,utilization_percent=u,idle=m<128 and u<5 and not assigned,assigned_live_task=assignment.get('task') if assigned else None))
   x=dict(time=time.time(),new_york_time=datetime.datetime.now(ZoneInfo('America/New_York')).isoformat(),heartbeat_interval_seconds=1800,host_load=list(os.getloadavg()),gpus=cards,queue=read(R/'status.json'),active_process=read(R/'active_process.json'),controllers={key:dict(pid=(p:=read(folder/'controller_pid.json')).get('pid'),alive=alive(p.get('pid'))) for key,folder in [('deployment_queue',R)]})
  except Exception as exc:x=dict(time=time.time(),heartbeat_error=repr(exc))
  tmp=R/'queue_status.tmp';tmp.write_text(json.dumps(x,indent=2));tmp.replace(R/'queue_status.json')
  with (R/'queue_heartbeats.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
  time.sleep(1800)
