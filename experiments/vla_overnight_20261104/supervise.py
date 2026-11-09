import json,time,math,os,signal,traceback,hashlib
from pathlib import Path
from report import generate
ROOT=Path(__file__).resolve().parent

def save(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def main():
 os.sched_setaffinity(0,{31})
 started=time.monotonic();manifest=json.loads((ROOT/'manifest.json').read_text())
 while True:
  manifest=json.loads((ROOT/'manifest.json').read_text())
  states={p.parent.name:json.loads(p.read_text()) for p in ROOT.glob('gpu*/status.json') if 'preliminary' not in str(p)}
  for w in manifest['workers']:
   name='gpu3_phases' if w['gpu']==3 else f"gpu{w['gpu']}_{w['tier']}"
   if name not in states:continue
   proc=Path(f"/proc/{w['pid']}")
   dead=not proc.exists()
   if not dead:
    try:dead=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
    except FileNotFoundError:dead=True
   if dead and states[name].get('state') not in ('complete','failed'):
    states[name]={'state':'failed','reason':'Process exited without successful completion'};save(ROOT/name/'status.json',states[name])
  paths=[ROOT/f'gpu{i}_{"L3" if i==2 else "L2"}'/'calibration.json' for i in range(3)]
  if not (ROOT/'frozen_workloads.json').exists():
   if all(p.exists() for p in paths):
    calibrations=[json.loads(p.read_text()) for p in paths]
    c=max(x['inference']['mean'] for x in calibrations)
    margin=max(x['margin_s'] for x in calibrations)
    config={'common_c_s':c,'common_margin_s':margin,'periods':{str(rho):math.ceil((2*c/rho)/.05)*.05 for rho in [.4,.7]},'source':'Maximum serving-only mean among GPUs 0/1/2 before candidate matrix','deadline':'release+P','lookahead':'All outstanding/current and predicted releases within next 2P; same information S2/S4','training_cost':'per-card calibration p95 for each real legal segment; same estimates for S2/S4','margin_rule':'max across GPUs of max(0.010, calibration p95-mean)','warmup_seconds':20,'measurement_seconds':180,'drain_cap_seconds':60,'repeats':3,'frozen_before_matrix':True}
    save(ROOT/'frozen_workloads.json',config)
    manifest['frozen_workloads']=config
    manifest['sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')}
    for name in ['experiment.json','data_split.json','sample_sequence.json','audit/parent_download.json']:
     f=Path('/home/hengyi/vla_followup_20261020')/name
     manifest.setdefault('source_references',{})[name]={'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
    save(ROOT/'manifest.json',manifest)
   elif any(states.get(f'gpu{i}_{"L3" if i==2 else "L2"}',{}).get('state')=='failed' for i in range(3)):
    (ROOT/'STOP').write_text('At least one main GPU failed preflight; common workload was not frozen.\n')
  disk_bytes=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
  if disk_bytes>5*2**30 or time.monotonic()-started>8*3600:
   reason='artifact budget exceeded' if disk_bytes>5*2**30 else '8 hour execution guard reached'
   for w in manifest['workers']:
    name='gpu3_phases' if w['gpu']==3 else f"gpu{w['gpu']}_{w['tier']}"
    if states.get(name,{}).get('state') in ('complete','failed'):continue
    try:
     cmd=Path(f"/proc/{w['pid']}/cmdline").read_text()
     if str(ROOT) in cmd:
      os.kill(w['pid'],signal.SIGTERM);save(ROOT/name/'status.json',{'state':'failed','reason':reason})
    except ProcessLookupError:pass
  done=len(states)==4 and all(s.get('state') in ('complete','failed') for s in states.values())
  state='complete' if done and all(s['state']=='complete' for s in states.values()) else 'finished_with_failures' if done else 'running'
  save(ROOT/'status.json',{'state':state,'workers':states,'artifact_bytes':disk_bytes,'elapsed_seconds':time.monotonic()-started})
  try:generate(final=done)
  except Exception:save(ROOT/'report_error.json',{'traceback':traceback.format_exc()})
  if done:break
  time.sleep(30)

if __name__=='__main__':main()
