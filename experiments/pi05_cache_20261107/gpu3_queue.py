import time,json,subprocess,os
from pathlib import Path
r=Path('/home/hengyi/pi05_cache_20261107');py='/home/hengyi/vla_followup_20261020/venv/bin/python'
while True:
 p=r/'prototype/status.json'
 if p.exists():
  d=json.loads(p.read_text())
  if d['state'] in ['complete','failed']:break
 time.sleep(2)
for script,args in [('precision_check.py',[]),('run_mps.py',['3','action'])]:
 with open(r/(script+'.log'),'w') as f:
  p=subprocess.run([py,'-u',str(r/script)]+args,env=dict(os.environ,VLA_GPU='3'),stdout=f,stderr=subprocess.STDOUT)
 (r/(script+'.exit.json')).write_text(json.dumps({'returncode':p.returncode}))
