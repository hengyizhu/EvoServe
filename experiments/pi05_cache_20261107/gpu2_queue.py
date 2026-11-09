import time,json,subprocess,os
from pathlib import Path
r=Path('/home/hengyi/pi05_cache_20261107');py='/home/hengyi/vla_followup_20261020/venv/bin/python'
while True:
 p=r/'cache_cost/status.json'
 if p.exists() and json.loads(p.read_text()).get('state') in ['complete','failed']:break
 time.sleep(2)
for script,args in [('service_kv_check.py',[]),('learn.py',['--variant','a','--seed','909090','--steps','2','--brightness','.35'])]:
 with open(r/(script+'.gpu2.log'),'w') as f:
  p=subprocess.run([py,'-u',str(r/script)]+args,env=dict(os.environ,VLA_GPU='2'),stdout=f,stderr=subprocess.STDOUT)
 (r/(script+'.gpu2.exit.json')).write_text(json.dumps({'returncode':p.returncode}))
