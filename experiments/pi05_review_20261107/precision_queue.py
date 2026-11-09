import time,json,subprocess,os
from pathlib import Path
r=Path('/home/hengyi/pi05_review_20261107')
while True:
 p=r/'numerical_status.json'
 if p.exists() and json.loads(p.read_text()).get('state') in ['complete','failed']:break
 time.sleep(2)
p=subprocess.run(['/home/hengyi/vla_followup_20261020/venv/bin/python','-u',str(r/'precision_isolation.py')],env=dict(os.environ,VLA_GPU='2'),stdout=open(r/'precision_isolation.log','w'),stderr=subprocess.STDOUT)
(r/'precision_isolation_exit.json').write_text(json.dumps({'returncode':p.returncode}))
