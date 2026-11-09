import time,json,subprocess,os
from pathlib import Path
r=Path('/home/hengyi/pi05_review_20261107')
while not (r/'precision_isolation_exit.json').exists():time.sleep(2)
p=subprocess.run(['/home/hengyi/vla_followup_20261020/venv/bin/python','-u',str(r/'openloop_review.py')],env=dict(os.environ,VLA_GPU='2'),stdout=open(r/'openloop_review.log','w'),stderr=subprocess.STDOUT)
(r/'openloop_exit.json').write_text(json.dumps({'returncode':p.returncode}))
