import time,json,subprocess
from pathlib import Path
r=Path('/home/hengyi/pi05_review_input_20261107');prior=Path('/home/hengyi/pi05_review_20261107/matrix_replicates/action/status.json')
while not prior.exists() or json.loads(prior.read_text()).get('state')!='finished':time.sleep(2)
p=subprocess.run(['/home/hengyi/vla_followup_20261020/venv/bin/python','-u',str(r/'run_mps.py'),'3','action'],stdout=open(r/'run.log','w'),stderr=subprocess.STDOUT)
(r/'exit.json').write_text(json.dumps({'returncode':p.returncode}))
