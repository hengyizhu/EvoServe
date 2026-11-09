import time,json,subprocess
from pathlib import Path
R=Path('/home/hengyi/pi05_cache_20261107')
while not (R/'learning_gate.json').exists():time.sleep(2)
gate=json.loads((R/'learning_gate.json').read_text())
if gate['state']=='failed':
 while not all(json.loads((R/f'gpu{g}_learning_status.json').read_text()).get('state')=='gate_failed' for g in [0,1]):time.sleep(1)
 jobs=[]
 for g in [0,1]:
  p=subprocess.Popen(['python3','-u',str(R/'learning_mirror_queue.py'),str(g)],stdout=open(R/f'mirror_queue_gpu{g}.log','w'),stderr=subprocess.STDOUT,start_new_session=True);jobs.append({'gpu':g,'pid':p.pid})
 (R/'gate_recovery.json').write_text(json.dumps({'state':'mirror_candidate','reason':'lowlight parent did not meet <60%','jobs':jobs},indent=2))
else:(R/'gate_recovery.json').write_text(json.dumps({'state':'not_needed'}))
