import subprocess,os,json
from pathlib import Path
r=Path('/home/hengyi/pi05_cache_20261107')
p=subprocess.run(['/home/hengyi/vla_followup_20261020/venv/bin/python','-u',str(r/'evaluate_gate.py'),'--name','parent_mirror_pilot','--task','3','--end','10','--mirror'],env=dict(os.environ,VLA_GPU='2',CUDA_VISIBLE_DEVICES='2',MUJOCO_EGL_DEVICE_ID='2'),stdout=open(r/'parent_mirror_pilot.log','w'),stderr=subprocess.STDOUT)
if p.returncode==0:(r/'parent_mirror_pilot_done.json').write_text(json.dumps({'state':'complete'}))
else:(r/'parent_mirror_pilot_failure.json').write_text(json.dumps({'returncode':p.returncode}))
