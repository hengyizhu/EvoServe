import subprocess,os,json
from pathlib import Path
r=Path('/home/hengyi/pi05_review_20261107');old=Path('/home/hengyi/pi05_cache_20261107');py='/home/hengyi/vla_followup_20261020/venv/bin/python'
for name in ['a_seed20261108','a_seed20261107']:
 (r/'eval_status.json').write_text(json.dumps({'state':'running','name':name}))
 with open(r/(name+'_eval.log'),'w') as f:
  p=subprocess.run([py,'-u',str(r/'evaluate_review.py'),'--name',name,'--delta',str(old/'learning'/name/'final.pt'),'--mirror','--end','10'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MUJOCO_EGL_DEVICE_ID='0',VLA_GPU='0'),stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:
  (r/'eval_status.json').write_text(json.dumps({'state':'failed','name':name,'returncode':p.returncode}));raise SystemExit(1)
(r/'eval_status.json').write_text(json.dumps({'state':'complete'}))
