import time,json,subprocess
from pathlib import Path
r=Path('/home/hengyi/pi05_review_20261107')
while True:
 p=r/'matrix_replicates/action/status.json'
 if p.exists() and json.loads(p.read_text()).get('state')=='finished':break
 out=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=timestamp,index,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
 with open(r/'gpu3_telemetry.csv','a') as f:f.write(out)
 time.sleep(1)
