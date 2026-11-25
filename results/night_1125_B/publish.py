"""Publish completed rounds; stop this campaign at the configured New York cutoff."""
from datetime import datetime
from pathlib import Path
import subprocess,sys,time,json
from zoneinfo import ZoneInfo
here=Path(__file__).resolve().parent
root=here.parents[1]
zone=ZoneInfo('America/New_York')
config=Path.home()/'evoserve-machine-b/night_1125_b/deadline.txt'
units=['night1125-b-gpu0','night1125-b-gpu1']
def command(args,**kwargs):
    return subprocess.run(args,cwd=root,check=True,**kwargs)
while True:
    now=datetime.now(zone)
    deadline=datetime.fromisoformat(config.read_text().strip())
    active=any(subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0 for unit in units)
    if now>=deadline and active:
        subprocess.run(['systemctl','--user','stop',*units],check=True)
        (here/'cutoff.json').write_text(json.dumps(dict(stopped_at=datetime.now(zone).isoformat(),reason='morning deadline'),indent=2))
        active=False
    command([sys.executable,str(here/'refresh.py')])
    command(['git','add','-f',str(here.relative_to(root))])
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=root).returncode:
        command(['git','commit','-qm','Update overnight B evidence and completed round counts'])
    command(['git','push','origin','HEAD:night-1125-b'])
    if not active:break
    time.sleep(max(1,min(1800,(deadline-datetime.now(zone)).total_seconds())))
