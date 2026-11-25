"""Publish until both experimental queues finish."""
from pathlib import Path
import subprocess,sys,time
here=Path(__file__).resolve().parent
root=here.parents[1]
units=['night1125-b-gpu0','night1125-b-gpu1']
def command(args,**kwargs):
    return subprocess.run(args,cwd=root,check=True,**kwargs)
while True:
    active=any(subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0 for unit in units)
    command([sys.executable,str(here/'refresh.py')])
    command(['git','add','-f',str(here.relative_to(root))])
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=root).returncode:
        command(['git','commit','-qm','Update overnight B evidence and completed round counts'])
    command(['git','push','origin','HEAD:night-1125-b'])
    if not active:break
    time.sleep(1800)
