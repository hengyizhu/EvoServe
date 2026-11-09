import json,time,os
from pathlib import Path
from interference_parallel import save
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'benchmark_parallel'
manifest=json.loads((BASE/'launch.json').read_text())
while True:
    states=[];rows=[]
    for worker in manifest['workers']:
        directory=BASE/f"gpu{worker['gpu']}_r{worker['repeat']}"
        state=json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else {'state':'starting'}
        if state['state'] not in ('complete','failed') and not Path(f"/proc/{worker['pid']}").exists():state={'state':'failed','reason':'Worker process exited without completion'}
        states.append({**worker,**state})
        if (directory/'summary.json').exists():rows.extend(json.loads((directory/'summary.json').read_text()))
    assert len({r['name'] for r in rows})==len(rows)
    save(BASE/'summary.json',rows)
    state='complete' if all(s['state']=='complete' for s in states) else 'failed' if any(s['state']=='failed' for s in states) else 'running'
    save(ROOT/'interference_status.json',{'state':state,'execution':'parallel_per_repeat','completed':len(rows),'total':54,'workers':states,'insufficient_training_trials':[r['name'] for r in rows if not r['adequate_training_steps']]})
    if state=='complete' or all(s['state'] in ('complete','failed') for s in states):break
    time.sleep(15)
