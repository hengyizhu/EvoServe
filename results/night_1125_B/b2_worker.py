"""Two independent queues; no failed arm is retried."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path.home()/'EvoServe/machine_b'))
from paired_branch_and_lwd import root,suite,start,save
gpu=int(sys.argv[1])
out=root/'night_1125_b'
out.mkdir(exist_ok=True)
baseline=suite/'online/night1125_camera_s10_i1'
for task,interval in ([('camera',1),('camera',4)] if gpu==0 else [('texture',1),('camera',6)]):
    name=f'night1125_{task}_s10_i{interval}'
    dest=out/(name+'.json')
    if interval>1:
        while not (out/'night1125_camera_s10_i1.json').exists() or json.loads((out/'night1125_camera_s10_i1.json').read_text())['state']=='running':
            time.sleep(10)
        if json.loads((out/'night1125_camera_s10_i1.json').read_text())['state']!='complete':
            save(dest,dict(state='discarded',reason='paired camera baseline failed'));continue
    args=['--root',root,'--name',name,'--mode','correction','--task',task,'--variant','hg_dagger','--rank',32,'--lr',5e-5,'--rounds',12,'--seed',20261110,'--online-only','--skip-retention']
    if interval>1:args+=['--deployment-interval',interval,'--paired-baseline',baseline]
    folder=suite/'online'/name
    save(dest,dict(state='running',gpu=gpu,folder=str(folder)))
    code=start(gpu,'lwd_learning.py',args,name).wait()
    save(dest,dict(state='discarded' if code else 'complete',gpu=gpu,exit_code=code,folder=str(folder)))
