"""Exact first-k prefixes; run from this directory inside the prototype checkout."""
import csv
import json
from pathlib import Path
import sys
import warnings
import numpy as np
from scipy.stats import spearmanr

here=Path(__file__).resolve().parent
proto=here.parents[1]/'prototype'
sys.path.insert(0,str(proto))
from evoserve.core import LearningPrior
online=Path.home()/'evoserve-machine-b/work/machine_b_learning/online'
def write(name,rows):
    with (here/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
def runs():
    for seed in (8,9):
        for task in ('camera','texture','lowlight','Y'):
            name=f'pure_hg_{task}_s{seed:02d}_a1'
            if task=='texture' and seed==8:name='texture_hg_online08_a1'
            if task=='Y' and seed==8:name='Y_pure_r32_s08_a1'
            yield task,seed,1,name
        for period in (2,4,6):
            yield 'camera',seed,period,f'period{period}_hg_camera_s{seed:02d}_a1'
    yield 'texture',9,4,'period4_hg_texture_s09_a1'
cfg=json.loads((proto/'configs/learning_prior.json').read_text())
prior=LearningPrior(cfg['points'],cfg['default_amplitude'])
raw=[];pred=[]
for task,seed,interval,run in runs():
    rows=list(map(json.loads,(online/run/'rounds.jsonl').read_text().splitlines()))
    assert len(rows)==12
    for r in rows:
        raw.append(dict(run=run,task=task,seed=seed,interval=interval,round=r['round'],step=r['model_step'],autonomous=r['unassisted_successes'],episodes=r['rollout_episodes'],assisted=r['rollout_successes'],takeovers=r['takeovers'],takeover_frames=r['takeover_frames'],control_frames=r['control_frames']))
    for k in (1,2,3):
        past=rows[:k]; last=past[-1]; nxt=rows[k]
        y=last['unassisted_completion_rate']; actual=nxt['unassisted_completion_rate']
        history=[(b['model_step']-a['model_step'],b['unassisted_completion_rate']-a['unassisted_completion_rate']) for a,b in zip(past,past[1:])]
        delta=nxt['model_step']-last['model_step']
        assert delta>=0
        proposal=prior.predict(y,delta,history)['success']
        for model,val in [('prior',proposal),('last',y),('mean',np.mean([r['unassisted_completion_rate'] for r in past]))]:
            pred.append(dict(run=run,task=task,seed=seed,interval=interval,k=k,model=model,delta_steps=delta,last=y,actual=actual,predicted=val,abs_error_pp=100*abs(val-actual),gain=val-y,actual_gain=actual-y))
write('b1_raw_rounds.csv',raw); write('b1_predictions.csv',pred)
mae=[]
for task in ['ALL','camera','texture','lowlight','Y']:
 for k in (0,1,2,3):
    group=[r for r in pred if (task=='ALL' or r['task']==task) and (not k or r['k']==k)]
    mae.append(dict(task=task,k=k or 'all',n=len(group)//3,**{m:np.mean([r['abs_error_pp'] for r in group if r['model']==m]) for m in ('prior','last','mean')}))
write('b1_mae.csv',mae)
rank=[]
for seed in (8,9):
 for k in (1,2,3):
    for m in ('prior','last','mean'):
        g=[r for r in pred if r['seed']==seed and r['k']==k and r['model']==m and r['interval']==1]
        assert len(g)==4
        xx=[r['gain'] for r in g]; yy=[r['actual_gain'] for r in g]
        rho=spearmanr(xx,yy).statistic if len(set(xx))>1 and len(set(yy))>1 else None
        rank.append(dict(seed=seed,k=k,model=m,tasks=4,spearman=rho,actual_tied=len(set(yy))==1,predicted_tied=len(set(xx))==1))
write('b1_rank.csv',rank)
# Secondary rolling-origin diagnostic. Keep the first-k test above independent.
rolling=[]
for task,seed,interval,run in runs():
    data=[v for v in raw if v['run']==run]
    for k in (1,2,3):
        for target in range(k,len(data)):
            past=data[target-k:target]; last=past[-1]; nxt=data[target]
            rate=lambda v:v['autonomous']/v['episodes']
            hist=[(b['step']-a['step'],rate(b)-rate(a)) for a,b in zip(past,past[1:])]
            for model,val in [('prior',prior.predict(rate(last),nxt['step']-last['step'],hist)['success']),('last',rate(last)),('mean',np.mean([rate(r) for r in past]))]:
                rolling.append(dict(run=run,task=task,seed=seed,interval=interval,k=k,target_round=target+1,model=model,
                                predicted=val,actual=rate(nxt),gain=val-rate(last),actual_gain=rate(nxt)-rate(last),
                                abs_error_pp=100*abs(val-rate(nxt))))
write('b1_rolling.csv',rolling)
rollmetrics=[]
for task in ['ALL','camera','texture','lowlight','Y']:
    g=[r for r in rolling if r['task']==task or task=='ALL']
    rollmetrics.append(dict(task=task,targets=len(g)//3,**{m:np.mean([r['abs_error_pp'] for r in g if r['model']==m]) for m in ('prior','last','mean')}))
write('b1_rolling_mae.csv',rollmetrics)
rollrank=[]
for seed in (8,9):
 for k in (1,2,3):
    for rnd in range(k+1,13):
        for m in ('prior','last','mean'):
            g=[v for v in rolling if (v['interval'],v['seed'],v['k'],v['target_round'],v['model'])==(1,seed,k,rnd,m)]
            xx=[v['gain'] for v in g];yy=[v['actual_gain'] for v in g]
            assert len(g)==4
            rho=spearmanr(xx,yy).statistic if len(set(xx))>1 and len(set(yy))>1 else None
            rollrank.append(dict(seed=seed,k=k,target=rnd,model=m,spearman=rho))
write('b1_rolling_rank.csv',rollrank)
