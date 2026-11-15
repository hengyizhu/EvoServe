"""Export only observed action-LoRA results, then freeze a seven-task prior."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict

p=argparse.ArgumentParser();p.add_argument('--experiments',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.experiments);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
observed=[];sources={}
def rows(rel):
 path=root/rel;sources[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
 return list(csv.DictReader(path.open()))
def add(task,seed,step,p,source,kind='observed'):
 observed.append(dict(task=task,seed=seed,step=int(step),success_rate=float(p),source=source,kind=kind))
parents={r['task']:float(r['success_rate']) for r in rows('pi05_extended_20261109/learning_curves.csv') if int(r['step'])==0}
parent={'drawer':parents['drawer_lowlight'],'book':parents['book'],'microwave':parents['microwave']}
rel='pi05_night_20261110/learning_curves.csv'
for r in rows(rel):
 name=r['task'];s=int(r['step'])
 if '_action_d100' not in name or s>4000:continue
 task=name.split('_')[1];seed=int(name.split('_')[2][1:]);add(task,seed,s,r['success_rate'],rel)
rel='pi05_loadscan_20261109/lora_book_curve.csv'
for r in rows(rel):add('book',8,r['step'],r['success_rate'],rel)
rel='pi05_adaptive_validate_20261111/parent_screening.csv'
for r in rows(rel):
 if r['passed']=='True':parent[r['group']]=int(r['successes'])/int(r['episodes'])
rel='pi05_adaptive_validate_20261111/results_summary.csv'
for r in rows(rel):
 if r['scheme']=='action' and r['task'].startswith('new_'):
  add(r['task'].split('_')[1],int(r['seed'])%100,r['total_steps'],r['success_rate'],rel)
rel='pi05_followon_20261112/B_parent_screening.csv'
for r in rows(rel):
 if float(r['success_rate'])<=.3:parent[r['key']]=float(r['success_rate'])
rel='pi05_followon_20261112/B_learning.csv'
for r in rows(rel):
 if r['scheme']=='action' and int(r['step'])<=4000:add(r['task'].removeprefix('new_'),int(r['seed'])%100,r['step'],r['success_rate'],rel)
curves=defaultdict(list)
for r in observed:curves[(r['task'],r['seed'])].append((r['step'],r['success_rate']))
for task,seed in list(curves):
 add(task,seed,0,parent[task],'parent screening summaries')
 curves[(task,seed)].append((0,parent[task]));curves[(task,seed)].sort()
assert set(parent)=={'drawer','book','microwave','X','Y','camera','texture'},parent
assert set(t for t,_ in curves)==set(parent)
def interpolate(points,x):
 for (a,u),(b,v) in zip(points,points[1:]):
  if a<=x<=b:return u+(v-u)*(x-a)/(b-a)
 return points[-1][1]
grid=[0,250,500,1000,2000,3000,4000];task_curves={};gains=[];counts={}
for task,p0 in parent.items():
 group=[v for (t,_),v in curves.items() if t==task]
 assert group and all(v[-1][0]==4000 for v in group)
 mean=[sum(interpolate(v,s) for v in group)/len(group) for s in grid]
 gain=mean[-1]-p0;assert gain>0,(task,gain)
 gains.append(gain);task_curves[task]=[(s,(v-p0)/gain) for s,v in zip(grid,mean)]
 counts[task]=[len(v) for v in group]
prior=dict(points=[(s,sum(v[i][1] for v in task_curves.values())/7) for i,s in enumerate(grid)],default_amplitude=sum(gains)/7,default_time_scale=1,task_normalized_curves=task_curves,parents=parent,observed_points_per_seed=counts,sources_sha256=sources,normalization='(p(step)-p(parent))/(mean p(4000)-p(parent)), seed mean within task, equal task weights; linear interpolation only between recorded checkpoints; no monotonic repair or clipping of measured curve',sparse_tasks=['X','Y'],sparse_note='X/Y have only parent and4000 action-only endpoints; intermediate prior values are linear interpolation, not measured learning curves.',historical_eval_denoising_steps=5,prototype_serving_denoising_steps=10,source_excludes=['full action update','visual LoRA','automatic/forced visual switch','reduced adaptation data'])
(out/'learning_prior.json').write_text(json.dumps(prior,indent=2)+'\n')
with (out/'prior_observations.csv').open('w') as f:
 w=csv.DictWriter(f,lineterminator='\n',fieldnames=list(observed[0]));w.writeheader();w.writerows(sorted(observed,key=lambda r:(r['task'],r['seed'],r['step'])))
print(json.dumps(dict(tasks=7,observations=len(observed),sparse=prior['sparse_tasks'],default_amplitude=prior['default_amplitude'])))
