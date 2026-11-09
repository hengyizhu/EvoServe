from pathlib import Path
import json,random,shutil
R=Path(__file__).resolve().parent;OLD=Path('/home/hengyi/pi05_lora_lr_20261108');BASE=Path('/home/hengyi/vla_followup_20261020');orig=json.loads((BASE/'data_split.json').read_text());meta=Path(orig['data_root'])/'meta';episodes=[json.loads(x) for x in (meta/'episodes.jsonl').read_text().splitlines()];tasks={x['task']:x['task_index'] for x in map(json.loads,(meta/'tasks.jsonl').read_text().splitlines())}
configs=[('drawer',3,orig['target']),('book',5,'pick up the book and place it in the back compartment of the caddy'),('microwave',9,'put the yellow and white mug in the microwave and close it')]
for name,tid,prompt in configs:
 split=dict(orig)
 if name!='drawer':
  pool=[x for x in episodes if prompt in x['tasks']];random.Random(20261020).shuffle(pool);assert len(pool)>=25
  split.update(target=prompt,target_train=pool[:20],target_validation=pool[20:25])
 split.update(eval_task_id=tid,target_data_index=tasks[prompt],history_data_index=tasks[orig['history']])
 for ep in split['target_train']:assert (Path(split['data_root'])/f"data/chunk-{ep['episode_index']//1000:03d}/episode_{ep['episode_index']:06d}.parquet").exists()
 (R/'splits'/f'{name}.json').write_text(json.dumps(split,indent=2))
 for seed in [20261108,20261109,20261110] if name=='drawer' else [20261108]:
  rng=random.Random(seed);current=[(e['episode_index'],i) for e in split['target_train'] for i in range(e['length']-9)];replay=[(e['episode'],e['start']) for e in split['replay_windows']];seq=[]
  for _ in range(4000):
   batch=[rng.choice(current) for _ in range(4)]+[rng.choice(replay) for _ in range(4)];rng.shuffle(batch);seq.append(batch)
  seq=json.loads(json.dumps(seq))
  for steps in [2000,4000]:(R/'sequences'/f'{name}_{seed}_{steps}.json').write_text(json.dumps(seq[:steps]))
  if name=='drawer':assert seq[:1000]==json.loads((OLD/'sequences'/f'{seed}.json').read_text())
  if name=='drawer' and seed==20261108:assert seq[:2000]==json.loads(Path('/home/hengyi/pi05_overnight_20261108/sequence_2000_s08.json').read_text())
for name,pcfgs in [('drawer_lowlight',[{'kind':'exposure','factor':.05}]),('drawer_shift',[{'kind':'camera_shift','fraction':.25}]),('book',[{'kind':'camera_shift','fraction':.25},{'kind':'camera_shift','fraction':.375},{'kind':'exposure','factor':.05}]),('microwave',[{'kind':'exposure','factor':.05},{'kind':'exposure','factor':.025},{'kind':'camera_shift','fraction':.25}])]:
 for i,cfg in enumerate(pcfgs):
  cfg['target_prompt']=dict((n,p) for n,t,p in configs)['drawer' if name.startswith('drawer') else name]
  (R/'tasks'/f'{name}_{i}.json').write_text(json.dumps(cfg,indent=2))
protocol={'seeds':[20261108,20261109,20261110],'families':{'full':{'lr':1e-5,'schedule':'50-step warmup then constant'},'lora32_heads':{'lr':5e-5,'schedule':'50-step warmup then cosine to10% at2000'}},'comparison_steps':2000,'curves_steps':4000,'curve_checkpoints':[100,250,500,1000,2000,3000,4000],'augmentation':False,'EMA':.999,'microbatch':4,'accumulation':2,'threshold':{'adaptation_successes':29,'history_successes':39,'seed_range_pp':25},'parent_gate':'successes<25/50','main_curves':['drawer_lowlight','book','microwave'],'appendix_curve':'drawer_shift','training_storage':'/var/tmp/hengyi_pi05_extended_20261109/training','cache':'pinned host service path, asynchronous H2D','resource_policy':'at most one cached trainer; remaining free GPUs evaluate; never preempt foreign work','extension':'replay original first2000 to recover optimizer/EMA trajectory; require bitwise EMA equivalence before step2001','curve_family':'full','curve_lr':1e-5}
(R/'protocol.json').write_text(json.dumps(protocol,indent=2));print('prepared splits and sequences')
for f in ['common.py','cache_engine.py','perturbations.py','eval_worker.py']:shutil.copy2(OLD/f,R/f)
p=R/'common.py';p.write_text(p.read_text().replace(str(OLD),str(R)))
p=R/'perturbations.py';s=p.read_text().replace("target=json.loads(Path('/home/hengyi/vla_followup_20261020/data_split.json').read_text())['target']", "target=cfg.get('target_prompt') or json.loads(Path('/home/hengyi/vla_followup_20261020/data_split.json').read_text())['target']");p.write_text(s)
p=R/'cache_engine.py';s=p.read_text().replace("ap.add_argument('--perturb');args=", "ap.add_argument('--perturb');ap.add_argument('--split',required=True);args=")
s=s.replace("split=json.loads(Path('/home/hengyi/vla_followup_20261020/data_split.json').read_text());tables={}","split=json.loads(Path(args.split).read_text());tables={}")
s=s.replace("assert row['task_index'] in [5,8]","assert row['task_index'] in [split['history_data_index'],split['target_data_index']]")
s=s.replace("if row['task_index']==8 else split['history']","if row['task_index']==split['target_data_index'] else split['history']")
s=s.replace("original_samples=Samples(self.model);checks=[]", "from task_samples import TaskSamples\n  original_samples=TaskSamples(self.model,args.split);checks=[]");p.write_text(s)
p=R/'eval_worker.py';s=p.read_text().replace(str(OLD),str(R));s=s.replace('parser.add_argument("--perturb")','parser.add_argument("--perturb")\n    parser.add_argument("--adapt-task", type=int, default=3)')
s=s.replace('([args.task] if args.task is not None else [3,0])','([args.task] if args.task is not None else [args.adapt_task,0])').replace('else [3, 0])','else [args.adapt_task, 0])')
s=s.replace('if task_id == 3 and', 'if task_id == args.adapt_task and').replace('"perturb":args.perturb}', '"perturb":args.perturb,"adapt_task":args.adapt_task}')
p.write_text(s)
