"""Single central scheduler remains owner of both allowed GPUs."""
import json,datetime,math
from pathlib import Path
from zoneinfo import ZoneInfo
R=Path(__file__).resolve().parent
RATES=[('5e-5',5e-5),('1e-4',1e-4),('2e-4',2e-4)]
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def save(p,d):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False));t.replace(p)
def result(name,task):
 p=R/'results'/f'{name}.jsonl'
 if not p.exists():return None
 try:a=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 except json.JSONDecodeError:return None
 a=[x for x in a if x['task']==task]
 if sorted(x['index'] for x in a)!=list(range(50)):return None
 return sum(x['success'] for x in a)
def gates(add,jobs):
 def reg(jid,argv,status,priority=3):add(jid,'benchmark',argv,[0,1],priority,{'status_path':str(status)})
 jid='lrscan_train08';reg(jid,[str(R/'train_cached.py'),'--seed','20261108','--screen','--job',jid],R/'jobs'/jid/'status.json')
 for tag,lr in RATES:
  name=f'lr{tag}_s20261108';ready=read(R/'training'/name/'checkpoint_1000.json');ename='screen_'+name
  if ready.get('state')=='ready':reg(ename,[str(R/'eval_worker.py'),'--name',ename,'--delta',ready['path'],'--brightness','.05','--task','3'],R/'results'/f'{ename}_status.json',2)
 selection=read(R/'selection.json');screen={tag:result(f'screen_lr{tag}_s20261108',3) for tag,_ in RATES}
 if not selection and all(screen[tag] is not None and jobs.get(f'screen_lr{tag}_s20261108',{}).get('state')=='complete' for tag,_ in RATES):
  tag,lr=min(RATES,key=lambda pair:(-screen[pair[0]],pair[1]));passed=screen[tag]>=29
  selection={'state':'selected','family':'lora32_heads' if passed else 'full','lr':lr if passed else 1e-5,'tag':tag if passed else 'full_1e-5','screen_successes':screen,'screen_best_tag':tag,'threshold_successes':29,'schedule':'cosine' if passed else 'constant after warmup50','reason':'best LoRA meets58%' if passed else 'no LoRA reaches58%; user-requested full fallback, augmentation OFF; prior full stability evidence was augmentation ON'};save(R/'selection.json',selection)
 O=Path('/home/hengyi/pi05_overnight_20261108');scores=[]
 if selection:
  if selection['family']=='lora32_heads':
   tag=selection['tag'];lr=selection['lr'];name=f'lr{tag}_s20261108';ename='confirm_'+name+'_history';ready=read(R/'training'/name/'checkpoint_1000.json')
   reg(ename,[str(R/'eval_worker.py'),'--name',ename,'--delta',ready['path'],'--brightness','.05','--task','0'],R/'results'/f'{ename}_status.json',2)
   for seed in [20261109,20261110]:
    name=f'lr{tag}_s{seed}';jid='lrconfirm_train'+str(seed);reg(jid,[str(R/'train_cached.py'),'--seed',str(seed),'--lr',str(lr),'--job',jid],R/'jobs'/jid/'status.json');ready=read(R/'training'/name/'checkpoint_1000.json');ename='confirm_'+name
    if ready.get('state')=='ready':reg(ename,[str(R/'eval_worker.py'),'--name',ename,'--delta',ready['path'],'--brightness','.05'],R/'results'/f'{ename}_status.json',2)
   scores=[{'seed':20261108,'adaptation':screen[tag],'history':result(f'confirm_lr{tag}_s20261108_history',0)}]+[{'seed':seed,'adaptation':result(f'confirm_lr{tag}_s{seed}',3),'history':result(f'confirm_lr{tag}_s{seed}',0)} for seed in [20261109,20261110]]
  else:
   # Seed08 full/no-augmentation final EMA already evaluated on both50-episode tasks.
   old=Path('/home/hengyi/pi05_stability_round2/results/full_b_s20261108_c1000.jsonl');rr=[json.loads(x) for x in old.read_text().splitlines()];assert all(sorted(x['index'] for x in rr if x['task']==t)==list(range(50)) for t in [0,3]);scores=[{'seed':20261108,'adaptation':sum(x['success'] for x in rr if x['task']==3),'history':sum(x['success'] for x in rr if x['task']==0),'reused':str(old)}]
   for seed in [20261109,20261110]:
    name=f'confirm_full_s{seed}';reg(name,[str(O/'train.py'),'--name',name,'--family','full','--lr','1e-5','--seed',str(seed),'--perturb',str(O/'tasks/lowlight_0.json')],O/'training'/name/'status.json');ready=read(O/'training'/name/'checkpoint_1000.json');ename=name+'_eval'
    if ready.get('state')=='ready':reg(ename,[str(R/'eval_worker.py'),'--name',ename,'--delta',ready['path'],'--perturb',str(O/'tasks/lowlight_0.json')],R/'results'/f'{ename}_status.json',2)
    scores.append({'seed':seed,'adaptation':result(ename,3),'history':result(ename,0)})
  if all(x['adaptation'] is not None and x['history'] is not None for x in scores):
   stable=all(x['adaptation']>=29 and x['history']>=39 for x in scores) and max(x['adaptation'] for x in scores)-min(x['adaptation'] for x in scores)<=12.5;save(R/'confirmation.json',{'state':'stable' if stable else 'not_stable','family':selection['family'],'scores':scores,'threshold':'adaptation>=29/50, history>=39/50, range<=25pp','seed08_reused':True})
 import importlib.util
 global overnight_hook
 if 'overnight_hook' not in globals():
  spec=importlib.util.spec_from_file_location('overnight_hooks',O/'queue_hooks.py');overnight_hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(overnight_hook)
 overnight_hook.gates(add,jobs,selection)
 render()
def render():
 selection=read(R/'selection.json');confirmation=read(R/'confirmation.json');lines=['# LoRA＋全量动作/时间投影：学习率筛选','',datetime.datetime.now(ZoneInfo('America/New_York')).strftime('%m-%d %H:%M %Z')+'｜动作专家18层Q/K/V/O与FFN gate/up/down采用rank32、alpha32 LoRA；动作输入/输出投影、time_mlp_in/out全量更新。可训练20,876,320参数。关增强，EMA0.999，microbatch4×累积2，1000步；固定父模型与低照度任务。','','| LR | seed08适应成功率（50回合） | 缓存训练1000步秒数 |','|---|---:|---:|']
 for tag,_ in RATES:
  n=result(f'screen_lr{tag}_s20261108',3);d=read(R/'training'/f'lr{tag}_s20261108'/'cost_summary.json');lines.append(f'| {tag} | '+(f'{2*n}%' if n is not None else '待测')+' | '+(f'{d["seconds"]:.1f}' if d else '待测')+' |')
 lines+=['',f'选中学习率：{selection.get("tag","待定")}；适应成功数最高者获选，同分取较小LR。','','| 确认seed | 适应成功率 | 历史成功率 |','|---|---:|---:|']
 if selection:
  for row in confirmation.get('scores',[]):lines.append(f"| {row['seed']} | {2*row['adaptation']}% | {2*row['history']}% |")
 lines+=['',f"配置：{selection.get('family','待定')}；稳定门槛判定：{confirmation.get('state','待定')}。seed08是筛选/历史复用点，seed09/10为新增确认。",'全部LoRA筛选使用50步warmup、cosine衰减到峰值10%；37个AdaRMS调制线性层也纳入LoRA，动作/时间投影全量训练。缓存生成另计。','[通宵汇总](/home/hengyi/pi05_overnight_20261108/早间汇总.md) · [模块与配置](protocol.json)']
 (R/'筛选报告.md').write_text('\n'.join(lines))
