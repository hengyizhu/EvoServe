"""Compact report: fixed-final learning table, 5 serving windows, paired E2E."""
import json,datetime,csv
from pathlib import Path
R=Path('/home/hengyi/pi05_stability_round2')
def read(p):
 try:return json.loads(Path(p).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def eval_result(name):
 p=R/'results'/f'{name}.jsonl'
 if not p.exists():return '—',[]
 rows=[]
 for line in p.read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 values=[]
 for task in [3,0]:
  a=[x for x in rows if x['task']==task];k=sum(x['success'] for x in a);values.append(f'{k*2}%' if len(a)==50 else f'{k}/{len(a)}*')
 return '/'.join(values),rows
def render():
 cfg=read(R/'protocol.json');gate=read(R/'task_gate.json');selection=read(R/'selection.json');decisions=read(R/'stability_decisions.json');queue=read(R/'queue_status.json');now=datetime.datetime.now(datetime.timezone.utc).strftime('%m-%d %H:%M UTC')
 lines=['# π0.5 第二轮：稳定学习与异步搬运（一页汇总）','',f'更新 {now}｜队列：{queue.get("state","准备中")}；运行{len(queue.get("active",[]))}，完成{queue.get("completed",0)}，待执行{queue.get("pending",0)}。',f'任务：{("曝光×"+str(gate.get("brightness"))+"；父模型"+str(gate.get("parent_successes"))+"/50") if gate.get("state")=="passed" else "自然曝光扰动待筛选（父模型须<50%）"}。历史任务不变；lr1e-5，EMA0.999，microbatch4×2。','', '**学习：固定第1000步EMA；单元格＝适应/历史成功率。**','', '| 更新方式 | 增强 | seed 08 | seed 09 | seed 10 | 稳定判定 |','|---|---|---:|---:|---:|---|']
 detail=[];evaluated_ckpts=0
 for family in ['full','lora32']:
  for variant in ['a','b']:
   cells=[]
   for seed in cfg['learning']['seeds']:
    name=f'{family}_{variant}_s{seed}';value,_=eval_result(name+'_c1000');cells.append(value)
    for step in [500,1000]:
     _,rr=eval_result(name+f'_c{step}')
     if step==1000 and len(rr)==100:evaluated_ckpts+=1
     for task in [3,0]:
      a=[x for x in rr if x['task']==task];detail.append({'family':family,'variant':variant,'seed':seed,'checkpoint':step,'task':task,'n':len(a),'successes':sum(x['success'] for x in a)})
   lines.append(f'| {family} | {"开(a)" if variant=="a" else "关(b)"} | '+' | '.join(cells)+f' | {decisions.get(family,{}).get(variant,{}).get("state","待评估")} |')
 lines+=['',f'全量与LoRA关增强仅新增1000步评估：已完成{evaluated_ckpts}/9个最终checkpoint；其余步数见[明细](learning_checkpoints.csv)。门槛：每seed提升≥25pp且≥40%，历史≥78%，适应极差≤25pp；a/b独立，以a选配置：**{selection.get("family",selection.get("state","待定"))}**。','', '**MPS30：单客户端250ms周期/时限；格内＝P99 ms / SLO% / steps·s⁻¹。**','', '| 60秒窗口 | 纯推理 | 重算前缀训练 | 缓存训练 |','|---|---|---|---|']
 for i in range(5):
  cells=[]
  for mode in ['pure','mps30_recompute','mps30_cache']:
   d=read(R/'matrix_replicates/action'/mode/f'n1_sync_r{i}'/'summary.json');cells.append(f'{d["p99_completed_ms"]:.1f}/{100*d["slo_rate"]:.1f}/{d["steps_per_s"]:.3f}' if d else '—')
  lines.append(f'| {i+1} | '+' | '.join(cells)+' |')
 lines+=['','**端到端学习：所选配置、关增强、同一B1前缀路径；格内＝适应/历史成功率。**','','| 条件 | seed 11 | seed 12 |','|---|---:|---:|']
 for mode in ['recompute','cache']:
  cells=[eval_result(f'e2e_{selection.get("family","pending")}_{mode}_s{seed}_c1000')[0] for seed in cfg['stage3']['seeds']];lines.append(f'| {mode} | '+' | '.join(cells)+' |')
 anomalies=[]
 if any(read(p).get('slo_rate',1)<1 for p in (R/'matrix_replicates/action').glob('*/*/summary.json')):anomalies.append('异步搬运修改后共跑仍有SLO违约；缓存吞吐收益不等于尾延迟已解决。')
 if gate.get('state')=='failed':anomalies.append('自然扰动筛选未达到父模型<50%，学习未启动。')
 if selection.get('state')=='no_stable_config':anomalies.append('两套配置的a均未过稳定门槛，端到端缓存学习未启动。')
 if any(read(p).get('unfinished',0)>0 for p in (R/'matrix_replicates/action').glob('*/*/summary.json')):anomalies.append('服务存在未完成请求，P99仅按已完成请求统计，SLO保留全部请求分母。')
 if queue.get('failed'):anomalies.append('失败任务：'+', '.join(queue['failed'])+'；原日志保留。')
 for family in ['full','lora32']:
  if decisions.get(family,{}).get('a',{}).get('state')=='stable' and decisions.get(family,{}).get('b',{}).get('state')=='not_stable':anomalies.append(f'{family}仅a过线，关闭增强的损伤单独保留。')
 lines+=['','**关键异常：** '+('；'.join(anomalies) if anomalies else '尚无已确认异常；“—”待执行，“*”为未完成回合，不能作最终成功率。'),'','配置与原始日志：[protocol](protocol.json) · [队列](jobs.json) · [学习](training/) · [评估](results/) · [搬运/服务](matrix_replicates/action/)']
 import cost_report;cost_report.render()
 lines+=['','参数、步时和显存：[同配置成本对比](LoRA与全量成本对比.md)。']
 tmp=R/'一页报告.tmp';tmp.write_text('\n'.join(lines));tmp.replace(R/'一页报告.md')
 with (R/'learning_checkpoints.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['family','variant','seed','checkpoint','task','n','successes']);writer.writeheader();writer.writerows(detail)
if __name__=='__main__':render()
