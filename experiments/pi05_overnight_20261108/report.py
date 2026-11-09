import json,datetime,math,csv
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;L=Path('/home/hengyi/pi05_lora_lr_20261108')
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def result(name):
 p=L/'results'/f'{name}.jsonl'
 if not p.exists():return None
 try:a=[json.loads(x) for x in p.read_text().splitlines()]
 except json.JSONDecodeError:return None
 if sorted(x['index'] for x in a)!=list(range(50)):return None
 return sum(x['success'] for x in a)
def main():
 s=read(L/'selection.json');c=read(L/'confirmation.json');lines=['# 通宵实验汇总','',datetime.datetime.now(ZoneInfo('America/New_York')).strftime('%m-%d %H:%M %Z'),'',f'配置：{s.get("family","筛选中")}；学习率 {s.get("lr","待定")}；三seed门槛判定：{c.get("state","待确认")}。LoRA筛选采用50步warmup＋cosine至10%峰值；动作/时间投影全量训练，所有动作专家线性层（含AdaRMS）加rank32 LoRA。',s.get('reason',''),'','| seed | 适应成功率 | 历史成功率 |','|---|---:|---:|']
 for x in c.get('scores',[]):lines.append(f'| {x["seed"]} | {2*x["adaptation"]}% | {2*x["history"]}% |')
 lines+=['','**限速与SLO：每格为P99 ms / SLO% / 实际steps/s，5×60秒。**','','| 限速 | 窗口1 | 窗口2 | 窗口3 | 窗口4 | 窗口5 |','|---|---|---|---|---|---|'];invalid=[]
 for cap in [-1,3,2,1]:
  cells=[]
  for i in range(1,6):
   d=read(R/'tail'/f'cap{cap}_window{i}'/'summary.json')
   if not d:cells.append('待测');continue
   p99=f'{d["p99_completed_ms"]:.1f}' if d['p99_completed_ms'] is not None else '无完成';suffix='' if d.get('other_cards_idle') else '*';cells.append(f'{p99}/{100*d["slo_rate"]:.2f}/{d["training"]["steps_per_s"]:.3f}{suffix}')
   if suffix:invalid.append(f'{cap}/{i}')
  lines.append('| '+('不限速' if cap==-1 else str(cap)+'/s')+' | '+' | '.join(cells)+' |')
 if not (R/'tail/before.json').exists():lines+=['','尾延迟尚未开始：只在GPU2、3空闲时准入；如一直被占用，此项如实保留为待测。']
 if invalid:lines+=['','* 这些窗口监测到GPU2/3占用，不能作为隔离测量：'+', '.join(invalid)]
 if read(R/'colorcast_gate.json').get('state')=='failed':lines+=['','关键异常：通道偏色的三个候选均未达到父模型成功率<50%的条件，第三条曲线尚缺合格任务；不以不合格候选替代。']
 data=[];fig,ax=plt.subplots(figsize=(7.2,4.0))
 for task,label in [('lowlight','Low exposure'),('camera_shift','Image-view shift'),('colorcast','Color cast')]:
  g=read(R/f'{task}_gate.json');points=[]
  if g.get('state')=='passed':points.append((0,g['successes']))
  for step in [100,250,500,1000,2000]:
   k=result(f'curve_{task}_c{step}')
   if k is not None:points.append((step,k))
  for step,k in points:data.append({'task':task,'step':step,'successes':k,'episodes':50,'success_rate':k/50,'perturb':g.get('perturb')})
  if points:
   x=[x[0] for x in points];y=[100*x[1]/50 for x in points];ax.plot(x,y,'o-',label=label)
 ax.set(xlabel='Training steps (EMA checkpoints)',ylabel='Success rate (%)',ylim=(0,100),xlim=(-30,2050));ax.set_xticks([0,100,250,500,1000,2000]);ax.grid(alpha=.2)
 if data:ax.legend()
 fig.tight_layout();fig.savefig(R/'learning_curves.png',dpi=180);fig.savefig(R/'learning_curves.svg');plt.close(fig)
 with (R/'learning_curves.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['task','step','successes','episodes','success_rate','perturb']);w.writeheader();w.writerows(data)
 lines+=['','![成功率与训练步数](learning_curves.png)','三条曲线是同一LIBERO抽屉任务的不同适应条件：低曝光、合成图像视野平移、通道偏色；每个新条件须父模型<50%才训练。每点50回合，单seed；未完成的点不补值。','', '[学习率筛选](../pi05_lora_lr_20261108/筛选报告.md) · [曲线原始数据](learning_curves.csv) · [GPU监测](tail/gpu_monitor.jsonl) · [协议](protocol.json)'];(R/'早间汇总.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
