import csv,json,math,datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from queue_hooks import scores,read,R,L,OLD

def ci(k):
 p=k/50;z=1.95996398454;d=1+z*z/50;c=(p+z*z/100)/d;h=z*math.sqrt(p*(1-p)/50+z*z/10000)/d
 return f'{2*k}% [{100*(c-h):.0f}–{100*(c+h):.0f}]'
def main():
 decisions=read(R/'decisions.json');lines=['# π0.5：2000步三seed对照与4000步学习曲线','',datetime.datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M %Z'),'','配置：关增强，服务路径缓存训练，microbatch4×累积2，EMA0.999；全量动作侧lr1e-5，50步warmup后恒定；LoRA rank32、lr5e-5，50步warmup后cosine至2000步的5e-6。LoRA覆盖动作专家q/k/v/o、gate/up/down和AdaRMS；动作投影与时间MLP全量更新。','', '| 配置 | seed | 2000步适应成功率 [95% CI] | 历史成功率 [95% CI] |','|---|---|---|---|']
 for family,label in [('full','全量'),('lora32_heads','LoRA')]:
  ds=decisions.get(family,{})
  for seed in [20261108,20261109,20261110]:
   row=next((x for x in ds.get('scores',[]) if x['seed']==seed),None)
   lines.append(f'| {label} | {seed%100:02d} | '+(f'{ci(row["adaptation"])} | {ci(row["history"])}' if row else '待评估 | 待评估')+' |')
 lines+=['', '全量判定：'+decisions.get('full',{}).get('state','pending')+'；LoRA判定：'+decisions.get('lora32_heads',{}).get('state','pending')+'。', '']
 lines+=['固定EMA第2000步，每任务50回合；适应≥58%、历史≥78%，跨seed适应极差≤25个百分点。全量seed08复用此前2000步训练与适应评估，补测历史任务；其余按各自seed采样序列。','']
 points=[0,100,250,500,1000,2000,3000,4000];data=[];fig,axes=plt.subplots(1,2,figsize=(11,3.8));labels={'drawer_lowlight':'Drawer / low exposure','book':'Book / selected perturbation','microwave':'Microwave / selected perturbation','drawer_shift':'Drawer / shift (appendix)'}
 lines+=['| 曲线 | 父模型 | 100 | 250 | 500 | 1000 | 2000 | 3000 | 4000 |','|---|---|---|---|---|---|---|---|---|']
 for task in labels:
  values={};tid={'drawer_lowlight':3,'drawer_shift':3,'book':5,'microwave':9}[task]
  if task in ['drawer_lowlight','drawer_shift']:
   oldtask='lowlight' if task=='drawer_lowlight' else 'camera_shift';values[0]=read(OLD/f'{oldtask}_gate.json')['successes']
   for step in points[1:6]:
    k=scores(f'curve_{oldtask}_c{step}',L).get(tid)
    if k is not None:values[step]=k
  else:
   gate=read(R/f'{task}_gate.json')
   if gate.get('state')=='passed':values[0]=gate['successes']
  for step in points[1:]:
   k=scores(f'extended_curve_{task}_c{step}').get(tid)
   if k is not None:values[step]=k
  lines.append('| '+{'drawer_lowlight':'抽屉低照度','drawer_shift':'附录：抽屉平移','book':'放书入收纳盒','microwave':'杯子放进微波炉'}[task]+' | '+' | '.join(f'{values[x]*2}%' if x in values else '待测' for x in points)+' |')
  if values:
   axis=axes[1] if task=='drawer_shift' else axes[0];xs=sorted(values);axis.plot(xs,[values[x]*2 for x in xs],'o-',label=labels[task])
   for step in xs:data.append({'task':task,'task_id':tid,'step':step,'successes':values[step],'episodes':50,'success_rate':values[step]/50})
 for ax,title in zip(axes,['Main: three distinct LIBERO-10 tasks','Appendix: original drawer shift']):
  ax.set(xlabel='Training steps',ylabel='Success rate (%)',ylim=(0,100),xlim=(0,4050),title=title);ax.grid(alpha=.2)
  if ax.lines:ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(R/'learning_curves.png',dpi=180);fig.savefig(R/'learning_curves.svg');plt.close(fig)
 with (R/'learning_curves.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['task','task_id','step','successes','episodes','success_rate']);w.writeheader();w.writerows(data)
 lines+=['','[学习曲线图](learning_curves.png) · [曲线CSV](learning_curves.csv)','', '曲线沿用全量动作侧lr1e-5、seed08，不根据新对照结果事后挑配置；新任务父模型须<50%方可训练。原两条曲线重放前2000步恢复optimizer轨迹，EMA逐位一致才接续3000/4000步。']
 for task in ['book','microwave']:
  gate=read(R/f'{task}_gate.json')
  if gate.get('state')=='failed':lines+=['',f'异常：{task}的预设候选均未通过父模型<50%门槛，未训练。']
 for p in (R/'training').glob('*/extension_equivalence.json'):
  d=read(p)
  if d.get('state')!='passed':lines+=['',f'异常：{p.parent.name}与原2000步EMA不一致，已停止，未拼接曲线。']
 lines+=['','[训练记录与checkpoint](training/) · [评估逐回合日志](results/) · [协议](protocol.json)'];(R/'汇总报告.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
