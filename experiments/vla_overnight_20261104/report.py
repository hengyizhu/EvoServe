"""Aggregate only measured results; regenerate a one-page report and standalone figures."""
from pathlib import Path
import json,csv,collections,math
ROOT=Path(__file__).resolve().parent

def readrows(path):
 with path.open() as f:return list(csv.DictReader(f))
def csvout(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def generate(final=False):
 summaries=[];requests=[];segments=[]
 for p in ROOT.glob('gpu[012]_*/**/runs/*/summary.json'):
  summaries.append(json.loads(p.read_text()))
  requests.extend(readrows(p.parent/'requests.csv'));segments.extend(readrows(p.parent/'train_segments.csv'))
 if summaries:csvout(ROOT/'summary.csv',summaries)
 if requests:csvout(ROOT/'requests.csv',requests)
 if segments:csvout(ROOT/'train_segments.csv',segments)
 phasepath=ROOT/'gpu3_phases/phase_pairs.csv';phases=readrows(phasepath) if phasepath.exists() else []
 if phases:csvout(ROOT/'phase_pairs.csv',phases)
 checks={str(p.relative_to(ROOT)):json.loads(p.read_text()) for pattern in ['gpu*/**/*correctness.json','gpu*/**/correctness.json'] for p in ROOT.glob(pattern) if 'preliminary' not in str(p)}
 checks['request_accounting']={'generated':sum(r['n_requests'] for r in summaries),'recorded':len(requests),'unfinished':sum(r['unfinished'] for r in summaries),'equal_counts':sum(r['n_requests'] for r in summaries)==len(requests)}
 (ROOT/'correctness.json').write_text(json.dumps(checks,indent=2))
 states={p.parent.name:json.loads(p.read_text()) for p in ROOT.glob('gpu*/status.json') if 'preliminary' not in str(p)}
 text=['# VLA overnight 实测报告', '', '范围：固定 π0.5 的 GPU 机制回放；不代表机器人闭环成功率或在线学习系统。', '',f'状态：主回放已完成 {len(summaries)}/195 组；阶段配对已记录 {len(phases)}/800 次。', '']
 for name,state in states.items():text.append(f'- {name}：{state.get("state")}，已完成 {state.get("completed", "尚在预检/校准")}。')
 text+=['','## 250 ms 原配置检查']
 for p in ROOT.glob('gpu[012]_*/real_250ms.json'):
  r=json.loads(p.read_text());text.append(f'- {p.parent.name}：{r["n_requests"]} 请求，违约率 {100*r["deadline_miss_rate"]:.1f}%，P95 {r["p95_ms"]:.1f} ms；'+('本次样本满足期限。' if r['deadline_miss_rate']==0 else '原配置不具备本次测试要求的实时可行性。'))
 text+=['','## 同卡配对主结果','数值为同卡、同负载、同相位三轮范围；未满三轮的组注明 n。S0 不可行的条件保留，不能将其固有违约归因于训练。','', '| 卡/训练 | 负载/相位 | 方法 | 轮数 | miss % | P95 ms | 完整更新/s |','|---|---|---|---:|---:|---:|---:|']
 groups=collections.defaultdict(list)
 for r in summaries:groups[(r['gpu_uuid'],r['profile'],r['load'],r['phase_pattern'],r['policy'])].append(r)
 uuids={}
 for line in json.loads((ROOT/'manifest.json').read_text())['gpus'].splitlines():
  bits=[x.strip() for x in line.split(',')];uuids[bits[1]]=bits[0]
 for (gpu,tier,load,phase,policy),rs in sorted(groups.items()):
  def span(key,scale=1):
   x=[r[key]*scale for r in rs if r[key] is not None]
   return f'{min(x):.2f}–{max(x):.2f}' if x else 'NA'
  text.append(f'| GPU {uuids.get(gpu,gpu)}/{tier} | {load}/{phase} | {policy} | {len(rs)} | {span("deadline_miss_rate",100)} | {span("p95_ms")} | {span("completed_updates",1/180)} |')
 # Keep the requested one-page report compact; full cell results remain in summary.csv.
 table_start=text.index('| 卡/训练 | 负载/相位 | 方法 | 轮数 | miss % | P95 ms | 完整更新/s |')
 (ROOT/'DETAILS.md').write_text('\n'.join(text)+'\n')
 compact=text[:table_start]+['完整同卡配对表见 [DETAILS.md](DETAILS.md)，原始统计见 [summary.csv](summary.csv)。']
 if summaries:
  compact+=['','已完成条件按方法汇总（跨条件范围，仅用于概览，不代表配对效果）：']
  for policy in ['S0','S1','S2','S3','S4']:
   rs=[r for r in summaries if r['policy']==policy]
   if rs:compact.append(f'- {policy}：{len(rs)} 组，miss {min(r["deadline_miss_rate"] for r in rs)*100:.1f}–{max(r["deadline_miss_rate"] for r in rs)*100:.1f}%，有效训练 {min(r["effective_samples_per_s"] for r in rs):.2f}–{max(r["effective_samples_per_s"] for r in rs):.2f} 样本/s。')
 if phases:
  compact+=['','## 阶段配对']
  pg=collections.defaultdict(list)
  for r in phases:pg[(r['profile'],r['inference_stage'],r['training_stage'])].append(r)
  for k,rs in sorted(pg.items()):
   import statistics
   compact.append(f'- {"/".join(k)}，n={len(rs)}：推理减速中位数 {statistics.median(float(r["inference_slowdown"]) for r in rs):.3f}×，联合效率中位数 {statistics.median(float(r["pair_gain"]) for r in rs):.3f}×。准备阶段成本单列，不能解释为完整迭代加速。')
 compact+=['','## 限制与未完成项','- L2 为动作侧更新；L3 为视觉+动作更新，语言权重仍冻结。均不是 LoRA。','- 固定 CPU 缓存循环复用原序列前 16 个有效 batch；噪声、时间与增强使用独立固定 RNG，未修改 loss/精度/有效 batch。','- Nsight Systems 未在环境中找到；仅尝试独立 PyTorch/CUPTI 代表性时间线，正式计时不启用 profiler。未解析 GPU kernel 时间线前，不声称已证实实际 kernel 重叠。','- 期限来自预先冻结的 serving-only 校准；周期信息为受控输入，不是真实动作队列预测器。','- 四卡并行仍共享主机资源。无收敛训练、候选发布、新成功率实验或跨系统性能胜出结论。']
 for name,s in states.items():
  if s.get('state')=='failed':compact.append(f'- 失败：{name}，详见 `{name}/status.json`。')
 (ROOT/'REPORT.md').write_text('\n'.join(compact)+'\n')
 if final:
  from analyze_traces import analyze
  trace_results=analyze()
  if trace_results:
   with (ROOT/'REPORT.md').open('a') as f:
    f.write('\n代表性 GPU kernel 时间线：\n')
    for name,value in trace_results.items():f.write(f'- {name}: 实际跨 stream kernel 时间交叠={value["observed_kernel_overlap"]}；详细时间见 trace_overlap.json。\n')
  plots(summaries,phases,requests,segments)

def plots(summaries,phases,requests,segments):
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
 labels=[];costs=[]
 for p in sorted(ROOT.glob('gpu[012]_*/calibration.json')):
  c=json.loads(p.read_text())
  for k,v in c['costs'].items():labels.append(f'{p.parent.name}/{k}');costs.append(v['mean']*1000)
 axes[0].barh(labels,costs);axes[0].set_xlabel('Mean legal segment wall time (ms)');axes[0].tick_params(axis='y',labelsize=7)
 for ax,key,title in [(axes[1],'inference_slowdown','Inference slowdown (lower is better)'),(axes[2],'pair_gain','Joint efficiency (higher is better)')]:
  matrix=np.full((4,2),np.nan)
  for i,(tier,ts) in enumerate([(t,s) for t in ['L2','L3'] for s in ['T_FWD','T_BWD']]):
   for j,ins in enumerate(['I_PREFIX','I_ACTION']):
    vals=[float(r[key]) for r in phases if (r['profile'],r['training_stage'],r['inference_stage'])==(tier,ts,ins)]
    if vals:matrix[i,j]=np.median(vals)
  im=ax.imshow(matrix,aspect='auto',cmap='viridis');ax.set_xticks([0,1],['I_PREFIX','I_ACTION']);ax.set_yticks(range(4),['L2/FWD','L2/BWD','L3/FWD','L3/BWD']);ax.set_title(title,fontsize=10)
  for i in range(4):
   for j in range(2):
    if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white')
  fig.colorbar(im,ax=ax)
 fig.savefig(ROOT/'01_phase_costs_and_interference.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(3,4,figsize=(16,10),layout='constrained')
 colors=dict(zip(['S0','S1','S2','S3','S4'],['black','tab:red','tab:orange','tab:blue','tab:green']))
 for i,gpu in enumerate([0,1,2]):
  for j,(load,phase) in enumerate([(x,y) for x in [.4,.7] for y in ['sync','staggered']]):
   ax=axes[i,j];rs=[r for r in summaries if f'_gpu{gpu}_' in r['run_id'] and 'extra' not in r['run_id'] and r['load']==load and r['phase_pattern']==phase]
   for policy,color in colors.items():
    rows=[r for r in rs if r['policy']==policy]
    ax.scatter([r['effective_samples_per_s'] for r in rows],[100*r['deadline_miss_rate'] for r in rows],c=color,label=policy,s=22)
   baselines=[100*r['deadline_miss_rate'] for r in rs if r['policy']=='S0']
   if baselines:ax.axhspan(min(baselines),max(baselines)+.0001,color='gray',alpha=.2)
   ax.set_title(f'GPU {gpu}, load={load}, {phase}');ax.set_xlabel('Effective training samples/s');ax.set_ylabel('Deadline miss (%)');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=7);fig.savefig(ROOT/'02_deadline_training_tradeoff.png',dpi=160);plt.close(fig)
 # Timeline uses measured CPU-submit/GPU-completion envelopes. Stage event durations are annotated, not claimed to be absolute CUDA clock alignment.
 selected=next((r for r in summaries if r['policy']=='S4' and r['profile']=='L3' and r['phase_pattern']=='staggered'),None)
 if selected:
  origin=selected['origin_ns'];run_id=selected['run_id'];rs=[r for r in requests if r['run_id']==run_id and (int(r['nominal_release_ns'])-origin)/1e9<5];ss=[r for r in segments if r['run_id']==run_id and (int(r['cpu_submit_ns'])-origin)/1e9<5]
  fig,ax=plt.subplots(figsize=(14,4),layout='constrained')
  for r in rs:
   release=(int(r['nominal_release_ns'])-origin)/1e9;deadline=(int(r['deadline_ns'])-origin)/1e9
   ax.plot([release],[2],marker='^',c='green');ax.plot([deadline],[2],marker='x',c='red')
   if r.get('finish_ns'):
    a=(int(r['start_ns'])-origin)/1e9;b=(int(r['finish_ns'])-origin)/1e9
    ax.broken_barh([(a,b-a)],(1.5,.3),facecolors='tab:blue')
    ax.text(a,1.86,f'P/A: {float(r["I_PREFIX_ms"]):.0f}/{float(r["I_ACTION_ms"]):.0f} ms',fontsize=6,rotation=30)
  palette={'T_FWD':'tab:orange','T_BWD':'tab:red','T_OPT':'tab:purple'}
  for r in ss:
   a=(int(r['cpu_submit_ns'])-origin)/1e9;b=(int(r['gpu_done_ns'])-origin)/1e9
   ax.broken_barh([(a,b-a)],(.5,.3),facecolors=palette.get(r['segment_kind'],'gray'));ax.text(a,.3,r['segment_kind'],fontsize=6,rotation=45)
  ax.set_yticks([.65,1.65,2],['Training','Inference envelope','Release / deadline']);ax.set_xlim(0,5);ax.set_xlabel('Seconds from measurement origin');ax.set_title('Representative S4 timeline; P/A = CUDA-event prefix/action durations (not absolute kernel positions)')
  fig.savefig(ROOT/'03_representative_timeline.png',dpi=160);plt.close(fig)

if __name__=='__main__':generate(final=True)
