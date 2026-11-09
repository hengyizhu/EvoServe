import json,math,time,datetime,statistics
from pathlib import Path
R=Path('/home/hengyi/pi05_review_20261107');OLD=Path('/home/hengyi/pi05_cache_20261107')
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def rows(p):
 if not p.exists():return []
 result=[]
 for line in p.read_text().splitlines():
  try:result.append(json.loads(line))
  except json.JSONDecodeError:pass
 return result
def quant(v,q):
 v=sorted(v);pos=(len(v)-1)*q;a=math.floor(pos);b=math.ceil(pos);return v[a]*(b-pos)+v[b]*(pos-a) if a!=b else v[a]
def summarize():
 replay={}
 for name in ['a_seed20261108','a_seed20261107']:
  ref={(x['task'],x['index']):x for x in rows(OLD/'results'/f'{name}.jsonl')};rr=rows(R/'results'/f'{name}.jsonl')
  if name=='a_seed20261107' and (R/'results/a_seed20261107_task0.jsonl').exists() and (R/'results/a_seed20261107_task3.jsonl').exists():rr=rows(R/'results/a_seed20261107_task3.jsonl')+rows(R/'results/a_seed20261107_task0.jsonl')
  replay[name]={'n':len(rr),'matching_success':sum(x['success']==ref[(x['task'],x['index'])]['success'] for x in rr),'matching_policy_steps':sum(x['policy_steps']==ref[(x['task'],x['index'])]['policy_steps'] for x in rr),'state_hash_manifest_matches':None}
  a=read(R/'results'/f'{name}_eval_manifest.json');b=read(OLD/'results'/f'{name}_eval_manifest.json')
  if a and b:
   bm={(x['task'],x['index']):x['state_sha256'] for x in b};replay[name]['state_hash_manifest_matches']=all(x['state_sha256']==bm[(x['task'],x['index'])] for x in a)
 systems=[]
 for root,tag in [(OLD,'原始'),(R,'CPU隔离复核')]:
  for mode in ['pure','mps30_recompute','mps30_cache']:
   summaries=[read(p) for p in sorted((root/'matrix_replicates/action'/mode).glob('*/summary.json'))]
   if not summaries:continue
   requests=[x for p in sorted((root/'matrix_replicates/action'/mode).glob('*/summary.json')) for x in rows(p.parent/'requests.jsonl')];complete=[x for x in requests if x.get('finish_ns') is not None];lat=[(x['finish_ns']-x['nominal_release_ns'])/1e6 for x in complete]
   outside=[x['service_ms']-x['I_PREFIX_ms']-x['I_ACTION_ms'] for x in complete]
   systems.append({'source':tag,'mode':mode,'windows':len(summaries),'requests':len(requests),'completed':len(complete),'pooled_p99_completed_ms':quant(lat,.99),'p99_window_range_ms':[min(x['p99_completed_ms'] for x in summaries),max(x['p99_completed_ms'] for x in summaries)],'late':sum(x['finish_ns']>x['deadline_ns'] for x in complete),'slo_rate':sum(x['finish_ns']<=x['deadline_ns'] for x in complete)/len(requests),'mean_steps_per_s':statistics.mean(x['steps_per_s'] for x in summaries),'outside_graph_median_ms':statistics.median(outside),'outside_graph_p99_ms':quant(outside,.99)})
 result={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'replay':replay,'systems':systems,'numerics':read(R/'numerical_checks.json'),'precision_isolation':read(R/'precision_isolation.json'),'status':{k:read(R/p) for k,p in [('eval','eval_status.json'),('numerical','numerical_status.json'),('openloop','openloop_status.json'),('serving','matrix_replicates/action/status.json')]}}
 
 if (R/'input_transfer_analysis.json').exists():result['additional_input_probe']=read(R/'input_transfer_analysis.json');result['additional_input_probe_root']='/home/hengyi/pi05_review_input_20261107/matrix_replicates/action/mps30_cache/n1_sync_probe'
 temp=R/'review_summary.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(R/'review_summary.json')
 lines=['# π0.5 异常复核报告','',f'更新时间：{result["updated_utc"]}。本轮仅定向复核，原始结果未覆盖。','',f'进度：`{result["status"]}`','', '## 1. 学习异常是否复现','', '| checkpoint | 重评回合 | 成败与原始一致 | 动作步数与原始一致 | 初态哈希一致 |','|---|---:|---:|---:|---|']
 for name,d in replay.items():lines.append(f'| {name} | {d["n"]}/20 | {d["matching_success"]} | {d["matching_policy_steps"]} | {d["state_hash_manifest_matches"]} |')
 lines+=['','固定选择两任务各初态0–9，使用原始评估种子与5步去噪，坏seed在GPU0重评，好seed的适应/历史任务分别在GPU0/GPU2独立加载checkpoint重评。该子集用于复现校验，不作为新的完整成功率估计。原始五份checkpoint参数完整且有限、均完成1000步；本次未发现载入错位或复评翻转的证据，但这不能证明训练流程不存在其他问题。','', '固定教师强制去噪损失（B4，eval模式）与原始闭环成功率：','', '| checkpoint | 适应成功率 | 历史成功率 | 适应验证loss32 | 历史回放loss32 |','|---|---:|---:|---:|---:|']
 for name in ['a_seed20261107','b_seed20261107','c_seed20261107','a_seed20261108','c_seed20261108']:
  rr=rows(OLD/'results'/f'{name}.jsonl');rates=[]
  for task in [3,0]:
   a=[x for x in rr if x['task']==task];rates.append(f'{sum(x["success"] for x in a)}/{len(a)}'+('（未满50）' if len(a)!=50 else ''))
  vals=read(R/f'validation_{name}.json');v={x['task']:x['split_loss32'] for x in vals if x['prefix_mode']=='eval'} if vals else {}
  lines.append(f'| {name} | {rates[0]} | {rates[1]} | {v.get("target_heldout",float("nan")):.6f} | {v.get("history_replay",float("nan")):.6f} |')
 lines+=['','固定样本为20个适应验证窗口（来自5个未用于训练的episode）及20个历史回放窗口（后者不是独立验证集）。坏seed的去噪loss更低，但闭环成功率差，不能用loss正常或更低来证明学习有效。native joint与split路径、train/eval模式的对照详见`validation_raw.jsonl`。采样核对每步均为4适应+4历史，a/b同seed窗口一致；未发现错配任务比例。','', '## 2. 是否只是5步去噪不足','', '| checkpoint | 历史动作MSE，5步 | 10步 | 20步 |','|---|---:|---:|---:|']
 op=read(R/'openloop_summary.json')
 if op:
  for name in ['parent','a_seed20261107','a_seed20261108','c_seed20261108']:
   vals={x['steps']:x['mean_mse7'] for x in op if x['checkpoint']==name and x['task']=='history_replay'};lines.append(f'| {name} | {vals[5]:.5f} | {vals[10]:.5f} | {vals[20]:.5f} |')
 lines+=['','同一10个历史窗口、相同初始噪声，比较归一化前7维动作MSE。a坏seed约为好seed的2.7倍，增加去噪步数未修复历史动作误差。c在适应样本上的10步误差下降，但历史误差未改善。开环MSE不是成功率，不能据此排除所有闭环采样效应；没有证据支持“只改10步就能解决整个异常”。','', '## 3. 缓存数值差异','', '相同B1前缀生成路径→拼B4的重算与缓存比较，loss及全部训练参数梯度完全一致。此前约7.17%比较同时改变了前缀计算batch，不能解释为缓存传输误差。原精度的B1/B4前缀KV差异从第一层约1.15%累积至最后一层约4.90%（本次四样本相对L2）。','', '仅关闭TF32后B1/B4梯度差约7.83%，再关闭BF16低精度归约约3.87%；均未归零。此前全FP32且关TF32约6.64e-6。这支持低精度、batch形状相关的计算差异，尚未定位单个kernel。严格缓存/重算学习对照应统一前缀生产batch、模式和输入，不能用一次FP32检查替代原精度学习验证。','', '## 4. 服务侧复核','', '| 条件 | 窗口/请求 | 合并P99 ms | 违约数 | SLO | steps/s | 图外耗时中位数 ms |','|---|---:|---:|---:|---:|---:|---:|']
 for d in systems:lines.append(f'| {d["source"]}/{d["mode"]} | {d["windows"]}/{d["requests"]} | {d["pooled_p99_completed_ms"]:.2f} | {d["late"]} | {d["slo_rate"]:.2%} | {d["mean_steps_per_s"]:.3f} | {d["outside_graph_median_ms"]:.2f} |')
 lines+=['','复核条件仍为单客户端250ms周期/250ms时限，训练MPS30%，推理100%，每条件3×60秒；未修改SLO或margin。纯推理无MPS。此次将全部服务线程固定CPU24–27，训练线程CPU28–31，其他评估/诊断用互不重叠物理核；旧评估进程原先可用全部64逻辑核，旧worker只固定主线程，存在主机争用混杂。新记录增加input/copy-submit/graph-submit/decode-wait分解。','', '图外耗时定义为service墙钟减prefix与denoise GPU事件时间，包含主机处理、提交、等待及图外拷贝等，不能把全部残差逐项归因为CPU。资源隔离后尾延迟改善支持原测量受主机环境影响，但这轮同时隔离了多个主机资源因素，不能宣称已定位唯一根因。新旧结果均保留；任何少量违约仍按原始记录报告。','', '## 判断与下一步','', '缓存传输正确性与同路径梯度等价已复核；跨batch数值等价未成立。学习坏seed的闭环退化仍是核心问题，不能将其当作偶发评估噪声忽略。接下来学习侧应优先检查训练中间checkpoint、补b的同seed对照，并采用多seed配对闭环评估；若评估真实缓存训练，必须用相同前缀生产路径。当前复核未重跑整套学习矩阵，也未事后改学习率或挑选较好seed覆盖原结果。','', '原始证据：`results/`、`validation_raw.jsonl`、`openloop_raw.jsonl`、`numerical_checks.json`、`precision_isolation.json`、`matrix_replicates/action/`、`gpu3_telemetry.csv`。完整条件见`protocol.json`。','']
 
 if (R/'input_probe_appendix.md').exists():lines.append((R/'input_probe_appendix.md').read_text())
 temp=R/'复核报告.tmp';temp.write_text('\n'.join(lines));temp.replace(R/'复核报告.md')
 return result['status']['eval'].get('state')=='complete' and result['status']['serving'].get('state')=='finished' and result['status']['openloop'].get('state')=='complete'
if __name__=='__main__':
 while True:
  try:
   if summarize():break
  except Exception as e:(R/'summary_error.txt').write_text(repr(e))
  time.sleep(20)
