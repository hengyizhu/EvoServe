import json,time,math,datetime
from pathlib import Path
R=Path('/home/hengyi/pi05_cache_20261107')
def read(p):
 try:return json.loads((R/p).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def ci(k,n):
 z=1.959963984540054;p=k/n;c=(p+z*z/(2*n))/(1+z*z/n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return f'{100*(c-h):.1f}–{100*(c+h):.1f}%'
def render():
 lines=['# π0.5 前缀缓存与学习门槛实验报告','',f'更新时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。这是随实验更新的报告；未完成项目明确标记，不补估计结果。','', '## 配置与当前状态','', '- 模型：RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT，父权重 SHA256 `4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683`。','- A6000 ×4；原有选择性 BF16/FP32 精度；动作侧全参数更新（action expert 与 action/time projections），不是 LoRA；视觉和语言前缀冻结。','- 训练 microbatch 4 ×累积2；学习每个变体1000步，AdamW FP32 master，学习率2.5e-5，50步预热，梯度裁剪1。','- 学习评估沿用历史5步去噪、horizon10、每5帧重规划、20 Hz、520动作步预算；服务性能使用CUDA Graph、10步去噪、batch1、250 ms周期与时限。两者明确区分。','', '| GPU | 工作 | 状态 |','|---|---|---|']
 for gpu,task,p in [(0,'a 主seed训练与评估','gpu0_learning_status.json'),(1,'c 主seed及第二seed训练与评估','gpu1_learning_status.json'),(2,'b 主seed训练与评估；缓存测量已完成','gpu2_learning_status.json'),(3,'a 第二seed训练与评估；MPS测量已结束','gpu3_learning_status.json')]:lines.append(f'| {gpu} | {task} | `{read(p)}` |')
 lines+=['','## 1. 学习侧门槛','', '原始抽屉任务固定初态0–49，父模型30/50=60%；历史任务44/50=88%。旧40回合与补齐10回合使用相同父模型、种子规则和评估协议。原始抽屉不满足“低于60%”，因此候选适应任务为**自定义固定低照度抽屉**：两路原始uint8图像变换为 floor(image×0.35)，训练和评估相同；历史任务不变。此条件不冒充 LIBERO-Plus/Pro。若低照度父模型仍不低于60%，使用第二个预先固定的候选：两路相机水平镜像、亮度不变；同样先完成固定50初态的父模型评估，再决定是否训练。镜像试验在GPU2先测0–9，后续复用这些记录，绝不按失败初态选择评估集。两种候选均完整保留结果。只有候选父模型完整50回合成功率低于60%，才启动完整学习矩阵。','',f'门槛状态：`{read("learning_gate_mirror.json") or read("learning_gate.json") or "评估中"}`','', '| 变体 | 任务 | 完成回合 | 成功率 | Wilson 95% CI |','|---|---|---:|---:|---:|']
 for p in sorted((R/'results').glob('*.jsonl')):
  if 'part' in p.name or 'missing' in p.name or 'pilot' in p.name:continue
  rows=[json.loads(x) for x in p.read_text().splitlines()]
  for task in [3,0]:
   rr=[x for x in rows if x['task']==task];n=len(rr)
   if n:k=sum(x['success'] for x in rr);lines.append(f'| {p.stem} | {"适应" if task==3 else "历史"} | {n}/50 | {k}/{n} ({k/n:.1%}) | {ci(k,n)} |')
 lines+=['','a：增强开启、任意帧；b：无增强、任意帧；c：无增强、起点向下对齐到每个 episode 的5帧倍数。a/b/c 使用相同 episode 混合及随机种子，4适应+4历史样本/步。c 的起点对齐是离线“重规划时刻”代理，不是机器人实采服务轨迹。所有变体用同一 split-prefix 训练实现；不将它与旧 unified-forward 实现宣称为逐位等价。第二seed为20261108；主seed为20261107。`a_seed909090` 仅两步代码冒烟，不进入学习结论。','', '原始日志：[results](results/)、[learning](learning/)。每次评估完整50回合之后才报告最终成功率；单seed的区间只描述回合采样不确定性，不代表跨seed方差。','', '## 2. 缓存成本','']
 layout=read('cache_cost/kv_layout.json');cost=read('cache_cost/summary.json')
 lines+=['| 指标 | 结果 |','|---|---:|']
 if layout:lines.append(f'| 单样本 KV 字节（不含mask） | {layout["bytes_per_sample"]:,} B / {layout["bytes_per_sample"]/2**20:.3f} MiB |')
 if cost:
  for label,value in [('纯推理P99',cost['baseline']['response_ms']['p99']),('异步导出KV的P99',cost['export']['response_ms']['p99']),('P99增加',cost['p99_increase_ms']),('8样本H2D CUDA中位数',cost['h2d_8_cuda_ms']['p50']),('8样本H2D主机墙钟中位数',cost['h2d_8_wall_ms']['p50']),('单样本D2H CUDA中位数',cost['d2h_ms']['p50'])]:lines.append(f'| {label} | {value:.3f} ms |')
  lines.append(f'| 对比前轮8样本前缀重算 | 910.168 ms（不同阶段，非完整训练加速比） |')
 else:lines.append('| 传输/推理测量 | 进行中 |')
 lines+=['','D2H 使用 pinned 主机缓冲区，等待 prefix 完成事件后与去噪重叠；下次 graph replay 等待拷贝完成，防止缓存覆盖。固定单客户端250ms到达，交替两组 baseline/导出，每条件正式500请求。8样本H2D纯传输不包含主机拼批；下节完整训练吞吐包含主机拼批和H2D。','', '原始日志：[cache_cost](cache_cost/)，包括逐请求时间戳、KV逐张量形状、传输事件耗时及逐位传输校验。','', '## 3. 缓存训练与共执行','', '| 条件 | 训练 steps/s | 推理 P99 ms | SLO |','|---|---:|---:|---:|']
 solo=read('prototype/solo_summary.json')
 if solo:lines.append(f'| 缓存训练单独运行 | {solo["steps_per_s"]:.4f} | — | — |')
 for p in sorted((R/'matrix_replicates/action').glob('*/*/summary.json')):
  d=json.loads(p.read_text());lines.append(f'| {p.parent.parent.name}/{p.parent.name} | {d.get("steps_per_s",0):.4f} | {d["p99_completed_ms"]:.3f} | {d["slo_rate"]:.2%} |')
 lines+=['| 前轮MPS30，无缓存、增强开启 | 0.300 | 232.9 | 100%（120/120） |','', '本轮额外提供**无增强、eval模式前缀的重算对照**，以减少与前轮0.30 steps/s直接比较的混杂。缓存原型使用64个固定输入的有界 pinned host cache，初始化不计入吞吐，计入每步CPU拼批/H2D。它是系统原型，不代表完整数据集缓存命中率。MPS30限制训练，推理100；并不保留独占SM。每模式3×60秒，每窗口240请求；未完成请求保留在SLO分母。','', '原始日志：[prototype](prototype/)、[matrix_replicates/action](matrix_replicates/action/)。','', '## 4. 梯度异常与机制排查','', '| 检查 | 梯度相对 L2 差异 |','|---|---:|']
 for p in ['prototype/gradient_checks.json','prototype/precision_checks.json']:
  for k,v in read(p).items():
   if isinstance(v,dict) and 'gradient_relative_l2' in v:lines.append(f'| {k} | {v["gradient_relative_l2"]:.9g} ({100*v["gradient_relative_l2"]:.6f}%) |')
 lines+=['','同批次KV经主机往返后梯度差为0；混合精度B1服务缓存与B4重算仍有约7.17%差异。FP32且关闭TF32后，B1/B4差异降至约6.64e-6，B1服务缓存/B4重算降至约5.64e-6。这支持batch形状相关的低精度数值误差是主要因素，尚未定位到单个kernel；不能将原BF16缓存训练称为数值等价。','', 'token padding长度均200，拼批前后token和mask一致；loss按所有10×32维取均值，累积按样本数加权。关闭MHA fastpath并未消除差异。FP32只是诊断对照，性能和学习结果继续用原精度。','',f'真实CUDA Graph导出的KV对照：`{read("prototype/service_graph_kv_check.json") or read("prototype/service_graph_kv_failure.json") or "待完成"}`','', '## 硬件解释','', '本报告只测A6000。显存带宽、缓存、计算吞吐与kernel实现均可能改变前缀/去噪干扰，不能把A6000上的倍率直接外推到A100/H100；“前缀占动作侧训练约80%跨卡不变”也未验证。梯度差是数值路径问题，不由显存带宽本身解释。','']
 tmp=R/'缓存复用实验报告.tmp';tmp.write_text('\n'.join(lines));tmp.replace(R/'缓存复用实验报告.md')
if __name__=='__main__':
 while True:
  try:render()
  except Exception as e:(R/'report_error.txt').write_text(repr(e))
  states=[read(f'gpu{g}_learning_status.json').get('state') for g in [0,1,2,3]]
  if all(s in ['complete','gate_failed','failed'] for s in states) and read('gate_recovery.json').get('state') in ['not_needed','mirror_candidate'] and (read('gate_recovery.json').get('state')!='mirror_candidate' or bool(read('learning_gate_mirror.json'))) and read('matrix_replicates/action/status.json').get('state')=='finished' and (R/'learn.py.gpu2.exit.json').exists():break
  time.sleep(30)
