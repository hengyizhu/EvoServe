import json,pathlib,html,datetime,zoneinfo,hashlib
R=pathlib.Path('/home/hengyi/pi05_measure_20261107')
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

def table(headers,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
def f(x):return f'{x:.3f}' if isinstance(x,(float,int)) else str(x)
def main():
 parts=['<!doctype html><meta charset="utf-8"><title>π0.5 三组测量</title><style>body{font:16px system-ui;margin:40px;max-width:1300px;line-height:1.65}table{border-collapse:collapse;margin:20px 0}td,th{border:1px solid #bbb;padding:8px}th{background:#eee}code{background:#eee} .warn{color:#a43b00}</style><h1>π0.5 三组测量</h1>','<p>更新时间：'+datetime.datetime.now(zoneinfo.ZoneInfo('America/New_York')).isoformat(timespec='seconds')+'。所有结果来自本目录原始日志；待完成项不填推测值。</p>']
 parts.append('<p><b>已完成：实验 0 两种配置、实验 1 四个 batch、实验 2 八个条件（1,920 请求）。</b>冻结前缀占动作侧训练步约 56%，有复用潜力但尚未实现缓存命中实验；当前 batch=1 推理中位 294.7ms 超过实际 250ms 重规划周期。较宽松的双客户端周期下，本轮 MPS 50% 的延迟/训练吞吐折中优于现有分段实现，但这是单次短测。</p>')
 parts+=['<p>模型：RLinf-Pi05-LIBERO-130-fullshot-SFT，revision 6222623f635769bfc73c9472e29fab9b7fd8e027；BF16/FP32 混合精度主干（视觉 stem、norm 等保留 FP32），FP32 动作投影／AdamW master，未开启 torch.compile。只在进程内执行测量所需更新，未写回模型或已有 checkpoint。实验 0 实际 GPU 2，实验 1 实际 GPU 3；GPU 0/1 当时仍在完成 GIST。</p>','<h2>实验 0：训练单步拆解</h2><p>每个 optimizer step 累积 8 个 microbatch（microbatch=1，effective batch=8）；10 步预热＋50 步正式测量。各阶段 CUDA Event 计时后同步；下表为每个完整更新的阶段累计时间的中位数。预处理与 CPU 提交开销另记在 raw.jsonl。</p>']
 rows=[];anomalies=[];train_results={}
 for tier in ['action','action_vision_lora']:
  p=R/'training'/tier;s=read(p/'summary.json');par=read(p/'parity.json');cfg=read(p/'config.json')
  if s:
   train_results[tier]=s;rows.append([tier]+[f(s[k]['p50']) for k in ['prefix_ms','action_ms','backward_ms','optimizer_ms','wall_ms']]+[f(s['peak_allocated_bytes']['max']/2**30),f(s['peak_reserved_bytes']['max']/2**30)])
   if s['prefix_ms']['p50']<=0:anomalies.append(tier+'：前缀时间非正，禁止作为有效结果。')
  else:rows.append([tier,'待完成']+['—']*6)
  if par:anomalies.append(tier+'：split prefix/suffix 相对原生联合前向存在数值差异：loss abs='+format(par['loss_abs_diff'],'.6g')+'；全参数梯度相对 L2='+f(par['gradient_relative_l2']*100)+'%。记录原始 parity.json；拆解不是逐位等价的原生联合执行。')
 parts.append(table(['配置','前缀 ms','动作 FWD ms','BWD ms','OPT ms','总 wall ms','峰值 allocated GiB','峰值 reserved GiB'],rows))
 parts+=['<p>动作侧：原 L2 action expert 及动作/时间投影全量更新，LoRA rank 不适用；前缀明确 torch.no_grad，逐次断言 KV 无 grad_fn、不要求梯度。动作＋视觉 LoRA：同一动作侧，SigLIP 每层 MLP fc1/fc2 插入 rank=16、alpha=16、dropout=0 的临时 LoRA；基础视觉权重和语言权重冻结，视觉→语言→动作梯度路径保留，启用非重入 activation checkpointing。完整更新参数名、数量、dtype 在 training/*/config.json。不是视觉全量微调，也不是所有视觉层均加 LoRA。</p>','<p>显存 allocated/reserved 为 PyTorch 峰值，两者不能等同 NVML 整卡占用；CUDA context、其他库和缓存会产生差异。每种配置冷启动后测量，预热不计入 50 步；峰值不含预检梯度对比。</p>']
 if 'action' in train_results:
  s=train_results['action'];ratio=s['prefix_ms']['p50']/s['wall_ms']['p50'];parts.append('<p><b>前缀复用上界：</b>动作侧前缀占当前拆解步 wall 时间约 '+f(100*ratio)+'%；若每个训练输入都能命中完全等价的服务缓存、缓存访问零成本，理论步耗时上界加速约 '+f(1/(1-ratio))+'×。这不是已实现加速。现有训练有随机图像增强，缓存必须同时匹配图像/语言/预处理/权重版本/精度；只冻结前缀还不够。视觉 LoRA 更新时，服务 no_grad KV 不能提供视觉梯度，不能直接省去训练前缀。</p>')
 d=read(R/'data_and_control.json',{});parts.append('<p>历史训练 1,000 optimizer steps × batch 8 = 8,000 次样本窗口抽取；全数据集 273,465 帧。实际采样为每步 4 个当前任务＋4 个 replay：当前训练集 4,942 帧、4,762 个合法窗口起点，平均抽取 0.840 次/合法起点；1,000 个 replay 候选平均抽取 4 次。实际抽到 3,683 个唯一窗口，平均 2.172 次/已抽到窗口。不能用全数据集帧数直接解释本次两任务训练复用率。数据原始 FPS=10，与仿真控制 20 Hz 分开记录。</p>')
 parts.append('<h2>实验 1：推理拆解</h2><p>动作 horizon=10，去噪=10，batch=1/4/8/16；各 10 次预热＋50 次正式测量。端到端包含输入转换、前缀、去噪、解码和结果拷回 CPU，不含相机采集、网络和仿真 step。</p>')
 rows=[];denoise=[]
 for s in read(R/'inference/summary.json',[]):
  if 'status' in s:rows.append([s['batch'],s['status']]+['—']*8);continue
  gap=250-s['end_to_end_ms']['p50'];rows.append([s['batch'],f(s['prefix_ms']['p50']),f(s['denoise_total_ms']['p50']),f(s['end_to_end_ms']['p50']),f(s['prefix_tflops']),f(s['denoise_tflops']),f(s['prefix_tflops']/154.8*100),f(s['denoise_tflops']/154.8*100),f(gap),f(gap/250*100)])
  denoise.append([s['batch']]+[f(x) for x in s['denoise_each_median_ms']])
 parts.append(table(['batch','前缀 ms','10步去噪 ms','端到端 ms','前缀 TFLOPS','去噪 TFLOPS','前缀/峰值 %','去噪/峰值 %','250ms周期剩余 ms','有符号空隙 %'],rows));parts.append(table(['batch']+[f'去噪{i+1} ms' for i in range(10)],denoise))
 parts+=['<p>FLOPs 来自 PyTorch FlopCounterMode 支持的算子计数（乘加=2，含所支持 matmul/attention/卷积，未支持算子和逐元素算子未完全覆盖），是算术吞吐估算，不是 SM occupancy 测量。各阶段 TFLOPS=FLOPs/对应中位时间；混合 FP32/BF16 工作以 BF16 dense Tensor 理论峰值 154.8 TFLOPS 作参考，不能将低比例直接解释为可安全并发的空闲 SM。</p>','<p>峰值来源：<a href="https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf">NVIDIA Ampere GA102 白皮书 Table 3</a>。不使用稀疏 309.6 TFLOPS 分母。</p>','<p>LIBERO control_freq=20 Hz，每次执行 5 个动作即重规划，周期 250 ms；预测 horizon=10 对应 500 ms，但现有执行策略不等于 500 ms 才重规划。负空隙原样保留，意味着当前配置无正的整周期余量；batch>1 的这一计算假设同批客户端同步重规划，不代表串行服务能力。所有周期计算基于中位延迟，不是最坏情况保证。</p>','<p>本机未找到 nsys，未获取 SM 硬件计数器；保存了单独运行的 torch.profiler/CUPTI 时间线 inference/batch1_trace.json，正式 50 次计时未开启 profiler。这张时间线不能替代 SM occupancy 指标。</p>']
 parts+=['<h2>实验 2：同卡干扰</h2><p>两个客户端各 1 秒发一次请求，错峰 0.5 秒，SLO=1 秒；每种模式 120 秒、240 请求，单次短测，未锁定 GPU 时钟，P99 仅作探索性诊断。此负载用于比较干扰，不能证明 250 ms 机器人周期可行。推理 batch=1、10 步去噪；训练仍 microbatch=1、累积8。</p>']
 rows=[]
 for tier in ['action','action_vision_lora']:
  for mode in ['free','segmented','mps30','mps50']:
   p=R/'interference'/tier/mode;s=read(p/'summary.json');fail=read(p/'failure.json')
   if s and 'p99_ms' in s:rows.append([tier,mode,f(s['p99_ms']),f(100*s['slo_rate']),f(s['steps_per_s']),s.get('requests',s.get('n_requests')),s.get('gpu',s.get('gpu_uuid'))])
   else:rows.append([tier,mode,'失败，见 failure.json' if fail else (s.get('reason','待完成') if s else '待完成')]+['—']*4)
 parts.append(table(['训练配置','模式','P99 ms','SLO %','optimizer steps/s','请求数','物理 GPU'],rows))
 parts+=['<p>free/MPS：独立训练、推理两个进程；MPS 仅限制训练客户端 active thread percentage 为 30/50，推理为100，这不是独占 SM 分区（<a href="https://docs.nvidia.com/deploy/mps/when-to-use-mps.html">NVIDIA MPS 文档</a>）。记录可见 SM 数验证限制生效，使用本次私有 MPS 管道，结束后关闭本次 daemon。segmented：复用现有单进程独立模型的 S4 分段准入；FWD/BWD/OPT 为合法边界，等待请求优先，两周期有限回放前瞻，校准 P95 片段时长，margin=max(10ms, 推理P95−均值)。free/MPS 每个进程各使用独立 4 个 CPU 核，segmented 单进程使用 4 核；进程组织和 CPU 预算不同属于该实现比较的限制。</p>','<h2>异常与边界</h2>']
 if (R/'training/preflight_dtype_failure.json').exists():anomalies.append('预检最初遇到动作 Float64/投影 Float32 不匹配；计时脚本补上原生 compute_loss 已有的 FP32 转换后重新预检。原始失败日志保留；未修改模型、未修补任何测量数字。')
 if (R/'failed_attempts').exists():anomalies.append('MPS 首次连接因 CUDA GPU ordinal 重映射失败，未产生测量记录；改为初始化前按 UUID 选择设备后重试。失败日志在 failed_attempts/，原始失败不删除。')
 parts.append('<ul>'+''.join('<li class="warn">'+html.escape(x)+'</li>' for x in anomalies)+'</ul>')
 parts.append('<h2>原始文件</h2><ul>'+''.join('<li><a href="'+x+'">'+x+'</a></li>' for x in ['experiment0_training.csv','experiment1_inference.csv','experiment2_interference.csv','validation.json','raw_reconciliation.json','checkpoint_integrity.json','hardware_inventory.json','protocol.json','data_and_control.json','training/action/raw.jsonl','training/action_vision_lora/raw.jsonl','inference/summary.json','inference/batch1_raw.jsonl','inference/batch4_raw.jsonl','inference/batch8_raw.jsonl','inference/batch16_raw.jsonl','interference/','logs/'])+'</ul>')
 (R/'测量报告.html').write_text('\n'.join(parts))
if __name__=='__main__':main()
