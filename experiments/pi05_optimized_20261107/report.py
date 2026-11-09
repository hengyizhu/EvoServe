import pathlib,json,html,csv,datetime,zoneinfo
R=pathlib.Path('/home/hengyi/pi05_optimized_20261107')
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

def f(x,n=2):return '—' if x is None else f'{x:.{n}f}'
def table(head,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
def csvout(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (R/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
 parts=['<!doctype html><meta charset="utf-8"><title>π0.5 CUDA Graph 与阶段干扰</title><style>body{font:16px system-ui;line-height:1.6;margin:36px;max-width:1600px}table{border-collapse:collapse;margin:18px 0}td,th{border:1px solid #bbb;padding:7px}th{background:#eee}.warn{color:#aa3700}details{margin:15px 0}</style><h1>π0.5 优化推理与分阶段干扰</h1>','<p>更新时间：'+datetime.datetime.now(zoneinfo.ZoneInfo('America/New_York')).isoformat(timespec='seconds')+'。旧 eager 原始记录保留；关于服务周期余量的分析以本报告的 CUDA Graph 实现为准，不代表所有可能优化的性能上限。</p>','<p>同一 RLinf-Pi05-LIBERO-130-fullshot-SFT checkpoint，BF16/FP32 混合精度，动作 horizon=10、10 步去噪。CUDA Graph 捕获完整前缀＋10步去噪，外部 CUDA Event 保留分阶段计时；仅在进程内把 CPU 常量创建/赋值换为 GPU 等值构造以满足捕获要求，未修改权重或仓库源码。对每个 batch 的 4 组不同输入，与未适配的原始 eager 动作逐项比较。未做量化、剪枝、改模型或修改 denoising 次数。</p>','<h2>1. 优化推理及控制周期</h2><p>各 batch 预热10次、正式100次；独立进程重新加载以释放图内存。输入转换、图输入复制、解码和动作拷回 CPU 都计入端到端；不含相机/网络/环境 step。统计表里的 P99 来自100次隔离调用，不是请求到达流 P99。</p>']
 if (R/'completion.json').exists():
  parts.insert(2,'<p><b>全部完成：70个主矩阵条件＋6个独立卡复核，四卡已释放。</b>核心结论：CUDA Graph单客户端有约109ms中位周期余量；MPS30%比50%更接近250ms时限，但保留一次违约；前缀和去噪均被拖慢，不能声称去噪可免费共享。现有分段实现microbatch4在窗口内零完整更新。2/4/8客户端FIFO batch1纯推理已过载，不能把其全部排队延迟归于训练。详情及数值边界如下。</p>')
 old={d['batch']:d for d in read(pathlib.Path('/home/hengyi/pi05_measure_20261107/inference/summary.json'),[])};rows=[];export=[];denoise=[]
 for d in read(R/'inference/summary.json',[]):
  rows.append([d['batch'],f(old[d['batch']]['end_to_end_ms']['p50']),f(d['prefix_ms']['p50']),f(d['denoise_ms']['p50']),f(d['end_to_end_ms']['p50']),f(d['end_to_end_ms']['p99']),f(d['gap_ms']),f(d['gap_ms']/250*100)])
  denoise.append([d['batch']]+[f(x) for x in d['denoise_each_median_ms']]);export.append({'batch':d['batch'],'prefix_p50_ms':d['prefix_ms']['p50'],'denoise_p50_ms':d['denoise_ms']['p50'],'end_to_end_p50_ms':d['end_to_end_ms']['p50'],'end_to_end_p99_ms':d['end_to_end_ms']['p99'],'prefix_tflops':d['prefix_tflops'],'denoise_tflops':d['denoise_tflops'],'gap_p50_ms':d['gap_ms'],'peak_allocated_bytes':d['peak_allocated_bytes'],'peak_reserved_bytes':d['peak_reserved_bytes']})
 parts.append(table(['batch','旧 eager E2E ms','Graph前缀 ms','Graph去噪 ms','Graph E2E P50 ms','Graph E2E P99 ms','250ms−P50 ms','有符号余量 %'],rows));parts.append('<details><summary>每一步去噪的中位耗时</summary>'+table(['batch']+[str(i+1)+' ms' for i in range(10)],denoise)+'</details>');csvout('optimized_inference.csv',export)
 parts+=['<p>LIBERO control_freq=20 Hz，当前每执行5个动作重规划，实际周期/主时限=250ms；预测10个动作对应500ms，但本轮没有改变执行间隔。正的中位余量不是每个请求的最坏情况余量。batch 4/8 的整批延迟也列出，避免把单客户端余量等同于多客户端容量。</p>','<p>FLOPs 沿用已计数的同一计算图算子工作量，结果在 optimized_inference.csv；CUDA Graph 改变提交方式而不改变数学操作。低 TFLOPS 不能直接证明 SM 空闲或训练可免费共执行；本机没有 nsys 硬件 SM 占用指标，本轮以阶段实际 slowdown 检验假设。</p>','<h2>2. microbatch 4 训练形态</h2><p>microbatch 4、累积2次、effective batch8；同样10步预热＋50步正式测量。动作侧全量更新；视觉 LoRA rank=16、alpha=16、SigLIP MLP fc1/fc2，视觉配置仍开启非重入 activation checkpointing。FP32 AdamW master；训练维持原 eager 路径，未做训练图编译，不能把这些成本当作所有训练实现的下界。每个样本保留原增强、噪声和时间采样种子。</p>']
 rows=[];export=[]
 for tier in ['action','action_vision_lora']:
  d=read(R/'training_mb4'/tier/'summary.json');par=read(R/'training_mb4'/tier/'microbatch_parity.json')
  if d:
   rows.append([tier]+[f(d[k]['p50']) for k in ['prefix_ms','action_ms','backward_ms','optimizer_ms','wall_ms']]+[f(d['peak_allocated_bytes']['max']/2**30),f(par['gradient_relative_l2']*100)])
   export.append({'tier':tier,**{k+'_p50':v['p50'] for k,v in d.items()},'gradient_relative_l2_vs_mb1':par['gradient_relative_l2']})
 parts.append(table(['更新范围','前缀 ms','动作前向 ms','反向 ms','OPT ms','总 wall ms','峰值 allocated GiB','相对MB1梯度差 %'],rows));csvout('training_mb4.csv',export)
 parts.append('<p class="warn">microbatch1→4 的同8样本累积梯度相对 L2 差约9.6%，原始 loss/梯度差保留在 microbatch_parity.json。只比较性能，不声称训练轨迹等价；未通过改变精度或重写数据消除该差异。此前 split 与 native 联合执行约1.9%的梯度差也仍属于方法边界。</p>')
 parts+=['<h2>3. 采样次数 k 与缓存边界</h2>'];d=read(R/'sampling_k.json',{});rows=[]
 for key in ['target','replay','combined']:
  if key in d:
   x=d[key];rows.append([key,x['candidate_unique_windows'],x['draws'],x['unique_observed_windows'],f(x['k_per_candidate'],3),f(x['k_per_observed_window'],3),f(100*x['within_training_exact_window_repeat_fraction'])])
 parts.append(table(['样本池','候选窗口','抽取次数','实际唯一窗口','k/候选窗口','k/已抽窗口','重复抽取占比 %'],rows));parts.append('<p>历史预算1,000 optimizer steps×batch8=8,000窗口。C的公式应使用相应样本池的k；当前任务未被抽到的候选窗口不能算成已训练样本。数据集共273,465帧，当前选定53个训练episode共14,513帧；每个训练项是10动作窗口，其起点复用次数不同于动作帧覆盖次数。</p>')
 rows=[]
 for x in read(R/'cache_sensitivity.json',[]):rows.append([x['training_shape'],f(x['k'],3),f(x['prefix_ms_per_sample']),f(x['other_compute_ms_per_sample']),f(x['A2_vs_A1_ideal_saving_fraction']*100)])
 parts.append(table(['训练形状','k','前缀C ms/样本','其他计算R ms/样本','A2相对A1理想节省 %'],rows));parts.append('<p>理想公式 C/(C+kR)：A1 每个不同输入只计算一次前缀，A2 全部命中已有服务缓存；忽略缓存读写/存储、预处理与服务成本。随机图像增强会使 A1 和 A2 都失效，且需要匹配权重版本、精度、输入和模型模式。本轮没有关闭增强、没有实现缓存，也没有评估去掉增强的学习质量。</p>')
 parts+=['<h2>4. 250ms 到达流：纯推理与共执行</h2><p>1客户端对照，以及2/4/8客户端的同步、均匀错峰到达；每客户端周期与时限均250ms。每条件30秒，之后5秒停止新派发，允许已开始的请求完成；batch=1 FIFO，不做动态合批。到达窗口结束后训练停止；排空结果不作为稳态长期保证。正式共执行均用CUDA Graph推理、microbatch4训练。每个模式开始前，两进程均完成预热，训练停止活动，额外测100次同卡/同MPS环境下的隔离推理作为阶段slowdown分母。</p>','<p class="warn">如果存在未完成请求，完整请求集合的P99不可观测，表中P99全体留空；P99已完成仅描述已完成子集。SLO分母始终包含所有请求。即使全部排空，过载窗口的P99也不是稳态容量结论。阶段统计只纳入训练活动窗口内开始且结束的请求，排空阶段不纳入阶段slowdown。主表倍率相对同卡、相同到达流的纯推理阶段中位数；另在CSV保留相对本模式100次隔离调用的倍率。</p>']
 results=[]
 for p in sorted((R/'matrix').glob('*/*/*/summary.json')):
  x=read(p)
  if x:results.append(x)
 pure={(x['tier'],x['clients'],x['pattern']):x for x in results if x['mode']=='pure'}
 for x in results:
  base=pure.get((x['tier'],x['clients'],x['pattern']))
  for phase,key in [('I_PREFIX_ms','prefix_vs_pure_arrival'),('I_ACTION_ms','denoise_vs_pure_arrival')]:
   x[key]=x['phases'][phase]['p50']/base['phases'][phase]['p50'] if base and base['phases'][phase] and x['phases'][phase] else None
 flat=[]
 for x in results:
  flat.append({k:x.get(k) for k in ['tier','mode','physical_gpu','clients','pattern','requests','completed','unfinished','slo_rate','p99_all_ms','p99_completed_ms','steps_per_s','prefix_slowdown_median','denoise_slowdown_median','phase_samples_within_window','prefix_vs_pure_arrival','denoise_vs_pure_arrival']})
 csvout('matrix_summary.csv',flat)
 phase_rows=[]
 for x in results:
  base=pure.get((x['tier'],x['clients'],x['pattern']))
  for phase in ['I_PREFIX_ms','I_ACTION_ms']:
   if not x['phases'][phase]:continue
   vals=x['phases'][phase];b=base['phases'][phase] if base else None
   phase_rows.append({'tier':x['tier'],'mode':x['mode'],'gpu':x['physical_gpu'],'clients':x['clients'],'pattern':x['pattern'],'phase':phase,'samples':x['phase_samples_within_window'],'p50_ms':vals['p50'],'p95_ms':vals['p95'],'p99_ms':vals['p99'],'pure_same_arrival_p50_ms':b['p50'] if b else None,'slowdown_p50':vals['p50']/b['p50'] if b else None})
 csvout('phase_interference.csv',phase_rows)
 heads=['范围','模式','GPU','N','到达','完成/总数','SLO %','P99全体 ms','P99已完成 ms','steps/s','前缀/同负载pure','去噪/同负载pure']
 def row(x):return [x['tier'],x['mode'],x['physical_gpu'],x['clients'],x['pattern'],str(x['completed'])+'/'+str(x['requests']),f(100*x['slo_rate']),f(x['p99_all_ms']),f(x['p99_completed_ms']),f(x['steps_per_s'],3),f(x['prefix_vs_pure_arrival'],3),f(x['denoise_vs_pure_arrival'],3)]
 parts.append('<h3>单客户端控制</h3>'+table(heads,[row(x) for x in results if x['clients']==1]));parts.append('<h3>纯推理多客户端容量</h3>'+table(heads,[row(x) for x in results if x['mode']=='pure' and x['clients']>1]));parts.append('<details><summary>全部条件（当前已完成 '+str(len(results))+'/70）</summary>'+table(heads,[row(x) for x in results])+'</details>')
 parts.append('<p>free/MPS使用独立训练、推理进程，各4个CPU核；分段沿用单进程独立模型、4个CPU核，S4两周期有限回放前瞻、固定10ms margin、校准P95 FWD/BWD/OPT成本。进程/CPU预算不同是实现比较的限制。MPS 30/50%是训练客户端上限而非独占分区，推理100%；记录可见SM数。全部为单次短测，未锁GPU频率。</p>')
 reps=[read(p) for p in sorted((R/'matrix_replicates').glob('*/*/*/summary.json'))];reps=[x for x in reps if x]
 rpure={x['tier']:x for x in reps if x['mode']=='pure'}
 for x in reps:
  base=rpure.get(x['tier'])
  for phase,key in [('I_PREFIX_ms','prefix_vs_pure_arrival'),('I_ACTION_ms','denoise_vs_pure_arrival')]:
   x[key]=x['phases'][phase]['p50']/base['phases'][phase]['p50'] if base and x['phases'][phase] else None
 csvout('replicate_summary.csv',[{k:x.get(k) for k in ['tier','mode','physical_gpu','requests','completed','unfinished','slo_rate','p99_all_ms','steps_per_s','prefix_vs_pure_arrival','denoise_vs_pure_arrival']} for x in reps])
 parts.append('<h2>5. 独立卡复核（GPU 0/1）</h2><p>GIST 完成释放 GPU 后，复核1客户端、250ms周期下纯推理与MPS 30/50%，不改变主矩阵。主矩阵GPU 2/3与复核GPU 0/1分别和各自同卡纯推理比较。功耗/温度/时钟记录在 gpu_telemetry.csv（从主矩阵 free 阶段中途开始，覆盖后续MPS及全部复核）；未锁频，单卡热状态仍是限制。</p>'+table(heads,[row(x) for x in reps]))
 failures=list((R/'matrix').glob('*/*/failure.json'))+list((R/'matrix_replicates').glob('*/*/failure.json'));parts.append('<p>当前失败条件：'+html.escape(', '.join(str(p.relative_to(R)) for p in failures) or '无')+'</p>')
 miss_path=R/'matrix/action_vision_lora/mps30/n1_sync/deadline_miss_timeline.json'
 if miss_path.exists():
  miss=read(miss_path);x=next(z for z in miss['neighbors'] if z['request_id']==miss['missed_request_id']);extra=x['service_ms']-x['I_PREFIX_ms']-x['I_ACTION_ms']
  parts.append('<h2>单次违约保留</h2><p>主矩阵 GPU3、视觉LoRA、MPS30%、1客户端：request '+str(x['request_id'])+'，端到端服务 '+f(x['service_ms'])+'ms，前缀 '+f(x['I_PREFIX_ms'])+'ms，去噪 '+f(x['I_ACTION_ms'])+'ms；两个GPU阶段之外约 '+f(extra)+'ms，尚不能区分输入/复制/提交/输出等来源。没有事后调 margin，也没有删除此请求。<a href="'+str(miss_path.relative_to(R))+'">原始请求及邻居时间线</a>。</p>')
 links=['completion.json','validation.json','checkpoint_integrity.json','single_client_queue_analysis.json','protocol.json','optimized_inference.csv','training_mb4.csv','matrix_summary.csv','phase_interference.csv','replicate_summary.csv','gpu_telemetry.csv','sampling_k.json','sampling_windows.csv','cache_sensitivity.json','inference/','training_mb4/','matrix/','logs/'];parts.append('<h2>原始数据与配置</h2><ul>'+''.join('<li><a href="'+p+'">'+p+'</a></li>' for p in links)+'</ul>');(R/'优化推理与阶段干扰报告.html').write_text('\n'.join(parts))
 return len(results)+len(reps)
if __name__=='__main__':main()
