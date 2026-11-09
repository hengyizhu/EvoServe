import json,datetime,statistics
from pathlib import Path
from zoneinfo import ZoneInfo
R=Path(__file__).resolve().parent
def read(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return {}
def render():
 lines=['# A1 基线：原1000步顺序、完整主机缓存', '', '更新 '+datetime.datetime.now(ZoneInfo('America/New_York')).strftime('%m-%d %H:%M %Z')+'；动作侧全量、关增强、BF16/FP32 AdamW，microbatch4×累积2。A1/A2共用低优先级stream，预取下一microbatch；A1冷启动，A2预生成另计。', '', '| 方法 | 单独1000步 steps/s | 共跑 steps/s | SLO | 三窗口 P99 ms |', '|---|---:|---:|---:|---|'];solo={};co={};transfer=[]
 for m in ['A0','A1','A2']:
  a=read(R/'solo'/m/'summary.json');solo[m]=a;b=[read(R/'co'/m/f'window{i}'/'summary.json') for i in range(1,4)];b=[x for x in b if x];co[m]=b;solo_s=f'{a["steps_per_s"]:.3f}' if a else '—';cs=f'{sum(x["training"]["complete_updates"] for x in b)/180:.3f}' if len(b)==3 else '—';slo=f'{100*sum(x["slo_rate"]*x["requests"] for x in b)/sum(x["requests"] for x in b):.2f}%' if len(b)==3 else '—';p99=' / '.join(f'{x["p99_completed_ms"]:.1f}' if x['p99_completed_ms'] is not None else '无完成请求' for x in b) or '—';lines.append(f'| {m} | {solo_s} | {cs} | {slo} | {p99} |')
  transfer.append({'mode':m,'windows':[{'training_h2d_ms':x['training']['h2d_ms'],'training_prefix_h2d_ms':x['training']['prefix_h2d_ms'],'training_wait_h2d_ms':x['training']['compute_wait_for_h2d_ms'],'inference_input_transfer':x['input_transfer']} for x in b]})
 if all(solo[m] for m in ['A1','A2']):lines+=['',f'A2/A1 单独吞吐＝{solo["A2"]["steps_per_s"]/solo["A1"]["steps_per_s"]:.2f}×；A1完整1000步命中率＝{solo["A1"]["hit_rate"]*100:.2f}%。']
 else:lines+=['','A2/A1倍数待测；原序列8000次采样/3653个唯一窗口，A1无淘汰预期命中54.34%（序列计算，非性能实测）。']
 if len(co['A1'])==len(co['A2'])==3:lines.append(f'共跑吞吐 A2/A1＝{sum(x["training"]["complete_updates"] for x in co["A2"])/sum(x["training"]["complete_updates"] for x in co["A1"]):.2f}×。')
 lines+=['','| 搬运耗时 ms：p50 / p99 | 训练每步 H2D（输入＋KV） | 推理输入 H2D | 推理输入准备总耗时 |','|---|---:|---:|---:|']
 for mode in ['A0','A1','A2']:
  def readrows(pattern):
   values=[]
   for p in (R/'co'/mode).glob(pattern):
    for line in p.read_text().splitlines():
     try:values.append(json.loads(line))
     except json.JSONDecodeError:pass
   return values
  st=readrows('window*/steps.jsonl');req=[x for x in readrows('window*/requests.jsonl') if x.get('finish_ns') is not None]
  def quantile(values):
   if not values:return '—'
   a=sorted(values)
   def q(f):
    pos=(len(a)-1)*f;lo=int(pos);hi=min(lo+1,len(a)-1);return a[lo]+(a[hi]-a[lo])*(pos-lo)
   return f'{q(.5):.3f} / {q(.99):.3f}'
  lines.append('| '+mode+' | '+quantile([x['h2d_ms'] for x in st])+' | '+quantile([x['input_h2d_cuda_ms'] for x in req])+' | '+quantile([x['input_wall_ms'] for x in req])+' |')
 lines+=['','每步H2D原始记录见 `co/<模式>/window*/steps.jsonl`；推理每请求输入CUDA搬运/CPU打包/提交/准备时间见同目录 `requests.jsonl`；分位数见 [搬运汇总](transfer_distributions.json)。CUDA传输区间可与计算重叠，不能直接当作额外步延迟。A1共跑从空缓存开始并跨窗口保留，三组按各自实际进度遍历同一序列。']
 (R/'半页报告.md').write_text('\n'.join(lines));(R/'transfer_distributions.json').write_text(json.dumps(transfer,indent=2))
if __name__=='__main__':render()
