import json,datetime
from pathlib import Path
from zoneinfo import ZoneInfo
R=Path(__file__).resolve().parent
def read(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return {}
def render():
 lines=['# 流式缓存稳态对照（接收端成本）','',datetime.datetime.now(ZoneInfo('America/New_York')).strftime('%m-%d %H:%M %Z')+'｜MPS30；单客户端B1、250ms周期/时限；每步4新＋4重放。最近512个新样本可重采样，移出即删缓存；历史1000窗口预热。各测20分钟，下面四个数依次对应5分钟段。','','| 到达/s | 方法 | steps/s（四段） | SLO%（四段） | P99 ms（四段） | 新/重放命中% | 新数据实际k | 前缀计算占步时% 新/重放 |','|---:|---|---|---|---|---|---:|---|']
 total={}
 for rate in [4,16]:
  for mode in ['A1','A2']:
   d=R/'runs'/f'{mode}_r{rate}';a=[read(d/f'window{i}'/'summary.json') for i in range(1,5)];a=[x for x in a if x];c=read(d/'cohort_summary.json')
   def join(fn):return '/'.join(fn(x) for x in a) or '—'
   hits={k:sum(x['training']['class_counts'][k]['hits'] for x in a) for k in ['new','replay']};samples={k:sum(sum(x['training']['class_counts'][k].values()) for x in a) for k in ['new','replay']};hr='/'.join(f'{100*hits[k]/samples[k]:.1f}' if samples[k] else '—' for k in hits)
   # Weighted prefix fractions from raw completed-step wall times.
   wall=0.;prefix={'new':0.,'replay':0.}
   for i in range(1,5):
    p=d/f'window{i}'/'steps.jsonl';cmd=read(d/f'command_{i+1:03d}.json')
    if not p.exists() or not cmd:continue
    end=cmd['origin_ns']+int(cmd['duration_s']*1e9)
    for line in p.read_text().splitlines():
     try:x=json.loads(line)
     except json.JSONDecodeError:continue
     if x['start_ns']>=cmd['origin_ns'] and x['finish_ns']<=end:
      wall+=1000*x['seconds']
      for k in prefix:prefix[k]+=x['prefix_compute_ms'][k]
   fractions='/'.join(f'{100*prefix[k]/wall:.2f}' if wall else '—' for k in prefix);k=c.get('k_complete_lifetime');kt=f'{k:.2f}' if k is not None else '—'
   lines.append(f'| {rate} | {mode} | '+join(lambda x:f'{x["training"]["steps_per_s"]:.3f}')+' | '+join(lambda x:f'{100*x["slo_rate"]:.2f}')+' | '+join(lambda x:f'{x["p99_completed_ms"]:.1f}' if x['p99_completed_ms'] is not None else '无完成')+f' | {hr} | {kt} | {fractions} |')
   if len(a)==4:total[(rate,mode)]=sum(x['training']['complete_updates'] for x in a)
 ratios=[f'{rate}/s：A2/A1={total[(rate,"A2")]/total[(rate,"A1")]:.2f}×' for rate in [4,16] if all((rate,m) in total for m in ['A1','A2']) and total[(rate,'A1')]]
 lines+=['','；'.join(ratios) if ratios else '收益倍数待完整测量。','', 'k按测量期内完整到达并移出窗口的样本计算（含零次采样）；记录数据回放模拟新经验，每次到达独立ID，不按相同帧去重。A2产出端成本沿用既有导出测量（P99约+0.33ms），不在本卡重计；接收端主机复制/存放、拼批与H2D计入实测。原始数据：[runs](runs/)（每步、每请求、每样本生命周期）；[配置](protocol.json)。']
 (R/'半页报告.md').write_text('\n'.join(lines))
if __name__=='__main__':render()
