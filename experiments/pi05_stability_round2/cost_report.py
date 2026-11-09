"""Matched short profiles, plus recorded complete-run step times."""
import json,statistics,datetime
from pathlib import Path
from zoneinfo import ZoneInfo
R=Path(__file__).resolve().parent
def read(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return {}
def render():
 lines=['# LoRA与全量动作侧成本对比','',datetime.datetime.now(ZoneInfo('America/New_York')).strftime('%m-%d %H:%M %Z')+'｜关增强，lr1e-5、EMA0.999、microbatch4×累积2；相同seed与前30步样本，预热10步、测20步；前缀冻结eval/no_grad。','','| 更新方式 | 可训练参数 | 单步中位 ms | 单步均值 ms | 峰值 allocated GiB | 峰值 reserved GiB |','|---|---:|---:|---:|---:|---:|'];output={}
 for fam in ['full','lora32']:
  d=read(R/'training'/f'cost_{fam}_b_s20261108_30'/'cost_summary.json');cfg=read(R/'training'/'full_b_s20261108'/'config.json');n=sum(x['numel'] for x in cfg.get('trainable',[])) if fam=='full' else read(R/'preflight_learning.json').get('lora_trainable_numel');n=d.get('trainable_numel',n)
  vals=[f'{d[k]*scale:.2f}' if k in d else '待测' for k,scale in [('step_seconds_median',1000),('step_seconds_mean',1000),('peak_cuda_allocated_bytes',1/2**30),('peak_cuda_reserved_bytes',1/2**30)]];lines.append('| '+fam+' | '+f'{n:,}'+' | '+' | '.join(vals)+' |');output[fam]=dict(d,trainable_numel=n)
 lines+=['','峰值显存为训练阶段CUDA分配器峰值，包含模型、梯度、FP32主权重/优化器状态、EMA与激活；不是nvidia-smi进程总占用。旧全量训练未记录此指标，表中采用两组同条件短测，不回填估计值。','','| 1000步正式训练（关增强） | seed08 单步中位ms | seed09 | seed10 |','|---|---:|---:|---:|']
 for fam in ['full','lora32']:
  vals=[]
  for seed in [20261108,20261109,20261110]:
   p=R/'training'/f'{fam}_b_s{seed}'/'steps.jsonl';rows=[]
   if p.exists():
    for line in p.read_text().splitlines():
     try:rows.append(json.loads(line))
     except json.JSONDecodeError:pass
   vals.append(f'{statistics.median(x["seconds"] for x in rows[10:])*1000:.1f}'+('（进行中）' if len(rows)<1000 else '') if len(rows)>10 else '待测')
  lines.append('| '+fam+' | '+' | '.join(vals)+' |')
 (R/'LoRA与全量成本对比.md').write_text('\n'.join(lines));(R/'cost_comparison.json').write_text(json.dumps(output,indent=2))
if __name__=='__main__':render()
