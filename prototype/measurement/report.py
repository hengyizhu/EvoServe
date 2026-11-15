from pathlib import Path
import json,csv,io,os,datetime,statistics
from zoneinfo import ZoneInfo
from duty_stats import mean_duty
R=Path(__file__).resolve().parent

def read(p):
 try:return json.loads(Path(p).read_text())
 except (OSError,ValueError):return {}
def main():
 rows=[];traces=[];bad=[];windows=[]
 for p in sorted((R/'task_state').glob('*.json')):
  if p.stem.endswith(('_smoke','_formal')):continue
  x=read(p)
  if x.get('state')=='skipped':bad.append(dict(item=p.stem,reason=x['reason'],power_limits_w={'0':225,'1':225,'2':225,'3':225}));continue
  if x.get('state')!='complete':continue
  y=x['result'];raw=Path(x['raw']);sched=read(raw/'measurement_schedule.json');lo=sched.get('origin_ns',0);hi=lo+round(y['duration_s']*1e9);mon=[]
  if (raw/'host_gpu_monitor.jsonl').exists():
   for line in (raw/'host_gpu_monitor.jsonl').read_text().splitlines():
    z=json.loads(line)
    if lo<=z['ns']<=hi:mon.append(z)
  environment={}
  if mon:
   environment=dict(gpu1_max_memory_mib=max(z['cards']['1']['memory_mib'] for z in mon),gpu2_util_mean=statistics.mean(z['cards']['2']['util'] for z in mon),gpu3_util_mean=statistics.mean(z['cards']['3']['util'] for z in mon),host_load1_mean=statistics.mean(z['load'][0] for z in mon),monitor_samples=len(mon))
  server=read(raw/'pair0/measurement/server_summary.json');cal=read(raw/'calibrations.json')
  k=x.get('kernel_profile',{});row=dict(longest_kernel_us=k.get('longest_kernel_us'),kernel_profile_condition='3 training updates, inference idle, excluded from SLO windows',case=p.stem,request_rate=y['global_rate'],busy_fraction=server.get('busy_fraction'),service_wall_busy_fraction=server.get('service_wall_busy_fraction'),reference_service_ms=y['reference_service_ms'],current_idle_service_ms=cal[0]['service_ms']['mean'] if cal else None,load=x['load'],experts=x['experts'],p99_ms=y['p99_ms'],slo=y['slo_rate'],expert_steps_s=x['expert_steps_s'],mean_duty=mean_duty(raw,y),power_limits_w=y['power_limits_before_w'],observed_power_limits_w=y['power_limits_during_w'],environment=environment,raw=x['raw'])
  if x.get('trace'):traces.append(dict(**row,kernel_csv=x['kernel_csv'],nsys_report=x['nsys_report'],validation=x['trace_validation']))
  else:
   rows.append(row)
   if (raw/'windows.csv').exists():
    for z in csv.DictReader((raw/'windows.csv').open()):windows.append(dict(case=p.stem,**z,expert_steps_per_s=x['experts']*float(z['training_steps_per_s']),power_limit_w=y['power_limits_before_w']['0'],raw=str(raw)))
 for name,data in [('summary.csv',rows),('traces.csv',traces),('skipped.csv',bad),('windows.csv',windows)]:
  s=io.StringIO()
  if data:
   w=csv.DictWriter(s,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
  p=R/name;t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(s.getvalue());t.replace(p)
 state=read(R/'status.json');text=['# 75/80%负载边界与原型v0进度',f"更新：{datetime.datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')}；{state.get('state','准备')}。",'启动时功耗上限：GPU0/1/2/3均225W；本任务未修改。逐点实际读数见CSV，GPU2/3外部负载与主机CPU压力见各run的host_gpu_monitor.jsonl。时限250ms，SLO≥99%。','配置：MPS不限额＋现有超时率反馈（5s窗口，目标≤1%，占空比可降到0）；动作LoRA r32，无增强，每专家microbatch4×累积2；推理batch1、10步去噪。','', '|负载|专家数|P99 ms|SLO|专家步/s|平均占空比|最长kernel μs（独立profile）|','|---:|---:|---:|---:|---:|---:|---:|']
 for x in rows:text.append(f"|{x['load']}|{x['experts']}|{('∞' if x['p99_ms'] is None else format(x['p99_ms'],'.3f'))}|{x['slo']:.2%}|{x['expert_steps_s']:.3f}|{x['mean_duty']:.2%}|{x['longest_kernel_us']}|")
 text+=['','|70%负载trace|kernel CSV|记录数/进程数|','|---|---|---|']
 for x in traces:text.append(f"|{x['case']}|[{Path(x['kernel_csv']).name}]({x['kernel_csv']})|{x['validation']['kernels']}/{x['validation']['processes']}|")
 text.append('Trace采用CUDA Graph node tracing；其中延迟不替代未profile的SLO测量。')
 text.append('跳过：'+('；'.join(x['item']+':'+x['reason'] for x in bad) if bad else '暂无'))
 text.append('名义负载沿用142.499ms单请求参考时长；实测忙碌占比、实际空载服务时长及每分钟数据见CSV。profile为测量窗口结束后的独立训练，不计入SLO。')
 text.append('原型分支：prototype。M1/M2/M3结果按完成顺序追加；GPU阶段仅在本轮测量全部结束并通过CPU检查后开始。')
 for p in sorted((R/'milestones').glob('*.json')):
  z=read(p);text.append(p.stem+': '+json.dumps(z,ensure_ascii=False))
 p=R/'一页汇总.md';t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text('\n'.join(text)+'\n');t.replace(p)
if __name__=='__main__':main()
