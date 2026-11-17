"""Detached single-GPU two-knob comparison, with immutable code snapshot per campaign."""
from pathlib import Path
import argparse,csv,json,math,os,signal,statistics,subprocess,sys,threading,time,traceback
import measure
from measure import save,read,quantile
sys.path.insert(0,str(measure.W))
from arrival_plan import schedule
REF_MS=142.49900977
CONFIGS={'adaptive':('feedback','adaptive',1),'fixed8':('feedback','fixed',8),'fixed1':('feedback','fixed',1),'mps30_1':('mps30','fixed',1),'inference_only':('mps100','fixed',1)}
EXTERNAL_POLICY='gpu0_exclusive_gpu123_monitored'

def rows(p):
 if not p.exists():return []
 out=[]
 for line in p.read_text().splitlines():
  try:out.append(json.loads(line))
  except json.JSONDecodeError:pass
 return out
def write_csv(p,rs):
 if not rs:return
 with Path(p).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

def summarize_run(out):
 out=Path(out);cfg=read(out/'measurement_schedule.json');pair=out/'pair0';origin=cfg['origin_ns'];duration=cfg['duration_s'];plan=schedule(dict(cfg,rate=cfg['global_rate']));requests={r['request_id']:r for r in rows(pair/'measurement/requests.jsonl')};steps=[s for s in rows(pair/'measurement/steps.jsonl') if origin<=s['start_ns'] and s['finish_ns']<=origin+round(duration*1e9)]
 controls=rows(pair/'measurement/controller.jsonl');decisions=rows(pair/'measurement/feedback_windows.jsonl');conf=read(out/'config.json');initial=conf['initial_experts'];stage_rows=[];window_rows=[]
 counts=[0]*8
 assert steps or conf.get('inference_only'),'No completed training updates in smoke/measurement'
 if conf.get('inference_only'):assert not steps,'Inference-only baseline performed training'
 for s in steps:
  assert s['experts'] in [1,2,4,8] and len(set(s['expert_ids']))==s['experts']
  counts=[n+int(e in s['expert_ids']) for e,n in enumerate(counts)]
  assert s['optimizer_steps']==counts and s['ema_steps']==counts,'Expert state counters lost at a count switch'
  assert math.isfinite(s['loss'])
 save(out/'step_boundary_validation.json',dict(state='passed',updates=len(steps),expert_steps=sum(counts),optimizer_steps=counts,ema_steps=counts))
 def stats(lo,hi):
  ps=[r for r in plan if lo<=r['nominal_release_ns']<hi];rs=[requests[r['request_id']] for r in ps if r['request_id'] in requests];lat=[(requests[r['request_id']]['finish_ns']-r['nominal_release_ns'])/1e6 if r['request_id'] in requests else float('inf') for r in ps];ss=[s for s in steps if lo<s['finish_ns']<=hi];n=len(ps);p99=quantile(lat);secs=(hi-lo)/1e9;passed=sum(r['finish_ns']<=r['deadline_ns'] for r in rs)
  return dict(requests=n,completed=len(rs),slo_rate=passed/n if n else None,timeout_rate=1-passed/n if n else None,p99_ms=p99 if p99 is not None and math.isfinite(p99) else None,unfinished=n-len(rs),expert_steps=sum(s['expert_steps'] for s in ss),expert_steps_per_s=sum(s['expert_steps'] for s in ss)/secs,training_steps_per_s=len(ss)/secs,merged_experts_min=min([s['experts'] for s in ss],default=None),merged_experts_max=max([s['experts'] for s in ss],default=None))
 elapsed=0
 for i,p in enumerate(cfg['phases']):
  lo=origin+round(elapsed*1e9);elapsed+=p['seconds'];hi=origin+round(elapsed*1e9)
  stage_rows.append(dict(configuration=conf['configuration']+(' / smoke5m' if duration==300 and ('smoke' in conf['configuration'] or conf.get('smoke')) else ''),stage=i+1,target_load=p['load'],rate=p['rate'],seconds=p['seconds'],**stats(lo,hi),power_limit_w=225))
 for i in range(math.ceil(duration/5)):
  lo=origin+int(i*5e9);hi=min(origin+round(duration*1e9),lo+5_000_000_000);prior=[r for r in controls if r['observed_ns']<=lo];state=prior[-1] if prior else {};ps=[r for r in plan if lo<=r['nominal_release_ns']<hi];ob=next((r for r in decisions if r['window_start_ns']==lo),{})
  window_rows.append(dict(configuration=conf['configuration'],t_s=i*5,stage=ps[0]['stage'] if ps else None,target_load=ps[0]['target_load'] if ps else None,**stats(lo,hi),merged_experts_at_window_start=0 if conf.get('inference_only') else state.get('experts_after',initial),duty_at_window_start=0. if conf.get('inference_only') else state.get('duty_after',1.),observed_p99_ms=ob.get('p99_ms'),observed_timeout_rate=ob.get('timeout_rate'),observed_p99_status=ob.get('p99_status'),power_limit_w=225))
 write_csv(out/'windows_5s.csv',window_rows);write_csv(out/'stages.csv',stage_rows)
 result=dict(state='complete',configuration=conf['configuration'],duration_s=duration,total_expert_steps=sum(s['expert_steps'] for s in steps),stages=stage_rows,power_limit_w=225,reference_service_ms=REF_MS,completed_training_steps=len(steps),final_optimizer_steps=steps[-1]['optimizer_steps'] if steps else [],actual_merged_counts=sorted(set(s['experts'] for s in steps)),raw=str(out),fresh_calibrations=read(out/'calibrations.json'))
 result['external_load_policy']=conf.get('external_load_policy','gpu1_idle_original')
 mon=[x for x in rows(out/'host_gpu_monitor.jsonl') if origin<=x['ns']<origin+round(duration*1e9)]
 if mon:
  ext={f'gpu{g}_{k}':v for g in [1,2,3] for k,v in dict(util_mean=statistics.mean(x['cards'][str(g)]['util'] for x in mon),util_max=max(x['cards'][str(g)]['util'] for x in mon),memory_mib_min=min(x['cards'][str(g)]['memory_mib'] for x in mon),memory_mib_max=max(x['cards'][str(g)]['memory_mib'] for x in mon)).items()}
  ext['host_load1_mean']=statistics.mean(x['load'][0] for x in mon);ext['host_load1_max']=max(x['load'][0] for x in mon);result['external_load_summary']=ext
  save(out/'external_load_summary.json',ext)
 save(out/'result.json',result)
 if conf['configuration']=='adaptive':
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,axes=plt.subplots(4,1,figsize=(10,8),sharex=True)
  for ax,key,label in zip(axes,['target_load','merged_experts_at_window_start','duty_at_window_start','p99_ms'],['Offered load','Merged experts','Training duty','P99 (ms)']):
   ax.step([r['t_s']/60 for r in window_rows],[float('nan') if r[key] is None else r[key] for r in window_rows],where='post');ax.set_ylabel(label);ax.grid(alpha=.25)
  axes[-1].axhline(250,color='red',ls='--');axes[-1].set_xlabel('Minutes');fig.suptitle('Adaptive count + duty | A6000 225 W');fig.tight_layout();fig.savefig(out/'adaptive_timeline.png',dpi=160);fig.savefig(out/'adaptive_timeline.pdf');plt.close(fig)
 return result

class Run(measure.Experiment):
 def __init__(self,out,configuration,initial=None):
  self.admission_gpus=[0];self.inference_only=configuration=='inference_only'
  policy,mode,count=CONFIGS[configuration];self.control=mode;self.initial=initial or count;self.configuration=configuration if initial is None else f'fixed{initial}'
  super().__init__(out,'single',policy,'A2',False,0)
 def env(self,gpu,train=False):
  e=super().env(gpu,train);e.update(DYNAMIC_BANK='1',EXPERT_COUNT='8',CONTROL_MODE=self.control,INITIAL_EXPERTS=str(self.initial));return e
 def sample(self):
  super().sample()
  if self.mon:
   x=self.mon[-1]
   if x['cards'][0]['power_limit_w']!=225:raise RuntimeError('GPU0 225 W power setting changed externally')
   if time.monotonic()-getattr(self,'last_idle_check',0)>5:
    self.last_idle_check=time.monotonic();raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True);processes=[];own={p.pid for p in self.children};foreign=[];mps=[]
    for line in raw.splitlines():
     uuid,pid,name,memory=[v.strip() for v in line.split(',',3)];pid=int(pid);r=dict(gpu_uuid=uuid,pid=pid,name=name,memory_mib=memory)
     try:r.update(uid=Path(f'/proc/{pid}').stat().st_uid,cpu_affinity=sorted(os.sched_getaffinity(pid)))
     except (FileNotFoundError,ProcessLookupError,PermissionError):pass
     processes.append(r)
     if uuid==self.before['cards'][0]['uuid'] and pid not in own:
      if name.endswith('nvidia-cuda-mps-server') and r.get('uid')==os.getuid():mps.append(r)
      else:foreign.append(r)
    with (self.out/'compute_process_monitor.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),ns=time.monotonic_ns(),processes=processes))+'\n')
    if foreign or len(mps)>1:raise RuntimeError('External compute process on GPU0: '+json.dumps(foreign or mps))
 def command(self,cfg,timeout):
  cfg=dict(cfg)
  if cfg['op']=='run':cfg['inference_only']=self.inference_only
  return super().command(cfg,timeout)
 def execute_phases(self,phases):
  self.start();self.command(dict(op='activate',mode='A2',variant='A2'),180)
  rate=.7/(REF_MS/1000);self.window('warmup',512/rate+5,rate,0)
  pre=dict(op='run',mode='A2',variant='A2',name='preflight',global_rate=phases[0]['rate'],clients=4,duration_s=60,round=0,rate_cap=0,origin_ns=time.monotonic_ns()+5_000_000_000,phases=[dict(p,seconds=60/len(phases)) for p in phases],reset_expert_states=True)
  save(self.out/'preflight_schedule.json',pre);self.command(pre,240)
  server=read(self.out/'pair0/preflight/server_summary.json');train=read(self.out/'pair0/preflight/train_summary.json')
  assert server['completed']>0 and (train['complete_updates']==0 if self.inference_only else train['complete_updates']>0),'Preflight produced no serving/training progress'
  save(self.out/'preflight_result.json',dict(functional_pass=True,performance_pass=server['slo_rate']>=.99,slo_rate=server['slo_rate'],p99_ms=server['p99_ms'],training_updates=train['complete_updates'],inference_only=self.inference_only,external_load_policy=EXTERNAL_POLICY))
  duration=sum(p['seconds'] for p in phases);cfg=dict(op='run',mode='A2',variant='A2',name='measurement',global_rate=phases[0]['rate'],clients=4,duration_s=duration,round=1,rate_cap=0,origin_ns=time.monotonic_ns()+5_000_000_000,phases=phases,reset_expert_states=True)
  save(self.out/'config.json',dict(configuration=self.configuration,initial_experts=self.initial,total_experts=8,reference_service_ms=REF_MS,deadline_ms=250,policy=self.policy,power_limit_w=225,train_cpu=[4,5,6,7],server_cpu=[0,1,2,3],phases=phases,inference_only=self.inference_only,external_load_policy=EXTERNAL_POLICY,baseline_path='Same A2 serving/cache export/receiver; no optimizer updates; trainer remains resident and ingests host cache' if self.inference_only else None))
  save(self.out/'measurement_schedule.json',cfg);self.command(cfg,duration+240);return summarize_run(self.out)

def update_report(root):
 root=Path(root);items=read(root/'queue_status.json').get('items',[]);stages=[];totals=[];external=[]
 for item in items:
  p=Path(item.get('out',''))/'result.json'
  if item.get('state')=='complete' and p.exists():
   r=read(p);condition=r.get('external_load_policy','gpu1_idle_original');stages.extend(dict(s,external_load_policy=condition) for s in r['stages']);totals.append(dict(task=item['name'],configuration=r['configuration'],seconds=r['duration_s'],expert_steps=r['total_expert_steps'],raw=r['raw'],external_load_policy=condition))
   ex=r.get('external_load_summary')
   if ex:external.append(dict(task=item['name'],external_load_policy=condition,**ex))
 write_csv(root/'stage_summary.csv',stages);write_csv(root/'total_expert_steps.csv',totals)
 write_csv(root/'external_load_comparison.csv',external)
 lines=['# 双旋钮动态负载测量','',f'更新：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}；新协议只要求GPU0独占，225 W 不变。GPU1/2/3外部负载、进程绑核及主机CPU负载持续记录。','',f'负载按历史单请求 {REF_MS:.6f} ms 定义；70%/85% 为 {0.7/(REF_MS/1000):.4f}/{0.85/(REF_MS/1000):.4f} 请求/秒。250 ms 时限，SLO ≥99%；各配置同一到达序列、8 个独立专家、每专家有效 batch 8、服务缓存。','', '| 配置/条件 | 阶段/负载 | SLO | P99 ms | 专家步/秒 | 专家步总数 |','|---|---|---:|---:|---:|---:|']
 for r in stages:lines.append(f"| {r['configuration']} / {'原GPU1空闲' if r['external_load_policy']=='gpu1_idle_original' else '新共享主机'} | {r['stage']} / {r['target_load']:.0%} | {r['slo_rate']:.3%} | {r['p99_ms'] if r['p99_ms'] is not None else '∞/未完成'} | {r['expert_steps_per_s']:.3f} | {r['expert_steps']} |")
 lines+=['','队列：']+[f"- {x['name']}: {x['state']}"+(f"；{x.get('error','')}" if x.get('error') else '') for x in items]
 lines+=['','正式每组 30 分钟总专家步数：']+[f"- {x['task']}: {x['expert_steps']}（{x['seconds']/60:g} 分钟）" for x in totals if x['seconds']==1800]
 lines+=['','异常与口径：原协议和新共享主机协议分别标注，外部负载不同的数据不直接作同条件优劣判定。见 external_load_comparison.csv。纯推理基线保留相同缓存导出/接收路径及常驻模型，但不进行训练更新。缺失请求计超时；保留原始超时，跨阶段更新归入完成时刻。','']
 (root/'summary.md').write_text('\n'.join(lines))

def campaign(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);state=dict(state='running',pid=os.getpid(),started=time.time(),items=[]);lock=threading.Lock();stop=threading.Event()
 def persist():save(root/'queue_status.json',state)
 def heartbeat():
  while not stop.is_set():
   try:
    x=measure.observe();x['gpu01_processes']={str(g):subprocess.check_output(['nvidia-smi',f'--id={g}','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip() for g in [0,1]}
    with (root/'heartbeats.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
    with lock:state['heartbeat']=time.time();state['gpu_idle']={g:not ps for g,ps in x['gpu01_processes'].items()};persist()
   except Exception:
    with (root/'heartbeat_errors.log').open('a') as f:f.write(traceback.format_exc())
   stop.wait(1800)
 threading.Thread(target=heartbeat,daemon=True).start()
 tasks=[('smoke5m','adaptive',100,None,None)]+[(k,k,600,None,None) for k in CONFIGS]+[(f'boundary80_m{n}','fixed1',600,.8,n) for n in [1,2,4,8]]
 smoke_ok=False
 for name,configuration,seconds,load,count in tasks:
  item=dict(name=name,configuration=configuration,state='pending');state['items'].append(item)
  if configuration=='adaptive' and name!='smoke5m' and not smoke_ok:item.update(state='skipped',error='5-minute adaptive smoke failed twice');persist();update_report(root);continue
  for attempt in [1,2]:
   out=root/f'{name}_attempt{attempt}';item.update(state='running',attempt=attempt,out=str(out),started=time.time());state['current']=name;persist();update_report(root)
   cmd=[sys.executable,'-u',__file__,'run','--root',str(out),'--configuration',configuration,'--phase-seconds',str(seconds)]
   if load is not None:cmd+=['--constant-load',str(load),'--initial',str(count)]
   with (root/f'{name}_attempt{attempt}.log').open('w') as log:
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);item['pid']=p.pid;persist()
    try:code=p.wait(timeout=3600)
    except subprocess.TimeoutExpired:
     os.killpg(p.pid,signal.SIGTERM)
     try:p.wait(timeout=90)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
     code=124
   if code==0 and (out/'result.json').exists():
    item.update(state='complete',finished=time.time());smoke_ok=smoke_ok or name=='smoke5m';break
   item.update(state='retrying' if attempt==1 else 'failed',error=f'exit {code}; {out}/failure.json',finished=time.time());persist();update_report(root)
  persist();update_report(root)
 state.update(state='finished',finished=time.time(),current=None);persist();update_report(root);stop.set()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['queue','run','report']);ap.add_argument('--root',required=True);ap.add_argument('--configuration',choices=list(CONFIGS),default='adaptive');ap.add_argument('--phase-seconds',type=float,default=600);ap.add_argument('--constant-load',type=float);ap.add_argument('--initial',type=int);a=ap.parse_args()
 if a.action=='queue':campaign(a.root);return
 if a.action=='report':update_report(a.root);return
 def terminate(sig,frame):raise RuntimeError(f'terminated {sig}')
 signal.signal(signal.SIGTERM,terminate);e=None
 try:
  e=Run(a.root,a.configuration,a.initial);phases=[dict(load=x,rate=x/(REF_MS/1000),seconds=a.phase_seconds) for x in ([a.constant_load] if a.constant_load is not None else [.7,.85,.7])];e.execute_phases(phases)
 except Exception:save(Path(a.root)/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
 finally:
  if e:e.close()
if __name__=='__main__':main()
