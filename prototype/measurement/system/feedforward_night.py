"""Recalibrated feedforward/duty/queue-guard measurements, then two-card learning."""
import argparse,csv,json,os,shutil,subprocess,sys,time,traceback,threading
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import adaptive_campaign as old
from measure import save,read,observe
PY=old.measure.PY

def build_table(prior,service_ms):
 prior=Path(prior);reference=old.REF_MS;points=[]
 cases={.7:[('fixed8_attempt1',8,[0,2]),('fixed1_recovery2',1,[0,2])],.8:[(f'boundary80_m{n}_recovery1',n,[0]) for n in [1,2,4,8]],.85:[('fixed8_attempt1',8,[1]),('fixed1_recovery2',1,[1])]}
 for load,items in cases.items():
  evidence=[]
  for folder,n,stages in items:
   p=prior/folder/'result.json';r=read(p);slo=min(r['stages'][k]['slo_rate'] for k in stages);evidence.append(dict(experts=n,slo_rate=slo,passed=slo>=.99,source=str(p),stages=[k+1 for k in stages],external_load_policy=r.get('external_load_policy')))
  safe=max(x['experts'] for x in evidence if x['passed']);points.append(dict(old_load=load,arrival_rate=load/(reference/1000),new_load=load*service_ms/reference,safe_experts=safe,evidence=evidence))
 return points

class Session(old.Run):
 def __init__(self,out,rank):self.rank=rank;super().__init__(out,'fixed1')
 def env(self,gpu,train=False):
  e=super().env(gpu,train);e.update(ACTION_LORA_RANK=str(self.rank),ACTION_LORA_ALPHA=str(self.rank),ACTION_LR='2e-4' if self.rank==128 else '5e-5');return e

def chart(out):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 rs=list(csv.DictReader((out/'windows_5s.csv').open()));fig,axes=plt.subplots(4,1,figsize=(10,8),sharex=True)
 for ax,key,label in zip(axes,['target_load','merged_experts_at_window_start','duty_at_window_start','p99_ms'],['New load','Merged experts','Duty','P99 (ms)']):
  ax.step([float(r['t_s'])/60 for r in rs],[float(r[key]) if r[key] else float('nan') for r in rs],where='post');ax.set_ylabel(label);ax.grid(alpha=.25)
 axes[-1].axhline(250,color='red',ls='--');axes[-1].set_xlabel('Minutes');fig.tight_layout();fig.savefig(out/'control_timeline.png',dpi=150);fig.savefig(out/'control_timeline.pdf');plt.close(fig)

def run_case(e,root,name,controller,count,loads,seconds,table,reference):
 out=root/'results'/name;out.mkdir(parents=True,exist_ok=False);pair=out/'pair0';pair.mkdir()
 e.command(dict(op='activate',mode='A2',variant=name,discard_previous_variant=True),180)
 e.command(dict(op='calibrate',samples=100),180);cal=read(e.pairs[0]['out']/'calibration.json')
 def command(label,duration,phases,reset):
  cfg=dict(op='run',mode='A2',variant=name,name=f'{name}_{label}',global_rate=phases[0]['rate'],clients=4,duration_s=duration,round=1 if label=='measurement' else 0,rate_cap=0,origin_ns=time.monotonic_ns()+5_000_000_000,phases=phases,reset_expert_states=reset,queue_guard=True,feedforward_protocol=True,controller=controller,initial_experts=count,safe_table=table,reference_service_ms=reference)
  save(out/f'{label}_schedule.json',cfg);e.command(cfg,duration+240);return cfg
 warmrate=.7/(reference/1000);warm=512/warmrate+5
 command('warmup',warm,[dict(load=.7,rate=warmrate,seconds=warm)],True)
 cfg=command('measurement',seconds*len(loads),[dict(load=u,rate=u/(reference/1000),seconds=seconds) for u in loads],True)
 (pair/'measurement').symlink_to(e.pairs[0]['out']/cfg['name'],target_is_directory=True)
 for f in ['host_gpu_monitor.jsonl','compute_process_monitor.jsonl']:(out/f).symlink_to(e.out/f)
 save(out/'config.json',dict(configuration=name,initial_experts=count,rank=e.rank,alpha=e.rank,queue_guard=True,controller=controller,reference_service_ms=reference,external_load_policy=old.EXTERNAL_POLICY,power_limit_w=225,train_cpu=[4,5,6,7],server_cpu=[0,1,2,3]))
 save(out/'calibrations.json',[cal]);r=old.summarize_run(out);r.update(rank=e.rank,alpha=e.rank,controller=controller,queue_guard=True,reference_service_ms=reference)
 save(out/'result.json',r)
 guard_rows=old.rows(pair/'measurement/steps.jsonl');assert all(x['queue_guard']['queue_depth']==0 for x in guard_rows),'A training step began with a non-empty observed inference queue'
 save(out/'queue_guard_validation.json',dict(state='passed',updates=len(guard_rows),all_observed_queues_empty=True,scope='Atomic snapshot immediately before entering the training step; arrivals after the snapshot are not predicted'))
 if controller=='feedforward':chart(out)
 return r

def report(root,state):
 allrows=[];ext=[];totals=[]
 for job in state['systems']:
  p=Path(job.get('result',''))
  if job['state']=='complete' and p.is_file():
   r=read(p);allrows.extend(dict(x,rank=r['rank']) for x in r['stages']);totals.append(dict(case=job['name'],rank=r['rank'],seconds=r['duration_s'],expert_steps=r['total_expert_steps']));ext.append(dict(case=job['name'],**r.get('external_load_summary',{})))
 old.write_csv(root/'system_stages.csv',allrows);old.write_csv(root/'system_totals.csv',totals);old.write_csv(root/'external_load_comparison.csv',ext)
 lines=['# 前馈控制与 r128 两阶段学习','',f"更新：{datetime.now(ZoneInfo('America/New_York')).isoformat()}。225 W；GPU0延迟测量，GPU1/2/3外部负载持续记录。",'','| 组别 | rank | 阶段/新负载 | SLO | P99 ms | 专家步/秒 |','|---|---:|---|---:|---:|---:|']
 for x in allrows:lines.append(f"| {x['configuration']} | {x['rank']} | {x['stage']}/{x['target_load']:.0%} | {x['slo_rate']:.3%} | {x['p99_ms']} | {x['expert_steps_per_s']:.3f} |")
 lines+=['','状态：']+[f"- {j['name']}: {j['state']}" for j in state['systems']]
 lines+=['','学习任务与原始结果见 learning/learning_summary.csv 和 learning/summary.md；机器B数字标为用户提供。所有队列失败记录保留，不修改阈值。']
 (root/'summary.md').write_text('\n'.join(lines)+'\n')

def main(root):
 root=Path(root);state=dict(state='calibrating',pid=os.getpid(),started=time.time(),systems=[],phase='latency');lock=threading.RLock();stop=threading.Event()
 def persist():
  with lock:save(root/'queue_status.json',state);report(root,state)
 def heartbeat():
  while not stop.is_set():
   try:
    snap=observe();snap['processes']={str(g):subprocess.check_output(['nvidia-smi',f'--id={g}','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip() for g in range(4)}
    with (root/'heartbeats.jsonl').open('a') as f:f.write(json.dumps(snap)+'\n')
    with lock:state.update(heartbeat=time.time(),gpu_idle={g:not p for g,p in snap['processes'].items()});persist()
   except Exception:
    with (root/'heartbeat_errors.log').open('a') as f:f.write(traceback.format_exc())
   stop.wait(1800)
 threading.Thread(target=heartbeat,daemon=True).start();persist()
 while not (root/'load_calibration.json').exists():
  if (root/'calibration_failure.json').exists():raise RuntimeError('Calibration failed; see calibration_failure.json')
  time.sleep(5)
 # Calibration result is published before process teardown; wait for the owned GPU to clear.
 reference=read(root/'load_calibration.json')['service_ms'];table=build_table(root.parent/'pi05_adaptive_experts_20261116',reference);save(root/'safe_expert_table.json',dict(points=table,interpolation='minimum of adjacent measured safe counts',outside_range='nearest endpoint count; duty feedback and queue guard remain active',scope='Latest same-bank r32 measurements at70/80/85%; source host conditions retained'))
 tasks=[('r32_smoke','feedforward',1,[.7,.85,.7],100,32),('r32_feedforward_first','feedforward',1,[.7,.85,.7],600,32),('r32_fixed8','fixed',8,[.7,.85,.7],600,32),('r32_fixed1','fixed',1,[.7,.85,.7],600,32),('r32_feedforward_repeat','feedforward',1,[.7,.85,.7],600,32)]
 tasks += [(f'r128_u{int(u*100)}_m{n}','fixed',n,[u],300,128) for u in [.7,.8] for n in [1,2,4,8]]
 e=None;rank=None;generation=0
 try:
  for name,controller,count,loads,seconds,requested_rank in tasks:
   job=dict(name=name,state='pending',rank=requested_rank);state['systems'].append(job);persist()
   if state.get('latency_smoke_failed') and requested_rank==32 and name!='r32_smoke':job.update(state='skipped',error='r32 smoke failed twice');persist();continue
   for attempt in [1,2]:
    job.update(state='running',attempt=attempt,started=time.time());state.update(state='running',phase='latency',current=name);persist()
    try:
     if e is None or rank!=requested_rank:
      if e:e.close();e=None
      generation+=1;rank=requested_rank;e=Session(root/f'runtime_r{rank}_{generation}',rank);e.start()
     result=run_case(e,root,name if attempt==1 else name+'_retry',controller,count,loads,seconds,table,reference)
     job.update(state='complete',finished=time.time(),result=str(Path(result['raw'])/'result.json'));persist();break
    except Exception:
     failure=traceback.format_exc();save(root/f'{name}_failure{attempt}.json',dict(time=time.time(),traceback=failure));job.update(state='retrying' if attempt==1 else 'failed',error=failure.splitlines()[-1]);persist()
     if e:
      try:e.close()
      except Exception:pass
      e=None
   if name=='r32_smoke' and job['state']!='complete':state['latency_smoke_failed']=True
 finally:
  if e:e.close()
 state.update(phase='learning',current=None);persist()
 ready=root/'learning/code_ready.json';deadline=time.monotonic()+3600
 while not ready.exists():
  if time.monotonic()>deadline:raise RuntimeError('Learning implementation not ready after latency phase +1h')
  time.sleep(5)
 with (root/'learning_queue.log').open('a') as log:
  p=subprocess.Popen([PY,'-u',str(root/'learning/learning_queue.py'),'--root',str(root/'learning')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['learning_pid']=p.pid;persist();rc=p.wait()
 state.update(state='finished' if rc==0 else 'finished_with_learning_failure',finished=time.time(),current=None);persist();stop.set()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
 try:main(a.root)
 except Exception:save(Path(a.root)/'fatal_error.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
