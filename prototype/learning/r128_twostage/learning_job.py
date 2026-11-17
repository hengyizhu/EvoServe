"""Smoke, forced visual warm-start, cached continuation, then fixed-point evaluation."""
import argparse,csv,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from learning_queue import read,save,summarize

def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).exists() else []
def main(a):
 root=Path(a.root);job=read(a.job);out=root/'task_state'/job['name'];out.mkdir(parents=True,exist_ok=True)
 cfg=read(root/'configs'/f"{job['name']}.json")
 cachebase=Path('/var/tmp' if a.gpu==0 else '/dev/shm')/'hengyi_pi05_r128_20261116/prefix_store';cachebase.mkdir(parents=True,exist_ok=True)
 # Preserve the cache location on a retry, even when a different GPU claims it.
 configpath=out/'config.json'
 if configpath.exists():cfg=read(configpath)
 else:cfg['cache_dir']=str(cachebase/job['name']);save(configpath,cfg)
 def run(cmd,label):
  with (out/f'{label}.log').open('a') as log:
   p=subprocess.Popen([sys.executable,'-u',*cmd],cwd=root,stdout=log,stderr=subprocess.STDOUT)
   save(out/'status.json',dict(state='running',phase=label,gpu=a.gpu,pid=p.pid,time=time.time()));rc=p.wait()
  if rc:raise RuntimeError(f'{label}: exit {rc}; see {out/ (label+".log")}')
 def train(c,path,smoke=False,dest=None):
  if not path.exists():save(path,c)
  else:assert read(path)==c
  cmd=[str(root/'train_stage.py'),'--config',str(path)]
  if dest:cmd+=['--out',str(dest)]
  if smoke:cmd+=['--smoke']
  run(cmd,'train_'+c['name']);return dest or root/'training'/c['name']
 def evaluate(delta,result,episodes,smoke=False):
  cmd=[str(root/'eval_parallel.py'),'--delta',str(delta),'--task',str(cfg['eval_task']),'--gpu',str(a.gpu),'--perturb',cfg['perturb'],'--out',str(result),'--workers','6','--episodes',str(episodes)]
  if smoke:cmd+=['--smoke']
  run(cmd,'eval_'+result.stem)
 if not read(out/'smoke.json').get('passed'):
  sc=dict(cfg,name=job['name']+'_smoke_visual',cache_dir=str(cachebase/(job['name']+'_smoke_visual')))
  sp=train(sc,out/'smoke_visual_config.json',True,out/'smoke_visual')
  evaluate(sp/'ema_1.pt',out/'smoke_eval.jsonl',1,True)
  cc=dict(sc,name=job['name']+'_smoke_cached',vision=False,adaptive_switch=False,force_visual=False,switch_step=1,source_state=str(sp/'state_1.pt'),source_name=sc['name'],cache_dir=str(cachebase/(job['name']+'_smoke_cached')))
  cp=train(cc,out/'smoke_cached_config.json',True,out/'smoke_cached')
  assert read(cp/'switch_validation.json')['optimizer_and_action_EMA_preserved'] and read(cp/'parity.json')['gradient_relative_l2']<1e-6
  save(out/'smoke.json',dict(passed=True,visual_updates=1,cached_updates=1,full_eval_episodes=1,simulator_processes=6,time=time.time()))
  for c in [sc,cc]:
   p=Path(c['cache_dir']);assert p.parent==cachebase
   if p.exists():shutil.rmtree(p)
 visual=train(cfg,configpath);decision=read(visual/'switch_decision.json');assert 'triggered' in decision
 switch=decision['step'] if decision['triggered'] else None;final=visual
 if switch is not None and switch<4000:
  cc=dict(cfg,name=job['name']+'_cached',vision=False,cache=True,adaptive_switch=False,force_visual=False,switch_step=switch,source_state=str(visual/f'state_{switch}.pt'),source_name=cfg['name'],cache_dir=cfg['cache_dir']+'_cached',record_action_probes=False)
  # Worst-case BF16 KV footprint plus filesystem overhead; protect other data.
  seq=read(cfg['sequence']);remaining={tuple(k) for batch in seq[switch:] for k in batch};needed=len(remaining)*18*1024**2
  cache=Path(cc['cache_dir']);cache.mkdir(parents=True,exist_ok=True);existing=sum(p.stat().st_size for p in cache.glob('*.pt'));free=shutil.disk_usage(cache).free
  assert free+existing>needed+2*1024**3,f'Insufficient own cache space: need {needed}, free {free}, existing {existing}'
  save(out/'cache_capacity.json',dict(unique_remaining=len(remaining),reserved_estimate_bytes=needed,free_bytes=free,existing_bytes=existing,path=str(cache)))
  final=train(cc,out/'cached_config.json')
 for step in [1000,2000,4000]:
  phase=visual if switch is None or step<=switch else final;ck=phase/f'ema_{step}.pt';assert ck.is_file(),ck
  result=out/'evaluations'/f'step_{step}.jsonl';evaluate(ck,result,50)
  rs=rows(result);assert len(rs)==50 and {r['index'] for r in rs}==set(range(50))
  visual_rows=[r for r in rows(visual/'steps.jsonl') if r['step']<=step];action_rows=[r for r in rows(final/'steps.jsonl') if r['step']<=step] if final!=visual else []
  pop=sum(r['seconds'] for r in rows(final/'population_history.jsonl') if r['start_step']<step) if final!=visual else 0
  diagnostics=list(csv.DictReader((visual/'visual_metrics.csv').open()));diagnostic_s=sum(float(r['diagnostic_seconds']) for r in diagnostics if int(r['step'])<=step)
  train_s=sum(r['seconds'] for r in visual_rows+action_rows)
  point=dict(task=job['task'],method='forced_two_stage_r128',seed=cfg['seed'],step=step,switch_step=switch,training_plus_cache_seconds=train_s+pop,training_updates_seconds=train_s,cache_rebuild_seconds=pop,diagnostic_seconds=diagnostic_s,training_cache_diagnostics_seconds=train_s+pop+diagnostic_s,successes=sum(r['success'] for r in rs),episodes=50,success_rate=sum(r['success'] for r in rs)/50,machine_b_action_only_final_user_provided=job['machine_b_final'],rank=128,alpha=128,learning_rate=2e-4,peak_cuda_bytes=max(r['peak_cuda_bytes'] for r in visual_rows+action_rows),power_limit_w=225,raw=str(result))
  save(out/'points'/f'{step}.json',point);summarize(root)
 save(out/'status.json',dict(state='complete',switch=decision,gpu=a.gpu,finished=time.time()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--job',required=True);p.add_argument('--gpu',required=True,type=int);a=p.parse_args()
 try:main(a)
 except Exception:
  job=read(a.job);save(Path(a.root)/'task_state'/job['name']/'status.json',dict(state='failed',error=traceback.format_exc(),time=time.time()));raise
