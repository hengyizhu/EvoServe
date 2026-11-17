"""Claim pending learning jobs on GPU0, and GPU1 whenever it becomes free."""
import argparse,csv,fcntl,json,os,subprocess,sys,threading,time,traceback
from pathlib import Path

def read(p):return json.loads(Path(p).read_text()) if Path(p).is_file() else {}
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.{threading.get_ident()}.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)
def gpu_processes(g):
 return subprocess.check_output(['nvidia-smi',f'--id={g}','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
def summarize(root):
 rows=[]
 for p in sorted((root/'task_state').glob('*/points/*.json')):rows.append(read(p))
 if rows:
  with (root/'learning_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# r128 强制两阶段学习','', '225 W；r128/alpha128，lr 2e-4，EMA 0.999，seed08；每点50回合、6进程仿真。机器B对照是用户提供的4000步结果。', '', '| 任务 | 冻结步 | 总步数 | 训练+缓存秒 | 成功率 | B只训动作4000步 |','|---|---:|---:|---:|---:|---:|']
 for x in rows:lines.append(f"| {x['task']} | {x['switch_step']} | {x['step']} | {x['training_plus_cache_seconds']:.1f} | {x['success_rate']:.0%} | {x['machine_b_action_only_final_user_provided']:.0%} |")
 lines+=['','训练用时为实际更新与缓存重建之和，不含加载和评估；诊断探测耗时单列于CSV。评估按固定初始状态、每回合独立随机流执行。','', '队列：']
 state=read(root/'queue_status.json')
 lines += [f"- {j['name']}: {j['state']}"+(f" ({j['error']})" if j.get('error') else '') for j in state.get('jobs',[])]
 (root/'summary.md').write_text('\n'.join(lines)+'\n')

def main(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);(root/'task_state').mkdir(exist_ok=True)
 lockfile=(root/'queue.lock').open('a');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
 jobs=read(root/'jobs.json');prior=read(root/'queue_status.json');byname={j['name']:j for j in prior.get('jobs',[])}
 for j in jobs:
  old=byname.get(j['name'],{})
  j.update(state='complete' if old.get('state')=='complete' else 'failed' if old.get('state')=='failed' else 'pending',attempts=old.get('attempts',0))
 state=dict(state='running',pid=os.getpid(),jobs=jobs,power_limit_w=225,resource_policy='GPU0; GPU1 joins whenever process-free; never touch external processes')
 mutex=threading.RLock();stop=threading.Event()
 def persist():
  with mutex:save(root/'queue_status.json',state);summarize(root)
 def heartbeat():
  while not stop.is_set():
   try:
    proc={str(g):gpu_processes(g) for g in range(4)};snapshot=dict(time=time.time(),processes=proc,gpu_idle={g:not x for g,x in proc.items()},host_load=os.getloadavg(),gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,power.limit,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
    with (root/'heartbeats.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
    with mutex:state.update(heartbeat=time.time(),gpu_idle=snapshot['gpu_idle']);persist()
   except Exception:
    with (root/'heartbeat_errors.log').open('a') as f:f.write(traceback.format_exc())
   stop.wait(1800)
 def worker(gpu):
  empty_since=None
  while True:
   with mutex:
    if not any(j['state'] in ['pending','running','retrying'] for j in jobs):return
    pending=[j for j in jobs if j['state']=='pending']
   if not pending:time.sleep(10);continue
   if gpu_processes(gpu):empty_since=None;time.sleep(10);continue
   if empty_since is None:empty_since=time.monotonic()
   if time.monotonic()-empty_since<30:time.sleep(5);continue
   power=float(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
   if power!=225:
    with mutex:state[f'gpu{gpu}_blocked']=f'Observed power {power} W, expected unchanged225 W';persist()
    time.sleep(30);continue
   with mutex:
    pending=[j for j in jobs if j['state']=='pending']
    if not pending:continue
    job=min(pending,key=lambda j:(j['preferred_gpu']!=gpu,j['priority']));job.update(state='running',gpu=gpu,started=time.time());persist()
   env=os.environ.copy();env.update(VLA_GPU=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu),LEARNING_CPUS=','.join(map(str,range(gpu*16,gpu*16+16))),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
   for k in ['CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE','PI_MPS']:env.pop(k,None)
   jobfile=root/'task_state'/job['name']/'job.json';save(jobfile,job)
   for attempt in range(job['attempts']+1,3):
    with mutex:job.update(attempts=attempt,state='running');persist()
    with (jobfile.parent/f'attempt{attempt}.log').open('a') as log:
     p=subprocess.Popen([sys.executable,'-u',str(root/'learning_job.py'),'--root',str(root),'--job',str(jobfile),'--gpu',str(gpu)],env=env,stdout=log,stderr=subprocess.STDOUT)
     with mutex:job['pid']=p.pid;persist()
     rc=p.wait()
    with mutex:
     if rc==0:job.update(state='complete',finished=time.time());persist();break
     job.update(state='retrying' if attempt==1 else 'failed',error=read(jobfile.parent/'status.json').get('error',f'exit {rc}'));persist()
   with mutex:
    if job['state']=='running':job.update(state='failed',error='Retry budget exhausted before queue restart');persist()
   empty_since=None
 threading.Thread(target=heartbeat,daemon=True).start();persist()
 threads=[threading.Thread(target=worker,args=(g,)) for g in [0,1]]
 for t in threads:t.start()
 for t in threads:t.join()
 state.update(state='complete' if all(j['state']=='complete' for j in jobs) else 'complete_with_failures',finished=time.time());persist();stop.set()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
