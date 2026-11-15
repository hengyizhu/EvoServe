"""Learning-only migration check: first1000 updates of the reference4000-step schedule."""
from pathlib import Path
import os,sys,json,subprocess,argparse,time
R=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--threads',type=int,default=2);p.add_argument('--cpus',help='comma-separated physical CPU IDs; omitted: inherited CPU allocation');p.add_argument('--name',default='machine_b_book_s08_1000');p.add_argument('--dry-run',action='store_true');a=p.parse_args();w=Path(a.work).resolve();N=w/'pi05_machine_b_20261115';E=w/'pi05_extended_20261109';ref=json.loads((R/'machine_b/book_reference.json').read_text())
if a.cpus:
 chosen={int(x) for x in a.cpus.split(',')};assert chosen<=os.sched_getaffinity(0);os.sched_setaffinity(0,chosen)
env=dict(os.environ,VLA_CPU_POLICY='shared',VLA_TORCH_THREADS=str(a.threads),OMP_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads))
for k in list(env):
 if k.startswith('CUDA_MPS_') or k=='PI_MPS':env.pop(k)
seq=json.loads((E/'sequences/book_20261108_4000.json').read_text());assert len(seq)==4000
(N/'sequences').mkdir(exist_ok=True);(N/'training').mkdir(exist_ok=True)
for name,n in [('book_1000',1000),('book_preflight',1)]:(N/'sequences'/f'{name}.json').write_text(json.dumps(seq[:n]))
base=[sys.executable,str(R/'tools/run.py'),'--work',str(w),'--gpu',str(a.gpu)]
def run(entry,args):
 cmd=base+(['--dry-run'] if a.dry_run else [])+[entry]+list(map(str,args));print('COMMAND',cmd,flush=True);subprocess.run(cmd,env=env,check=True)
def train(name,n,sequence):
 run('pi05_machine_b_20261115/train_book.py',['--name',name,'--family','lora32_heads','--lr','5e-5','--seed',20261108,'--steps',n,'--schedule-horizon',4000,'--split',E/'splits/book.json','--sequence',N/'sequences'/sequence,'--perturb',E/'tasks/book_0.json','--checkpoints',n])
def evaluate(name,n,episodes):
 run('pi05_machine_b_20261115/eval_worker.py',['--name',name+f'_c{n}','--delta',N/'training'/name/f'ema_{n}.pt','--task',5,'--adapt-task',5,'--perturb',E/'tasks/book_0.json','--end',episodes,'--resume'])
# Failed execution retries use a new run name; no successful rounds silently mixed.
for attempt in [1,2]:
 name=a.name+f'_attempt{attempt}'
 try:
  train(name+'_preflight',1,'book_preflight.json');evaluate(name+'_preflight',1,1)
  train(name,1000,'book_1000.json');evaluate(name,1000,50)
  if a.dry_run:break
  rows=[json.loads(s) for s in (N/'results'/f'{name}_c1000.jsonl').read_text().splitlines()];assert len(rows)==50 and len({x['index'] for x in rows})==50
  success=sum(x['success'] for x in rows);power=subprocess.check_output(['nvidia-smi','--query-gpu=index,power.limit','--format=csv'],text=True)
  result=dict(state='complete',successes=success,episodes=50,success_rate=success/50,reference_successes=7,reference_episodes=50,difference_pp=100*(success/50-.14),cosine_horizon=4000,updates=1000,reference=ref,power_limits=power,cpu_affinity=sorted(os.sched_getaffinity(0)),quality_threshold=None,note='Report difference; no new quality pass threshold invented.')
  (N/(a.name+'_summary.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));break
 except subprocess.CalledProcessError:
  if attempt==2:raise
