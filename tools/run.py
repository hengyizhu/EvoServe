"""Run one materialized experiment; inherits the allocation, never starts a queue."""
import argparse,os,sys,json,subprocess,shlex
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--gpu',required=True,type=int);p.add_argument('--allowed-gpus',default=None);p.add_argument('--dry-run',action='store_true');p.add_argument('entry');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
 w=Path(a.work).resolve();cfg=json.loads((w/'workspace.json').read_text());entry=(w/a.entry).resolve();assert entry.is_relative_to(w) and entry.is_file()
 env=dict(os.environ);allowed=a.allowed_gpus or str(a.gpu);assert a.gpu in list(map(int,allowed.split(',')))
 visible=env.get('CUDA_VISIBLE_DEVICES')
 if visible and all(x.strip().isdigit() for x in visible.split(',')):
  assert set(map(int,allowed.split(',')))<=set(map(int,visible.split(','))),'GPU outside CUDA_VISIBLE_DEVICES allocation'
 elif visible:raise RuntimeError('UUID/MIG allocations require mapping to node physical ordinals before this legacy-compatible runner; do not override an allocation blindly')
 env.update(VLA_GPU=str(a.gpu),VLA_ALLOWED_GPUS=allowed,VLA_NODE_CPUS=json.dumps(sorted(os.sched_getaffinity(0))),CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(R/'tools')+os.pathsep+env.get('PYTHONPATH',''),LIBERO_CONFIG_PATH=str(w/'vla_followup_20261020/libero_config'),TOKENIZER_DIR=str(w/'vla_followup_20261020/assets/.cache/openpi'),OPENPI_DATA_HOME=str(w/'vla_followup_20261020/assets/.cache/openpi'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 args=a.args[1:] if a.args[:1]==['--'] else a.args;cmd=[cfg['python'],'-u',str(entry)]+args
 if a.dry_run:print(shlex.join(cmd));return
 # Query before a private MPS server changes the worker-visible SM count.
 sm=subprocess.check_output([cfg['python'],'-c','import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)'],env=env,text=True).strip();env['VLA_FULL_SM_COUNT']=str(int(sm))
 os.execvpe(cmd[0],cmd,env)
if __name__=='__main__':main()
