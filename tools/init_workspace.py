"""Materialize relocatable scripts without changing the archived source snapshot."""
import argparse,json,os,re,shutil,sys,hashlib
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
def materialize(work,model,data,python):
 work=Path(work).resolve();model=Path(model).resolve();data=Path(data).resolve();work.mkdir(parents=True,exist_ok=True)
 if (work/'workspace.json').exists():raise FileExistsError('Use a new workspace; refusing to overwrite an experiment')
 mappings={'/home/hengyi/vla_followup_20261020/venv/bin/python':str(python),'/home/hengyi/.cache/huggingface/lerobot/physical-intelligence/libero':str(data),'/var/tmp/hengyi_pi05_extended_20261109':str(work/'pi05_extended_20261109')}
 for src in (REPO/'experiments').iterdir():
  mappings['/home/hengyi/'+src.name]=str(work/src.name)
  for parent in ['Experiments','Environment-and-Earlier-Runs']:
   mappings['/home/hengyi/Workspace/01_VLA-Serving/'+parent+'/'+src.name]=str(work/src.name)
 mappings['/home/hengyi/Workspace/05_Models-and-Dependencies/huggingface/lerobot/physical-intelligence/libero']=str(data)
 changes=[]
 for src in (REPO/'experiments').rglob('*'):
  if not src.is_file():continue
  dst=work/src.relative_to(REPO/'experiments');dst.parent.mkdir(parents=True,exist_ok=True);original=src.read_text();text=original
  for old,new in sorted(mappings.items(),key=lambda kv:-len(kv[0])):text=text.replace(old,new)
  # A6000's 84 SMs are not a portable MPS validation threshold.
  if src.suffix=='.py':
   text=re.sub(r'(?P<op><|>=)84\b',lambda m:m['op']+"int(__import__('os').environ['VLA_FULL_SM_COUNT'])",text)
   if 'os.sched_setaffinity(' in text:
    text=text.replace('os.sched_setaffinity(','__import__("vla_portability").set_affinity(')
   text=re.sub(r"torch\.set_num_threads\(4\)", "torch.set_num_threads(int(__import__('os').environ.get('VLA_TORCH_THREADS','4')))",text)
   text=re.sub(r"(\b(?:OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS)\s*=\s*)[\"']4[\"']",lambda m:m[1]+"__import__('os').environ.get('VLA_TORCH_THREADS','4')",text)
   text=text.replace('assert GPU in [0,1]','assert GPU in __import__("vla_portability").allowed_gpus()')
  dst.write_text(text)
  if text!=original:changes.append(str(dst.relative_to(work)))
 for name in ['vla_followup_20261020','vla_ladder_20261020']:
  base=work/name
  (base/'parent').symlink_to(model,target_is_directory=True)
  (base/'RLinf').symlink_to(REPO/'vendor/RLinf',target_is_directory=True)
  (base/'venv').symlink_to(Path(python).parent.parent,target_is_directory=True)
  for d in ['audit','benchmark','results','logs','training','libero_config','assets/.cache/openpi/big_vision']:(base/d).mkdir(parents=True,exist_ok=True)
 shutil.copy2(work/'vla_followup_20261020/audit/parent_download.json',work/'vla_ladder_20261020/audit/parent_download.json')
 # Historical scripts sometimes create child directories without parents=True.
 for p in (REPO/'experiments').iterdir():
  for name in ['logs','results','jobs']:(work/p.name/name).mkdir(exist_ok=True)
 for name in ['pi05_measure_20261107','pi05_optimized_20261107']:
  for d in ['training','inference','training_mb4']:(work/name/d).mkdir(exist_ok=True)
 (work/'workspace.json').write_text(json.dumps({'repo':str(REPO),'work':str(work),'model':str(model),'data':str(data),'python':str(python),'materialized_changes':changes},indent=2))
 return work
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--model',required=True);p.add_argument('--data',required=True);a=p.parse_args();print(materialize(a.work,a.model,a.data,Path(sys.executable).absolute()))
