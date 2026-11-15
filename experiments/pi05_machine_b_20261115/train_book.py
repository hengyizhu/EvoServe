"""Cached action training, fixed-step EMA evaluation, and verified curve extension."""
import argparse,os,sys,json,time,traceback,shutil
from pathlib import Path
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
for key in ['name','family','split','sequence','perturb']:ap.add_argument('--'+key,required=True)
ap.add_argument('--lr',required=True,type=float);ap.add_argument('--seed',required=True,type=int);ap.add_argument('--steps',required=True,type=int);ap.add_argument('--schedule-horizon',type=int);ap.add_argument('--checkpoints',required=True);ap.add_argument('--verify-2000')
a=ap.parse_args();OUT=R/'training'/a.name;OUT.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(R));sys.argv=[str(R/'cache_engine.py'),'--mode','A2','--phase','solo','--out',str(OUT),'--seed',str(a.seed),'--lr',str(a.lr),'--family',a.family,'--sequence',a.sequence,'--perturb',a.perturb,'--split',a.split]
from cache_engine import Engine,torch,save,append,GPU,fingerprint

def main():
 tr=Engine('A2');assert len(tr.sequence)==a.steps;tr.total_steps=a.schedule_horizon or a.steps;tr.schedule='cosine' if a.family=='lora32_heads' else 'constant'
 frozen=fingerprint((n,p) for n,p in tr.model.named_parameters() if not p.requires_grad)
 cfg={'family':a.family,'lr':a.lr,'schedule':tr.schedule,'warmup':50,'cosine_min_fraction':.1,'EMA':.999,'seed':a.seed,'steps':a.steps,'cosine_horizon':tr.total_steps,'perturb':json.loads(Path(a.perturb).read_text()),'split':a.split,'sequence':a.sequence,'cache':'service B1 prefix; pinned host; microbatch prefetch','augmentation':False,'microbatch':4,'accumulation':2,'trainable_numel':sum(p.numel() for _,p in tr.pairs),'checkpoints':[int(s) for s in a.checkpoints.split(',')],'verify_2000':a.verify_2000}
 assert cfg['trainable_numel']==(430098464 if a.family=='full' else 20876320)
 save(OUT/'config.json',cfg);save(OUT/'trainable_modules.json',[{'name':n,'numel':p.numel(),'dtype':str(p.dtype)} for n,p in tr.pairs]);tr.parity();tr.populate();start=time.monotonic();parent=json.loads(Path('/home/hengyi/vla_followup_20261020/audit/parent_download.json').read_text())
 for step in range(a.steps):
  row=tr.step();append(OUT/'steps.jsonl',row)
  if (step+1)%10==0:save(OUT/'status.json',{'state':'training','gpu':GPU,'step':step+1,'total_steps':a.steps,'elapsed_s':time.monotonic()-start})
  if step+1==2000 and a.verify_2000:
   prior=torch.load(a.verify_2000,map_location='cpu',weights_only=True,mmap=True);assert prior['step']==2000 and prior['seed']==a.seed
   assert set(prior['parameters'])=={n for n,p in tr.pairs};mismatches=[]
   for (n,p),e in zip(tr.pairs,tr.ema):
    if not torch.equal(e.detach().to(dtype=p.dtype).cpu(),prior['parameters'][n]):mismatches.append(n)
   save(OUT/'extension_equivalence.json',{'state':'passed' if not mismatches else 'failed','reference':a.verify_2000,'step':2000,'bitwise_equal':not mismatches,'mismatched_parameters':mismatches});del prior
   assert not mismatches,'EMA at2000 differs; refuse to splice old and new learning curves'
  if step+1 in cfg['checkpoints']:
   assert shutil.disk_usage(OUT).free>12*2**30,'checkpoint filesystem below12GiB'
   payload={'parameters':{n:e.detach().to(dtype=p.dtype).cpu() for (n,p),e in zip(tr.pairs,tr.ema)},'family':a.family,'seed':a.seed,'step':step+1,'ema_decay':.999,'lr':a.lr,'parent':parent,'weights_kind':'EMA','perturb':cfg['perturb'],'eval_task_id':json.loads(Path(a.split).read_text())['eval_task_id']}
   dest=OUT/f'ema_{step+1}.pt';tmp=dest.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(dest);save(OUT/f'checkpoint_{step+1}.json',{'state':'ready','path':str(dest)});del payload
 assert tr.hits==a.steps*8 and tr.misses==0
 assert fingerprint((n,p) for n,p in tr.model.named_parameters() if not p.requires_grad)==frozen
 save(OUT/'status.json',{'state':'complete','gpu':GPU,'steps':a.steps,'cosine_horizon':tr.total_steps,'cache_hits':tr.hits,'frozen_unchanged':True,'training_seconds':time.monotonic()-start})
if __name__=='__main__':
 try:main()
 except Exception:save(OUT/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
