import os,sys,argparse,json,time,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;L=Path('/home/hengyi/pi05_lora_lr_20261108');ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--family',required=True);ap.add_argument('--lr',type=float,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--perturb',required=True);a=ap.parse_args();OUT=R/'training'/a.name;OUT.mkdir(parents=True,exist_ok=False)
sequence=R/'sequence_2000_s08.json' if a.steps==2000 else L/'sequences'/f'{a.seed}.json'
sys.path.insert(0,str(L));sys.argv=[str(L/'cache_engine.py'),'--mode','A2','--phase','solo','--out',str(OUT),'--seed',str(a.seed),'--lr',str(a.lr),'--family',a.family,'--sequence',str(sequence),'--perturb',a.perturb]
from cache_engine import Engine,torch,np,save,append,GPU,fingerprint

def main():
 tr=Engine('A2');tr.total_steps=a.steps;tr.schedule='cosine' if a.family=='lora32_heads' else 'constant';frozen=fingerprint((n,p) for n,p in tr.model.named_parameters() if not p.requires_grad);tr.parity();tr.populate();parent=json.loads(Path('/home/hengyi/vla_followup_20261020/audit/parent_download.json').read_text());checkpoints=[100,250,500,1000,2000] if a.steps==2000 else [1000]
 save(OUT/'config.json',{'family':a.family,'lr':a.lr,'schedule':tr.schedule,'EMA':.999,'seed':a.seed,'steps':a.steps,'perturb':json.loads(Path(a.perturb).read_text()),'cache':'complete pinned service-path cache, all training hits','trainable_numel':sum(p.numel() for _,p in tr.pairs),'sequence':str(sequence),'microbatch':4,'accumulation':2,'checkpoints':checkpoints});start=time.monotonic()
 for step in range(a.steps):
  row=tr.step();append(OUT/'steps.jsonl',row)
  if (step+1)%10==0:save(OUT/'status.json',{'state':'training','gpu':GPU,'step':step+1,'elapsed_s':time.monotonic()-start})
  if step+1 in checkpoints:
   payload={'parameters':{n:e.detach().to(dtype=p.dtype).cpu() for (n,p),e in zip(tr.pairs,tr.ema)},'family':a.family,'seed':a.seed,'step':step+1,'ema_decay':.999,'lr':a.lr,'parent':parent,'weights_kind':'EMA','perturb':json.loads(Path(a.perturb).read_text())};path=OUT/f'ema_{step+1}.pt';tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path);save(OUT/f'checkpoint_{step+1}.json',{'state':'ready','path':str(path)});del payload
 assert tr.hits==a.steps*8 and tr.misses==0;assert fingerprint((n,p) for n,p in tr.model.named_parameters() if not p.requires_grad)==frozen;save(OUT/'status.json',{'state':'complete','gpu':GPU,'steps':a.steps,'frozen_unchanged':True,'cache_hits':tr.hits})
if __name__=='__main__':
 try:main()
 except Exception:save(OUT/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
