import os,sys,time,json,argparse,traceback,gc
from pathlib import Path
R=Path(__file__).resolve().parent;L=Path('/home/hengyi/pi05_lora_lr_20261108');ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();OUT=Path(a.out);cfg=json.loads((L/'selection.json').read_text());family=cfg['family'];lr=cfg['lr']
sys.path.insert(0,str(L));sys.argv=[str(L/'cache_engine.py'),'--mode','A2','--phase','co','--out',str(OUT),'--seed','20261108','--lr',str(lr),'--family',family]
# core_mps.py is deliberately a local copy so the existing import bootstrap works.
from cache_engine import Engine,torch,np,save,append,GPU,stat

def wait(p):
 while not p.exists():time.sleep(.05)
 return json.loads(p.read_text())
def main():
 tr=Engine('A2');tr.schedule='cosine' if family=='lora32_heads' else 'constant';tr.parity();tr.populate();original=[p.detach().cpu().clone() for _,p in tr.pairs];save(OUT/'train_ready.json',{'ready':True});previous_cap=None
 for i in range(50):
  cmd=wait(OUT/f'command_{i:03d}.json')
  if cmd['op']=='stop':break
  if cmd['op']=='calibrate':save(OUT/f'train_done_{i:03d}.json',{'complete':True});continue
  cap=cmd['rate_cap'];out=OUT/cmd['name'];out.mkdir(exist_ok=True)
  if cap!=previous_cap:
   tr.copy.synchronize();tr.stream.synchronize();tr.prefetched=None;tr.mi=tr.step_id=tr.hits=tr.misses=0
   with torch.no_grad():
    for (_,p),m,e,v in zip(tr.pairs,tr.masters,tr.ema,original):p.copy_(v);m.copy_(v.float());e.copy_(v.float())
   tr.optim=torch.optim.AdamW(tr.masters,lr=lr,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,foreach=False);previous_cap=cap;gc.collect()
  origin=cmd['origin_ns'];end=origin+int(cmd['duration_s']*1e9);time.sleep(max(0,(origin-time.monotonic_ns())/1e9));rows=[];last_start=None
  while time.monotonic_ns()<end:
   if cap>0 and last_start is not None:
    delay=last_start+1/cap-time.monotonic()
    if delay>0:time.sleep(delay)
   if time.monotonic_ns()>=end:break
   # Repeat the same finite sample trace if unlimited training passes1000 steps.
   if tr.mi>=len(tr.micros):tr.mi=0;tr.prefetched=None
   last_start=time.monotonic();row=tr.step();rows.append(row);append(out/'steps.jsonl',row)
  valid=[x for x in rows if x['start_ns']>=origin and x['finish_ns']<=end];save(out/'train_summary.json',{'steps_per_s':len(valid)/cmd['duration_s'],'complete_updates':len(valid),'rate_cap':cap,'h2d_ms':stat([x['h2d_ms'] for x in valid])});save(OUT/f'train_done_{i:03d}.json',{'complete':True})
 save(OUT/'train_status.json',{'state':'complete'})
if __name__=='__main__':
 try:main()
 except Exception:save(OUT/'train_status.json',{'state':'failed','traceback':traceback.format_exc()});raise
