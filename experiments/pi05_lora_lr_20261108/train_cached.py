import os,sys,argparse,json,time,gc,traceback,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--screen',action='store_true');ap.add_argument('--lr',type=float);ap.add_argument('--job',required=True);a=ap.parse_args();JOB=R/'jobs'/a.job;JOB.mkdir(parents=True,exist_ok=False)
sys.argv=[str(R/'cache_engine.py'),'--mode','A2','--phase','solo','--out',str(JOB),'--seed',str(a.seed),'--lr',str(a.lr or 1e-4)]
from cache_engine import Engine,torch,np,save,append,GPU,fingerprint

def tag(lr):return {5e-5:'5e-5',1e-4:'1e-4',2e-4:'2e-4'}[lr]
def main():
 save(JOB/'status.json',{'state':'loading','gpu':GPU});tr=Engine('A2');initial=[p.detach().cpu().clone() for _,p in tr.pairs];frozen=fingerprint((n,p) for n,p in tr.model.named_parameters() if not p.requires_grad)
 actual={n:p.numel() for n,p in tr.pairs};dense={n:v for n,v in actual.items() if not 'lora_' in n};assert set(n.split('.')[0] for n in dense)=={'action_in_proj','action_out_proj','time_mlp_in','time_mlp_out'};assert sum(actual.values())==20876320
 save(JOB/'module_audit.json',{'family':'lora32_heads','rank':32,'alpha':32,'dropout':0,'layer_ids':list(range(18)),'lora_targets':'expert1 Q/K/V/O; FFN gate/up/down; all37 AdaRMS ada_modulation linear modules','dense_trainable':dense,'trainable_numel':sum(actual.values()),'parameters':actual,'adapter_modules':[n for n,m in tr.model.named_modules() if hasattr(m,'lora_A')],'ffn_adapter_modules':[n for n,m in tr.model.named_modules() if hasattr(m,'w_gating_lora_a')]})
 tr.parity();tr.populate();parent=json.loads(Path('/home/hengyi/vla_followup_20261020/audit/parent_download.json').read_text())
 rates=[5e-5,1e-4,2e-4] if a.screen else [a.lr]
 for lr in rates:
  name=f'lr{tag(lr)}_s{a.seed}';out=R/'training'/name;out.mkdir(parents=True,exist_ok=False);tr.copy.synchronize();tr.stream.synchronize();tr.prefetched=None;tr.mi=0;tr.step_id=0;tr.hits=tr.misses=0;tr.learning_rate=lr
  with torch.no_grad():
   for (_,p),m,e,v in zip(tr.pairs,tr.masters,tr.ema,initial):p.copy_(v);m.copy_(v.float());e.copy_(v.float())
  tr.model.zero_grad(set_to_none=True);tr.optim=torch.optim.AdamW(tr.masters,lr=lr,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,foreach=False);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.monotonic();rows=[]
  save(out/'config.json',{'family':'lora32_heads','seed':a.seed,'lr':lr,'EMA':.999,'schedule':'50-step linear warmup then cosine to 0.1 peak','rank':32,'alpha':32,'augmentation':False,'brightness':.05,'microbatch':4,'accumulation':2,'steps':1000,'prefix':'full precomputed frozen B1 service-path cache in pinned host memory; same path/sequence for all LRs','prefix_population_seconds':json.loads((JOB/'population.json').read_text())['seconds'],'trainable_numel':sum(actual.values()),'sequence':str(R/'sequences'/f'{a.seed}.json'),'initial_trainable_sha256':hashlib.sha256(b''.join(v.contiguous().view(torch.uint8).numpy().tobytes() for v in initial)).hexdigest()})
  for step in range(1000):
   row=tr.step();rows.append(row);append(out/'steps.jsonl',row)
   if (step+1)%10==0:
    status={'state':'training','name':name,'seed':a.seed,'lr':lr,'gpu':GPU,'step':step+1,'elapsed_s':time.monotonic()-started};save(out/'status.json',status);save(JOB/'status.json',status)
  elapsed=time.monotonic()-started;assert tr.hits==8000 and tr.misses==0
  payload={'parameters':{n:e.detach().to(dtype=p.dtype).cpu() for (n,p),e in zip(tr.pairs,tr.ema)},'family':'lora32_heads','seed':a.seed,'step':1000,'ema_decay':.999,'lr':lr,'brightness':.05,'parent':parent,'weights_kind':'EMA'};path=out/'ema_1000.pt';tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path);save(out/'checkpoint_1000.json',{'state':'ready','path':str(path),'step':1000});del payload
  summary={'state':'complete','gpu':GPU,'steps':1000,'seconds':elapsed,'steps_per_s':1000/elapsed,'step_median_ms':float(np.median([x['seconds'] for x in rows[10:]]))*1000,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'cache_hits':tr.hits,'cache_misses':tr.misses};save(out/'status.json',summary);save(out/'cost_summary.json',summary)
 assert fingerprint((n,p) for n,p in tr.model.named_parameters() if not p.requires_grad)==frozen
 save(JOB/'status.json',{'state':'complete','gpu':GPU,'seed':a.seed,'rates':rates,'frozen_unchanged':True})
if __name__=='__main__':
 try:main()
 except Exception:save(JOB/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
