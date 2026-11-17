"""State-preserving visual warm-start followed by frozen-prefix cached action adaptation."""
import argparse,sys,os,time,json,math,shutil,copy,csv
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out');ap.add_argument('--smoke',action='store_true');ap.add_argument('--benchmark',action='store_true');ap.add_argument('--until',type=int);args=ap.parse_args()
sys.argv=[sys.argv[0],'--config',args.config]+(['--out',args.out] if args.out else [])+(['--smoke'] if args.smoke or args.benchmark else [])
import engine_base as b
from engine_base import torch,np,tree_map,save,append,fingerprint,OUT,CACHE,cfg,GPU
R=Path(__file__).resolve().parent

def snapshot(tr):
 return {'step':tr.step_id,'names':[n for n,p in tr.pairs],'masters':[x.detach().cpu() for x in tr.masters],'ema':[x.detach().cpu() for x in tr.ema],'optimizer':tree_map(lambda x:x.detach().cpu() if torch.is_tensor(x) else x,tr.optim.state_dict()),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'config':cfg,'frozen_vision':{n:p.detach().cpu() for n,p in tr.model.named_parameters() if n.startswith('img.') and n.endswith(('.A','.B')) and not p.requires_grad}}
def atomic_state(path,state):
 tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
def freeze(tr):
 keep=[i for i,(n,p) in enumerate(tr.pairs) if not n.startswith('img.')]
 for n,p in tr.pairs:
  if n.startswith('img.'):p.requires_grad_(False)
 for i,m in enumerate(tr.masters):
  if i not in keep:tr.optim.state.pop(m,None)
 tr.pairs=[tr.pairs[i] for i in keep];tr.masters=[tr.masters[i] for i in keep];tr.ema=[tr.ema[i] for i in keep];tr.optim.param_groups[0]['params']=tr.masters
 tr.model.eval();tr.model.gradient_checkpointing_disable();tr.vision=False;tr.mode='A2'
 assert tr.pairs and all(not n.startswith('img.') for n,p in tr.pairs)
 assert sum(p.numel() for n,p in tr.pairs)>20876320 if cfg['family']=='lora128_vision16' else sum(p.numel() for n,p in tr.pairs)==20876320
 assert not any(p.requires_grad for p in tr.model.img.parameters())
def restore(tr,state):
 if state.get('frozen_vision'):
  tr.model.load_state_dict(state['frozen_vision'],strict=False);freeze(tr)
 assert [n for n,p in tr.pairs]==state['names']
 tr.optim.load_state_dict(state['optimizer'])
 with torch.no_grad():
  for (n,p),m,e,sm,se in zip(tr.pairs,tr.masters,tr.ema,state['masters'],state['ema']):m.copy_(sm);e.copy_(se);p.copy_(m)
 tr.step_id=state['step'];tr.mi=2*tr.step_id;tr.prefetched=None
 torch.set_rng_state(state['rng']);torch.cuda.set_rng_state(state['cuda_rng'])
def persist(tr,step):
 parameters={n:e.detach().to(dtype=p.dtype).cpu() for (n,p),e in zip(tr.pairs,tr.ema)}
 parameters.update({n:p.detach().cpu() for n,p in tr.model.named_parameters() if n.startswith('img.') and n.endswith(('.A','.B')) and not p.requires_grad})
 parent=json.loads(Path('/home/hengyi/Workspace/01_VLA-Serving/Environment-and-Earlier-Runs/vla_followup_20261020/audit/parent_download.json').read_text())
 payload={'parameters':parameters,'family':cfg['family'],'seed':cfg['seed'],'step':step,'ema_decay':.999,'lr':cfg['lr'],'parent':parent,'weights_kind':'EMA','frozen_visual_weights':('parent_restored' if cfg.get('reuse_parent_cache') else 'raw_at_switch') if not tr.vision else None,'perturb':json.loads(Path(cfg['perturb']).read_text()),'eval_task_id':cfg['eval_task']}
 atomic_state(OUT/f'ema_{step}.pt',payload)
 save(OUT/f'checkpoint_{step}.json',{'state':'ready','path':str(OUT/f'ema_{step}.pt')})
def checkpoint_mode(tr,on):
 if on:tr.model.gradient_checkpointing_enable({'use_reentrant':False})
 else:tr.model.gradient_checkpointing_disable()
 tr.model.llm.train();tr.model.img.encoder.train()
 assert bool(tr.model.llm.gradient_checkpointing)==on and bool(tr.model.img.encoder.gradient_checkpointing)==on

def benchmark(tr):
 # Same initial parameters, optimizer, samples and RNG in both conditions.
 state=snapshot(tr);tr.micros=tr.micros[:2]*20;results=[]
 for on in [True,False]:
  restore(tr,state);checkpoint_mode(tr,on);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[]
  for i in range(15):
   row=tr.step()
   if i>=5:times.append(row['seconds'])
  results.append({'checkpointing':on,'warmup':5,'measured_steps':10,'median_step_s':float(np.median(times)),'mean_step_s':float(np.mean(times)),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'raw_seconds':times})
 save(OUT/'checkpointing_benchmark.json',{'conditions':results,'speedup':results[0]['median_step_s']/results[1]['median_step_s'],'same_initial_state':True})
 save(OUT/'status.json',{'state':'complete','benchmark':True})

def main():
 target=min(cfg['stop_after'],args.until or cfg['stop_after'])
 adaptive=bool(cfg.get('adaptive_switch')) and not args.smoke
 prior_decision=json.loads((OUT/'switch_decision.json').read_text()) if (OUT/'switch_decision.json').exists() else None
 if adaptive and prior_decision and prior_decision.get('triggered'):target=prior_decision['step']
 if not args.smoke and not args.benchmark and (OUT/'status.json').exists():
  prior=json.loads((OUT/'status.json').read_text())
  if prior.get('state') in ['complete','segment_complete'] and prior.get('steps',0)>=target and (OUT/f'ema_{target}.pt').exists():return
 start_all=time.monotonic();tr=b.Engine();checkpoint_mode(tr,cfg.get('checkpointing',False))
 parent_visual={n:p.detach().cpu().clone() for n,p in tr.model.named_parameters() if n.startswith('img.') and n.endswith(('.A','.B'))}
 parent_visual_sha=fingerprint(tr.model.img.named_parameters())
 if args.benchmark:return benchmark(tr)
 resumed=False
 if (OUT/'resume.pt').exists():
  state=torch.load(OUT/'resume.pt',map_location='cpu',weights_only=False);assert state['config']==cfg;restore(tr,state);del state;resumed=True
 elif cfg.get('source_state'):
  while not Path(cfg['source_state']).exists():
   save(OUT/'status.json',{'state':'inputs_ready_waiting_for_visual_checkpoint','gpu':GPU,'switch_step':cfg['switch_step']})
   parent_status=Path(cfg['source_state']).parent/'status.json'
   producer=json.loads(parent_status.read_text()) if parent_status.exists() else {}
   assert producer.get('state') not in ['failed','complete'],'visual checkpoint missing after producer terminated'
   time.sleep(5)
  state=torch.load(cfg['source_state'],map_location='cpu',weights_only=False);restore(tr,state);assert tr.step_id==cfg['switch_step']
  action_before=fingerprint((n,p) for n,p in tr.pairs if not n.startswith('img.'))
  if cfg.get('restore_parent_vision'):
   tr.model.load_state_dict(parent_visual,strict=False)
   assert fingerprint(tr.model.img.named_parameters())==parent_visual_sha
   save(OUT/'parent_visual_restoration.json',{'state':'passed','parent_visual_sha256':parent_visual_sha,'restored_A_and_B':True,'all_B_zero':all(not bool(p.count_nonzero()) for n,p in tr.model.named_parameters() if n.startswith('img.') and n.endswith('.B'))})
  freeze(tr);assert fingerprint(tr.pairs)==action_before;del state
  save(OUT/'switch_validation.json',{'state':'passed','step':tr.step_id,'optimizer_and_action_EMA_preserved':True,'frozen_visual_sha256':fingerprint(tr.model.img.named_parameters()),'frozen_parameters':sum(p.numel() for p in tr.model.img.parameters()),'trainable_parameters':sum(p.numel() for n,p in tr.pairs)})
 if not tr.vision and not cfg.get('source_state') and not resumed:freeze(tr)
 if cfg.get('reuse_parent_cache'):
  assert fingerprint(tr.model.img.named_parameters())==parent_visual_sha,'parent vision not restored'
 if cfg.get('recompute_action'):tr.mode='A0'
 if not tr.vision:
  assert not any(p.requires_grad for p in tr.model.img.parameters());tr.keys=list(dict.fromkeys(tuple(k) for m in tr.micros[tr.mi:] for k in m))
 frozen=fingerprint((n,p) for n,p in tr.model.named_parameters() if not p.requires_grad)
 save(OUT/'trainable_modules.json',[dict(name=n,numel=p.numel(),dtype=str(p.dtype)) for n,p in tr.pairs])
 save(OUT/'config.json',dict(cfg,trainable_parameters=sum(p.numel() for n,p in tr.pairs),actual_checkpointing=bool(tr.model.llm.gradient_checkpointing),parallel_learning=True))
 if tr.vision:tr.vision_check()
 elif not cfg.get('recompute_action'):
  identity={'frozen_prefix_sha256':fingerprint((n,p) for n,p in tr.model.named_parameters() if n.startswith('img.') or (n.startswith('llm.') and not p.requires_grad)),'perturb':json.loads(Path(cfg['perturb']).read_text()),'switch_step':cfg['switch_step'],'source_state':cfg['source_state']}
  if (CACHE/'identity.json').exists():assert json.loads((CACHE/'identity.json').read_text())==identity
  else:save(CACHE/'identity.json',identity)
  save(OUT/'cache_identity.json',identity)
  # Use the actual next microbatch for cache/recompute gradient equivalence.
  original=tr.micros;tr.micros=tr.micros[tr.mi:]+tr.micros[:tr.mi];tr.parity();tr.micros=original;tr.pinned.clear()
  tr.populate()
  population=json.loads((OUT/'population.json').read_text());append(OUT/'population_history.jsonl',dict(population,start_step=tr.step_id,session_started_ns=time.time_ns()))
 if args.smoke:
  # Smoke inputs contain repeated samples; local positions prevent large source offset.
  tr.mi=0;tr.micros=tr.micros[:2]*3
 first=tr.step_id;total=first+1 if args.smoke else target;tr.prefetched=None
 log=OUT/'steps.jsonl'
 if log.exists():
  old=log.read_text().splitlines();valid=[]
  for line in old:
   try:row=json.loads(line)
   except json.JSONDecodeError:break
   if row['step']<=first:valid.append(line)
  if len(valid)!=len(old):log.rename(OUT/f'steps_uncommitted_{time.time_ns()}.jsonl');log.write_text('\n'.join(valid)+('\n' if valid else ''))
 diagnostics=None
 if tr.vision or cfg.get('record_action_probes'):
  from visual_metrics import VisualMetrics
  diagnostics=VisualMetrics(tr,OUT,cfg)
 torch.cuda.reset_peak_memory_stats()
 for step in range(first,total):
  row=tr.step();append(log,row)
  triggered=None
  if diagnostics and ((tr.vision and ((step+1)%cfg.get('metrics_interval',100)==0 or step+1==total)) or step+1 in cfg.get('probe_steps',[])):
   diagnostics.record(step+1)
   if adaptive:
    from switch_rule import decision
    triggered,checks=decision(list(csv.DictReader((OUT/'visual_metrics.csv').open())),skip_initial=cfg.get('force_visual',False))
    save(OUT/'rule_checks.json',checks)
    if triggered:
     triggered=dict(triggered,need_visual=triggered['reason']!='low_initial_distance',restore_parent_vision=triggered['reason']=='low_initial_distance')
     assert triggered['step']==step+1
     persist(tr,step+1);atomic_state(OUT/f'state_{step+1}.pt',snapshot(tr));atomic_state(OUT/'resume.pt',snapshot(tr))
     save(OUT/'switch_decision.json',dict(triggered,causal=True,rule_source='switch_rule.py'))
     total=step+1
     break
  if (step+1)%10==0:save(OUT/'status.json',{'state':'training','gpu':GPU,'step':step+1,'total_steps':total,'elapsed_s':time.monotonic()-start_all})
  if step+1 in cfg['checkpoints'] or (args.smoke and step+1==total):
   persist(tr,step+1)
   if step+1==cfg['stop_after'] or args.smoke:atomic_state(OUT/f'state_{step+1}.pt',snapshot(tr))
  if (step+1)%250==0 or step+1==total:atomic_state(OUT/'resume.pt',snapshot(tr))
 assert fingerprint((n,p) for n,p in tr.model.named_parameters() if not p.requires_grad)==frozen
 allrows=[json.loads(x) for x in log.read_text().splitlines()];expected_start=cfg.get('switch_step',0)+1
 assert [x['step'] for x in allrows]==list(range(expected_start,total+1)),(expected_start,total,len(allrows))
 history=[json.loads(x) for x in (OUT/'population_history.jsonl').read_text().splitlines()] if (OUT/'population_history.jsonl').exists() else []
 pop={'seconds':sum(x['seconds'] for x in history if x['start_step']<total)}
 timing={'updates':len(allrows),'sum_update_seconds':sum(x['seconds'] for x in allrows),'median_step_s':float(np.median([x['seconds'] for x in allrows])),'peak_cuda_bytes':max(x['peak_cuda_bytes'] for x in allrows),'cache_regeneration_seconds':pop['seconds'],'session_wall_s':time.monotonic()-start_all,'checkpointing':bool(tr.model.llm.gradient_checkpointing),'parallel_learning':True}
 if cfg.get('source_state'):
  source=Path(cfg['source_state']).parent;prefixrows=[json.loads(x) for x in (source/'steps.jsonl').read_text().splitlines() if json.loads(x)['step']<=cfg['switch_step']];timing['stage1_update_seconds']=sum(x['seconds'] for x in prefixrows);timing['total_training_plus_cache_seconds']=timing['stage1_update_seconds']+timing['sum_update_seconds']+pop['seconds']
 save(OUT/'timing.json',timing);save(OUT/f'timing_{total}.json',dict(timing,step=total,cumulative_seconds=timing.get('total_training_plus_cache_seconds',timing['sum_update_seconds'])))
 if adaptive and not (OUT/'switch_decision.json').exists() and total==cfg['stop_after']:save(OUT/'switch_decision.json',{'triggered':False,'step':total,'reason':'no_trigger_before_training_budget'})
 if not tr.vision and total>=cfg['stop_after']:
  assert CACHE.resolve().parent in [Path('/dev/shm/hengyi_pi05_r128_20261116/prefix_store'), Path('/var/tmp/hengyi_pi05_r128_20261116/prefix_store')]
  tr.copy.synchronize();shutil.rmtree(CACHE);save(OUT/'cache_cleanup.json',{'state':'complete','generated_cache_only':str(CACHE),'identity_retained':str(OUT/'cache_identity.json')})
 save(OUT/'status.json',{'state':'complete' if total>=cfg['stop_after'] or (adaptive and (OUT/'switch_decision.json').exists()) else 'segment_complete','steps':total,'gpu':GPU,'smoke':args.smoke,'frozen_unchanged':True})
if __name__=='__main__':
 import traceback
 try:main()
 except Exception:save(OUT/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
