import argparse,collections,gc,shutil,traceback
from common import *
from train import make_sequence
ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['full','lora32'],required=True);ap.add_argument('--variant',choices=['a','b'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--name',required=True);ap.add_argument('--stage',type=int,default=1);ap.add_argument('--cache',action='store_true');ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--profile-only',action='store_true');args=ap.parse_args();OUT=ROOT/'training'/args.name
def main():
 OUT.mkdir(parents=True,exist_ok=False);save(OUT/'status.json',{'state':'loading','pid':os.getpid(),'gpu':GPU});gate=json.loads((ROOT/'task_gate.json').read_text());assert gate['state']=='passed';torch.manual_seed(args.seed);np.random.seed(args.seed)
 model=load_model();pairs=setup_family(model,args.family);exposure_transform(model,gate['brightness']);samples=Samples(model);sequence=make_sequence(args.seed);save(OUT/'sequence.json',sequence);pin_threads()
 cfg={'family':args.family,'variant':args.variant,'seed':args.seed,'stage':args.stage,'cache':args.cache,'brightness':gate['brightness'],'steps':args.steps,'microbatch':4,'accumulation':2,'prefix_batch':1,'prefix_mode':'eval no_grad','ema_decay':.999,'lr':1e-5,'rank':32 if args.family=='lora32' else None,'alpha':32 if args.family=='lora32' else None,'saved_weights':'EMA','profile_only':args.profile_only,'trainable_numel':sum(p.numel() for _,p in pairs),'trainable':[{'name':n,'numel':p.numel(),'dtype':str(p.dtype)} for n,p in pairs]};save(OUT/'config.json',cfg)
 frozen=fingerprint((n,p) for n,p in model.named_parameters() if not p.requires_grad);masters=[torch.nn.Parameter(p.detach().float().clone()) for _,p in pairs];ema=[p.detach().float().clone() for _,p in pairs];optim=torch.optim.AdamW(masters,lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,foreach=False);cache=collections.OrderedDict();hits=misses=0
 def prefix(obs,key):
  nonlocal hits,misses
  if args.cache:
   if key not in cache:
    with torch.no_grad():_,mask,kv=model.build_prefix_cache(obs)
    cache[key]=tree_map(lambda t:t.detach().cpu().pin_memory(),(mask,kv));misses+=1
    if len(cache)>512:cache.popitem(last=False)
   else:hits+=1;cache.move_to_end(key)
   return tree_map(lambda t:t.cuda(non_blocking=True),cache[key])
  with torch.no_grad():_,mask,kv=model.build_prefix_cache(obs)
  assert all(not t.requires_grad for t in tree_leaves(kv));return mask,kv
 def batch_loss(step,micro):
  parts=[];prefixes=[]
  for j in range(4):
   ep,start=sequence[step][micro*4+j];part=prep(model,samples,ep,start,args.seed+step*8+micro*4+j,args.variant=='a');parts.append(part);prefixes.append(prefix(part[0],(ep,start)))
  obs=cat_obs([p[0] for p in parts]);x,u,t=[torch.cat([p[k] for p in parts],0) for k in [1,2,3]];mask,kv=tree_map(lambda *xs:torch.cat(xs,0),*prefixes)
  return (model.velocity_from_suffix(model.run_suffix(obs,x,t,kv,mask))-u).square().mean()
 # Verify cache and recomputation on the identical B1 path before any optimizer update.
 if args.stage==3:
  use_cache=args.cache;args.cache=False;first=batch_loss(0,0);first.backward();ref=[p.grad.detach().float().cpu() for _,p in pairs];v=float(first.detach());model.zero_grad(set_to_none=True)
  args.cache=True;second=batch_loss(0,0);second.backward();delta=den=0.
  for (_,p),g in zip(pairs,ref):delta+=float((p.grad.detach().float().cpu()-g).square().sum());den+=float(g.square().sum())
  check={'reference_loss':v,'cached_loss':float(second.detach()),'gradient_relative_l2':math.sqrt(delta/max(den,1e-30))};save(OUT/'cache_parity.json',check);assert check['gradient_relative_l2']<1e-6,check
  model.zero_grad(set_to_none=True);args.cache=use_cache;cache.clear();del ref,first,second;gc.collect();torch.cuda.empty_cache()
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();cost_rows=[]
 start_time=time.monotonic()
 for step in range(args.steps):
  t0=time.monotonic();losses=[]
  for micro in range(2):
   loss=batch_loss(step,micro);assert torch.isfinite(loss);(loss/2).backward();losses.append(float(loss.detach()));del loss
  for (_,p),m in zip(pairs,masters):assert p.grad is not None;m.grad=p.grad.detach().float()
  norm=torch.nn.utils.clip_grad_norm_(masters,1.,error_if_nonfinite=True);optim.param_groups[0]['lr']=1e-5*min(1.,(step+1)/50);optim.step()
  with torch.no_grad():
   for (_,p),m,e in zip(pairs,masters,ema):p.copy_(m);e.mul_(.999).add_(m,alpha=.001)
  optim.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True);torch.cuda.synchronize();row={'step':step+1,'loss':float(np.mean(losses)),'gradient_norm':float(norm),'seconds':time.monotonic()-t0,'elapsed_s':time.monotonic()-start_time,'cache_hits':hits,'cache_misses':misses,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()}
  cost_rows.append(row)
  with (OUT/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  save(OUT/'status.json',{'state':'training','gpu':GPU,**row,'eta_s':(args.steps-step-1)*row['elapsed_s']/(step+1)})
  if not args.profile_only and ((args.family!='lora32' and args.stage==1 and (step+1)%250==0) or step+1==args.steps):
   if shutil.disk_usage(ROOT).free<20*2**30:raise RuntimeError('disk free below20GiB; checkpoint not written')
   payload={'parameters':{n:e.detach().to(dtype=p.dtype).cpu() for (n,p),e in zip(pairs,ema)},'family':args.family,'seed':args.seed,'step':step+1,'ema_decay':.999,'brightness':gate['brightness'],'parent':json.load(open('/home/hengyi/vla_followup_20261020/audit/parent_download.json')),'weights_kind':'EMA'}
   path=OUT/f'ema_{step+1}.pt';tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path);save(OUT/f'checkpoint_{step+1}.json',{'state':'ready','path':str(path),'step':step+1});del payload
 measured=cost_rows[10:] if len(cost_rows)>10 else cost_rows
 save(OUT/'cost_summary.json',{'family':args.family,'variant':args.variant,'profile_only':args.profile_only,'trainable_numel':sum(p.numel() for _,p in pairs),'steps':args.steps,'warmup_steps':10 if len(cost_rows)>10 else 0,'measured_steps':len(measured),'step_seconds_mean':float(np.mean([x['seconds'] for x in measured])),'step_seconds_median':float(np.median([x['seconds'] for x in measured])),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu':GPU,'measurement':'host wall per complete update incl preparation/FWD/BWD/OPT/EMA, excludes checkpoint I/O; allocator peak since pre-training reset includes model, masters, EMA, moments and activations'})
 assert fingerprint((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen
 save(OUT/'status.json',{'state':'complete','steps':args.steps,'frozen_unchanged':True,'gpu':GPU,'elapsed_s':time.monotonic()-start_time,'cache_hits':hits,'cache_misses':misses})
if __name__=='__main__':
 try:main()
 except Exception:save(OUT/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
