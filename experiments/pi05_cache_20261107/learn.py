import os,sys,time,json,dataclasses,traceback,argparse,random
from pathlib import Path
sys.path.insert(0,'/home/hengyi/pi05_optimized_20261107')
from training_mb4 import *
from train import make_sequence
BASE=Path('/home/hengyi/pi05_cache_20261107')
ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['a','b','c'],required=True);ap.add_argument('--seed',type=int,default=20261107);ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--mirror',action='store_true');ap.add_argument('--brightness',type=float,default=1.0);args=ap.parse_args();OUT=BASE/'learning'/f'{args.variant}_seed{args.seed}';OUT.mkdir(parents=True,exist_ok=False)
def main():
 save(OUT/'status.json',{'state':'loading','gpu':GPU,'pid':os.getpid()});torch.manual_seed(args.seed);np.random.seed(args.seed);model=load_model();pairs=configure(model,'L2');samples=Samples(model);original_transform=model._input_transform_fn
 def transformed(raw):
  if raw['prompt']==samples.split['target'] and args.brightness!=1.:
   raw=dict(raw)
   for key in ['observation/image','observation/wrist_image']:raw[key]=(raw[key].astype(np.float32)*args.brightness).clip(0,255).astype(np.uint8)
  if raw['prompt']==samples.split['target'] and args.mirror:
   raw=dict(raw)
   for key in ['observation/image','observation/wrist_image']:raw[key]=np.ascontiguousarray(raw[key][:,::-1,:])
  return original_transform(raw)
 model._input_transform_fn=transformed;seq=make_sequence(args.seed)
 if args.variant=='c':seq=[[(ep,(start//5)*5) for ep,start in batch] for batch in seq]
 save(OUT/'sequence.json',seq);save(OUT/'config.json',{'variant':args.variant,'target_brightness':args.brightness,'target_mirror':args.mirror,'seed':args.seed,'steps':args.steps,'microbatch':4,'accumulation':2,'effective_batch':8,'augmentation':args.variant=='a','frame_rule':'floor to episode-relative multiple of 5' if args.variant=='c' else 'arbitrary','mixture':'4 adaptation + 4 history per update, same episodes/order across variants','updated_parameters':[{'name':n,'dtype':str(p.dtype),'numel':p.numel()} for n,p in pairs],'prefix':'frozen, no_grad, same split KV path as preceding measurement','model_mode':'train','optimizer':'AdamW FP32 masters, lr2.5e-5, warmup50, betas0.9/0.95, eps1e-8, wd1e-10, clip1','eval_denoise_steps':5,'note':'Evaluation retains historical learning protocol (5 denoise steps); serving cost uses 10.'})
 frozen=fingerprint((n,p) for n,p in model.named_parameters() if not p.requires_grad);masters=[torch.nn.Parameter(p.detach().float().clone()) for _,p in pairs];optim=torch.optim.AdamW(masters,lr=2.5e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,foreach=False);started=time.monotonic()
 for step,batch in enumerate(seq[:args.steps]):
  tic=time.monotonic();losses=[]
  for micro in range(2):
   parts=[]
   for j in range(4):
    index=step*8+micro*4+j;obs,ac=samples.get(*batch[micro*4+j]);ac=ac.float();seed=args.seed+index
    obs=model_module.preprocess_observation(obs,train=args.variant=='a',rng=torch.Generator().manual_seed(seed));obs=model_module._observation_to_dtype(obs,model.embed_dtype)
    noise=torch.randn(ac.shape,generator=torch.Generator().manual_seed(seed+100000)).cuda();t=torch.tensor(np.random.default_rng(seed+200000).beta(1.5,1.,size=1)*.999+.001,device='cuda',dtype=torch.float32);parts.append((obs,t[:,None,None]*noise+(1-t[:,None,None])*ac,noise-ac,t))
   obs=cat_obs([p[0] for p in parts]);x,u,t=[torch.cat([p[k] for p in parts],dim=0) for k in [1,2,3]]
   with torch.no_grad():_,mask,kv=model.build_prefix_cache(obs)
   assert all(not z.requires_grad for z in tree_leaves(kv))
   loss=(model.velocity_from_suffix(model.run_suffix(obs,x,t,kv,mask))-u).square().mean();assert torch.isfinite(loss);(loss/2).backward();losses.append(float(loss.detach()));del loss,obs,x,u,t,kv,mask,parts
  for (_,p),m in zip(pairs,masters):m.grad=p.grad.detach().float()
  norm=torch.nn.utils.clip_grad_norm_(masters,1.,error_if_nonfinite=True);optim.param_groups[0]['lr']=2.5e-5*min(1.,(step+1)/50);optim.step()
  with torch.no_grad():
   for (_,p),m in zip(pairs,masters):p.copy_(m)
  optim.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True);torch.cuda.synchronize();row={'step':step+1,'loss':float(np.mean(losses)),'gradient_norm':float(norm),'seconds':time.monotonic()-tic,'elapsed_s':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
  with (OUT/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  save(OUT/'status.json',{'state':'training',**row,'eta_s':(args.steps-step-1)*(time.monotonic()-started)/(step+1)})
 assert fingerprint((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen
 source=json.load(open('/home/hengyi/vla_followup_20261020/audit/parent_download.json'));torch.save({'parameters':{n:p.detach().cpu() for n,p in pairs},'parent':source,'tier':'L2','step':args.steps,'seed':args.seed},OUT/'final.pt');save(OUT/'status.json',{'state':'trained','steps':args.steps,'frozen_unchanged':True,'elapsed_s':time.monotonic()-started})
if __name__=='__main__':
 try:main()
 except Exception:save(OUT/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
