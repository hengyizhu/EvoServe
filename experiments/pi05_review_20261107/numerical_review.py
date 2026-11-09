import sys,os,time,json,gc,traceback
from pathlib import Path
sys.path.insert(0,'/home/hengyi/pi05_cache_20261107')
from cache_train import CacheTrain,compare,cat_obs,torch,np,model_module,tree_map,tree_leaves,save,Samples
R=Path('/home/hengyi/pi05_review_20261107');OLD=Path('/home/hengyi/pi05_cache_20261107')
def main():
 save(R/'numerical_status.json',{'state':'loading'});tr=CacheTrain();tr.build_host(8)
 # Same B1 production path on both sides: isolates caching from changing prefix batch shape.
 original=tr.prefix
 def single_prefix(obs):
  import dataclasses
  chunks=[]
  for i in range(obs.state.shape[0]):
   item=type(obs)(**{f.name:tree_map(lambda x:x[i:i+1] if torch.is_tensor(x) else x,getattr(obs,f.name)) for f in dataclasses.fields(obs)})
   chunks.append(original(item))
  return tree_map(lambda *xs:torch.cat(xs,dim=0),*chunks)
 tr.prefix=single_prefix;a=tr.gradient(4);b=tr.gradient(4,True);checks={'matched_B1_prefix_recompute_vs_cached':compare(a,b)};del a,b;tr.prefix=original
 parts=[tr.prepare(i)[0] for i in range(4)];batch=cat_obs(parts)
 def kv_compare():
  one=[original(o) for o in parts];ref=tree_map(lambda *xs:torch.cat(xs,dim=0),*one);new=original(batch);layers=[]
  assert torch.equal(ref[0],new[0])
  for i,(aa,bb) in enumerate(zip(ref[1],new[1])):
   d=r=0.;mx=0.
   for x,y in zip(aa,bb):
    x=x.float();y=y.float();d+=float((x-y).square().sum());r+=float(x.square().sum());mx=max(mx,float((x-y).abs().max()))
   layers.append({'layer':i,'relative_l2':(d/max(r,1e-30))**.5,'max_abs':mx})
  return layers
 checks['BF16_prefix_B1_vs_B4_per_layer']=kv_compare();save(R/'numerical_checks.json',checks)
 # Rebuild observations directly from held-out target episodes and existing history pool.
 samples=Samples(tr.model);split=samples.split;original_transform=tr.model._input_transform_fn
 def transform(raw):
  if raw['prompt']==split['target']:
   raw=dict(raw)
   for k in ['observation/image','observation/wrist_image']:raw[k]=np.ascontiguousarray(raw[k][:,::-1,:])
  return original_transform(raw)
 tr.model._input_transform_fn=transform
 selected=[]
 for ep in split['target_validation']:
  for start in np.linspace(0,ep['length']-10,4,dtype=int):selected.append(('target_heldout',ep['episode_index'],int(start)))
 for item in split['replay_windows'][::max(1,len(split['replay_windows'])//20)][:20]:selected.append(('history_replay',item['episode'],item['start']))
 prepared=[]
 for i,(task,ep,start) in enumerate(selected):
  ob,ac=samples.get(ep,start);ob=model_module.preprocess_observation(ob,train=False);ob=model_module._observation_to_dtype(ob,tr.model.embed_dtype);ac=ac.float()
  noise=torch.randn(ac.shape,generator=torch.Generator().manual_seed(740000+i)).cuda();t=torch.tensor(np.random.default_rng(750000+i).beta(1.5,1.,size=1)*.999+.001,device='cuda',dtype=torch.float32)
  prepared.append((ob,ac,noise,t));selected[i]={'task':task,'episode':ep,'start':start,'noise_seed':740000+i,'time_seed':750000+i}
 save(R/'validation_samples.json',selected);del samples;gc.collect()
 names=['parent','a_seed20261107','b_seed20261107','c_seed20261107','a_seed20261108','c_seed20261108'];rows=[]
 # tr has computed gradients only; no optimizer updates, so parent remains intact here.
 for name in names:
  if name!='parent':
   ck=torch.load(OLD/'learning'/name/'final.pt',map_location='cpu',weights_only=True,mmap=True);assert set(ck['parameters'])=={n for n,p in tr.pairs};tr.model.load_state_dict(ck['parameters'],strict=False);del ck
  save(R/'numerical_status.json',{'state':'validation','checkpoint':name})
  for mode in ['eval','train']:
   tr.model.train(mode=='train')
   for j in range(0,len(prepared),4):
    ob=cat_obs([x[0] for x in prepared[j:j+4]]);ac,noise,t=[torch.cat([x[k] for x in prepared[j:j+4]],dim=0) for k in [1,2,3]]
    with torch.no_grad():
     x=t[:,None,None]*noise+(1-t[:,None,None])*ac;u=noise-ac;mask,kv=original(ob);v=tr.model.velocity_from_suffix(tr.model.run_suffix(ob,x,t,kv,mask));loss=(v-u).square();native=tr.model.compute_loss(ob,ac,train=False,noise=noise,time=t)
    for k in range(len(prepared[j:j+4])):
     row={**selected[j+k],'checkpoint':name,'prefix_mode':mode,'split_loss32':float(loss[k].mean()),'split_loss7':float(loss[k,:,:7].mean()),'native_loss32':float(native[k].mean()),'velocity_rms':float(v[k].square().mean().sqrt())};rows.append(row)
     with (R/'validation_raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  summary=[]
  for mode in ['eval','train']:
   for task in ['target_heldout','history_replay']:
    rr=[x for x in rows if x['checkpoint']==name and x['prefix_mode']==mode and x['task']==task];summary.append({'checkpoint':name,'prefix_mode':mode,'task':task,'n':len(rr),**{k:float(np.mean([x[k] for x in rr])) for k in ['split_loss32','split_loss7','native_loss32','velocity_rms']}})
  save(R/f'validation_{name}.json',summary)
 save(R/'numerical_status.json',{'state':'complete'})
if __name__=='__main__':
 try:main()
 except Exception:save(R/'numerical_status.json',{'state':'failed','traceback':traceback.format_exc()});raise
