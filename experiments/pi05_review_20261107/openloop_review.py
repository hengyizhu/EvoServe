import os,sys,json,traceback
from pathlib import Path
sys.path.insert(0,'/home/hengyi/vla_overnight_20261104')
from core import *
R=Path('/home/hengyi/pi05_review_20261107');OLD=Path('/home/hengyi/pi05_cache_20261107')
def main():
 model=load_model().eval();samples=Samples(model);transform=model._input_transform_fn
 def mirrored(raw):
  if raw['prompt']==samples.split['target']:
   raw=dict(raw)
   for k in ['observation/image','observation/wrist_image']:raw[k]=np.ascontiguousarray(raw[k][:,::-1,:])
  return transform(raw)
 model._input_transform_fn=mirrored;manifest=json.load(open(R/'validation_samples.json'));selected=[x for task in ['target_heldout','history_replay'] for x in [m for m in manifest if m['task']==task][:10]];inputs=[samples.get(x['episode'],x['start']) for x in selected];rows=[]
 for name in ['parent','a_seed20261107','a_seed20261108','c_seed20261108']:
  if name!='parent':
   ck=torch.load(OLD/'learning'/name/'final.pt',map_location='cpu',weights_only=True,mmap=True);model.load_state_dict(ck['parameters'],strict=False);del ck
  save(R/'openloop_status.json',{'state':'running','checkpoint':name})
  for i,(ob,ac) in enumerate(inputs):
   noise=torch.randn(ac.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(800000+i))
   for steps in [5,10,20]:
    with torch.inference_mode():action=model.sample_actions(ob,num_steps=steps,noise=noise)
    row={**selected[i],'checkpoint':name,'steps':steps,'noise_seed':800000+i,'normalized_action_mse7':float((action[:,:,:7]-ac[:,:,:7].float()).square().mean()),'normalized_action_maxabs7':float(action[:,:,:7].abs().max()),'finite':bool(torch.isfinite(action).all())};rows.append(row)
    with (R/'openloop_raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 summary=[]
 for name in ['parent','a_seed20261107','a_seed20261108','c_seed20261108']:
  for task in ['target_heldout','history_replay']:
   for steps in [5,10,20]:
    rr=[x for x in rows if x['checkpoint']==name and x['task']==task and x['steps']==steps];summary.append({'checkpoint':name,'task':task,'steps':steps,'n':len(rr),'mean_mse7':float(np.mean([x['normalized_action_mse7'] for x in rr]))})
 save(R/'openloop_summary.json',summary);save(R/'openloop_status.json',{'state':'complete'})
if __name__=='__main__':
 try:main()
 except Exception:save(R/'openloop_status.json',{'state':'failed','traceback':traceback.format_exc()});raise
