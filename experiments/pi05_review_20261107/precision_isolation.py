import sys,traceback,gc
from pathlib import Path
sys.path.insert(0,'/home/hengyi/pi05_cache_20261107')
from cache_train import *
R=Path('/home/hengyi/pi05_review_20261107')
def diff(x,y):
 x=x.float();y=y.float();return {'relative_l2':float((x-y).norm()/x.norm().clamp_min(1e-30)),'max_abs':float((x-y).abs().max())}
def main():
 tr=CacheTrain();results={};flags={'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'bf16_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'float32_precision':torch.get_float32_matmul_precision()};save(R/'precision_original_flags.json',flags)
 for name in ['original','TF32_off','TF32_and_BF16_reduction_off']:
  if name!='original':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  if name=='TF32_and_BF16_reduction_off':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
  a=tr.gradient(1);b=tr.gradient(4);result={'gradient':compare(a,b)};del a,b
  obs=[tr.prepare(i)[0] for i in range(4)];bat=cat_obs(obs)
  with torch.no_grad():
   result['vision']={}
   for key in bat.images:
    one=torch.cat([tr.model.img(o.images[key])[0] for o in obs]);many=tr.model.img(bat.images[key])[0];result['vision'][key]=diff(one,many)
  results[name]=result;save(R/'precision_isolation.json',results);gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 try:main()
 except Exception:save(R/'precision_isolation_failure.json',{'traceback':traceback.format_exc()});raise
