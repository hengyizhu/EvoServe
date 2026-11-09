import sys,traceback
sys.path.insert(0,'/home/hengyi/pi05_cache_20261107')
from cache_train import *
OUT=R/'precision_checks.json'
def main():
 tr=CacheTrain();tr.build_host(8);a=tr.gradient(1,True);b=tr.gradient(4,True);result={'same_cached_B1_prefix_suffix_batch1_vs4':compare(a,b)};del a,b;save(OUT,result)
 # Diagnostic only: no optimizer update and no checkpoint write.
 tr.host_cache.clear();tr.model.float();tr.model.embed_dtype=torch.float32;tr.model.llm.embed_dtype=torch.float32;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 a=tr.gradient(1);b=tr.gradient(4);result['all_FP32_no_TF32_mb1_vs_mb4']=compare(a,b);del a,b;save(OUT,result)
 tr.build_host(8);a=tr.gradient(4);b=tr.gradient(4,True);result['all_FP32_service_B1_cache_vs_recompute_B4']=compare(a,b);save(OUT,result)
if __name__=='__main__':
 try:main()
 except Exception:save(R/'precision_failure.json',{'traceback':traceback.format_exc()});raise
