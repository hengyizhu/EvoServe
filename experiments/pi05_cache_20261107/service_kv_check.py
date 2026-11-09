import sys,traceback
sys.path.insert(0,'/home/hengyi/pi05_cache_20261107')
from cache_train import *
from cache_cost import ExportServing
def main():
 tr=CacheTrain();s=ExportServing(1);s.verify_graph();rows=[]
 for i in [0,1,2,3]:
  obs,x,u,t=tr.prepare(i)
  with torch.cuda.stream(s.stream),torch.inference_mode():
   s.copy_input(obs,torch.zeros_like(x));s.graph.replay();s.events[-1].synchronize()
   exported=tree_map(lambda q:q.cpu(),(s.exposed[1],s.exposed[2]))
  mask,kv=tr.prefix(obs);diffs=[]
  for a,b in zip(tree_leaves(exported),tree_leaves((mask,kv))):
   if a.dtype==torch.bool:assert torch.equal(a,b.cpu())
   else:diffs.append(float((a.float()-b.detach().float().cpu()).abs().max()))
  rows.append({'input':i,'graph_vs_eager_service_prefix_max_abs':max(diffs),'mask_equal':True})
 save(Path('/home/hengyi/pi05_cache_20261107/prototype/service_graph_kv_check.json'),rows)
if __name__=='__main__':
 try:main()
 except Exception:save(Path('/home/hengyi/pi05_cache_20261107/prototype/service_graph_kv_failure.json'),{'traceback':traceback.format_exc()});raise
