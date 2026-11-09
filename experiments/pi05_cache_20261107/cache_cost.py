import sys,time,json,traceback
from pathlib import Path
sys.path.insert(0,'/home/hengyi/pi05_optimized_20261107')
from graph_serving import *
from torch.utils._pytree import tree_map,tree_leaves
R=Path('/home/hengyi/pi05_cache_20261107/cache_cost');R.mkdir(exist_ok=True)
class ExportServing(GraphServing):
 def prefix(self,obs):
  result=super().prefix(obs);self.exposed=result;return result
def main():
 save(R/'status.json',{'state':'loading','pid':os.getpid()})
 s=ExportServing(1);save(R/'correctness.json',s.verify_graph())
 tensors=tree_leaves(s.exposed[2]);assert all(torch.is_tensor(x) for x in tensors)
 save(R/'kv_layout.json',{'bytes_per_sample':sum(x.numel()*x.element_size() for x in tensors),'tensors':[{'shape':list(x.shape),'dtype':str(x.dtype),'bytes':x.numel()*x.element_size()} for x in tensors],'mask_bytes':s.exposed[1].numel()*s.exposed[1].element_size()})
 host=[torch.empty_like(x,device='cpu',pin_memory=True) for x in tensors];copy=torch.cuda.Stream();done=torch.cuda.Event();done.record();copy_times=[]
 for export in [False,True,False,True]:
  block=len(list(R.glob('requests_*.jsonl')));rows=[];origin=time.monotonic_ns()+100_000_000
  for i in range(260):
   release=origin+i*250_000_000;time.sleep(max(0,(release-time.monotonic_ns())/1e9));start=stamp()
   with torch.cuda.stream(s.stream),torch.inference_mode():
    s.stream.wait_event(done);obs,noise=s.batch_input(i);s.copy_input(obs,noise);s.graph.replay()
    if export:
     a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
     with torch.cuda.stream(copy):
      copy.wait_event(s.events[1]);a.record()
      for h,x in zip(host,tensors):h.copy_(x,non_blocking=True)
      b.record();done.record()
    out=s.model.decode_actions(s.static_result,obs.state).cpu();s.events[-1].synchronize();end=stamp()
   if export:done.synchronize();copy_times.append(a.elapsed_time(b))
   row={'block':block,'export':export,'warmup':i<10,'request':i,'release_ns':release,'start_ns':start,'finish_ns':end,'response_ms':(end-release)/1e6,'service_ms':(end-start)/1e6,'prefix_ms':s.events[0].elapsed_time(s.events[1]),'denoise_ms':s.events[1].elapsed_time(s.events[-1]),'copy_ms':copy_times[-1] if export else None};rows.append(row)
   with (R/f'requests_{block}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if i%20==0:save(R/'status.json',{'state':'measuring','block':block,'request':i,'export':export})
  save(R/f'block_{block}.json',{'export':export,'response_ms':stats([x['response_ms'] for x in rows[10:]]),'service_ms':stats([x['service_ms'] for x in rows[10:]])})
 # Verify DMA contents after the final replay, before any graph buffer reuse.
 for h,x in zip(host,tensors):torch.testing.assert_close(h,x.cpu(),rtol=0,atol=0)
 pinned8=[torch.cat([h]*8,dim=0).pin_memory() for h in host];gpu8=[torch.empty_like(h,device='cuda') for h in pinned8];transfer=[]
 for i in range(110):
  _,row=timed('H2D_eight_prefixes',lambda:[g.copy_(h,non_blocking=True) for g,h in zip(gpu8,pinned8)])
  if i>=10:transfer.append(row)
 save(R/'h2d_raw.json',transfer)
 for h,g in zip(pinned8,gpu8):torch.testing.assert_close(h,g.cpu(),rtol=0,atol=0)
 records=[json.loads(x) for p in R.glob('requests_*.jsonl') for x in p.read_text().splitlines()];summary={}
 for mode in [False,True]:
  rr=[x for x in records if x['export']==mode and not x['warmup']];summary['export' if mode else 'baseline']={'n':len(rr),'response_ms':stats([x['response_ms'] for x in rr]),'service_ms':stats([x['service_ms'] for x in rr]),'slo_rate':sum(x['response_ms']<=250 for x in rr)/len(rr)}
 summary.update(p99_increase_ms=summary['export']['response_ms']['p99']-summary['baseline']['response_ms']['p99'],h2d_8_cuda_ms=stats([x['cuda_ms'] for x in transfer]),h2d_8_wall_ms=stats([x['wall_ms'] for x in transfer]),d2h_ms=stats(copy_times),transfer_integrity=True,policy='D2H waits on prefix completion, overlaps denoising; next replay waits on copy completion to prevent overwriting source. One pinned host slot; copy completion checked outside request response.')
 save(R/'summary.json',summary);save(R/'status.json',{'state':'complete'})
if __name__=='__main__':
 try:main()
 except Exception:save(R/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise
