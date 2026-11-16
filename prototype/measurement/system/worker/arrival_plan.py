"""Fixed nominal releases; stage transitions do not reset the trainer/controller."""
import math
def schedule(cfg):
 origin=cfg['origin_ns'];offset=cfg.get('source_offset',0);stride=cfg.get('source_stride',1);clients=cfg.get('clients',4)
 phases=cfg.get('phases') or [dict(seconds=cfg['duration_s'],rate=cfg.get('global_rate',cfg.get('rate')))]
 plan=[];elapsed=0.;index=0
 assert abs(sum(p['seconds'] for p in phases)-cfg['duration_s'])<1e-6
 for stage,p in enumerate(phases):
  for i in range(math.ceil(p['seconds']*p['rate']-1e-10)):
   release=origin+round((elapsed+i/p['rate'])*1e9)
   if index%stride==offset:plan.append(dict(request_id=index,client_id=index%clients,nominal_release_ns=release,deadline_ns=release+250_000_000,stage=stage+1,target_load=p.get('load')))
   index+=1
  elapsed+=p['seconds']
 return plan
