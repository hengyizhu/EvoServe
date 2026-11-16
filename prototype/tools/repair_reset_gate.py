"""One explicit queued insertion at the next idle boundary; never interrupt a run."""
import argparse,sys,time,fcntl,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evoserve.campaign import Campaign
from evoserve.io import read,save
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
with (root/'reset_gate_repair.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 job=read(root/'pending_reset_gate.json',{})
 if job.get('state')=='pending':
  save(root/'pending_reset_gate.json',dict(job,state='running',started=time.time()))
  try:
   campaign=Campaign(root,read(root/'prototype_campaign_reset_batch.json'))
   cfg=campaign.setting('ours',seconds=600,changes=[dict(expert=3,at_s=0,perturb=dict(kind='exposure',factor=.05))])
   cfg['observation_monitoring']='Actual emission/physics lateness separate from stale camera image; both reported. No deadline or freshness values changed.'
   result=campaign.trial('reset_M1_10min_freshness_split',cfg,lambda x:campaign.health(x,learning=True,slo=True))
   old=read(campaign.states/'reset_M1_10min.json')
   if old:save(root/'status_history/reset_M1_10min_before_monitor_fix.json',old)
   fields={k:v for k,v in result.items() if k not in ['item','state','power_limit_w','updated']};fields['replacement_for']='old short-test gate conflated stale camera observations with actual fixed-period emission lateness; original measurements preserved'
   campaign.record('reset_M1_10min',result['state'],**fields)
   save(root/'pending_reset_gate.json',dict(job,state='complete',passed=result['state']=='complete',raw=result.get('raw'),finished=time.time()))
  except Exception:
   save(root/'pending_reset_gate.json',dict(job,state='failed',error=traceback.format_exc(),finished=time.time()))
