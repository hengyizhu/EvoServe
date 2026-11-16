"""Reset/batching repair gates; original M2/M3 are suspended until these pass."""
import sys,time,traceback
from .campaign import Campaign
from .io import read,save
class RevisionCampaign(Campaign):
 def run(self):
  assert read(self.root/'measurements_complete.json')
  assert read(self.root/'audit/reset_excluded_cpu/summary.json',{}).get('state')=='passed'
  self.record('reset_cpu_probe','passed',raw=str(self.root/'audit/reset_excluded_cpu'))
  lowlight=[dict(expert=3,at_s=0,perturb=dict(kind='exposure',factor=.05))]
  for label,method in [('reset_inference','no_learning'),('reset_M1','ours')]:
   smoke=self.trial(label+'_smoke',self.setting(method,seconds=60,changes=lowlight),lambda x:self.health(x,learning=method=='ours',slo=True))
   if smoke['state']=='complete':
    self.trial(label+'_10min',self.setting(method,seconds=600,changes=lowlight),lambda x:self.health(x,learning=method=='ours',slo=True))
   else:self.record(label+'_10min','blocked',reason='short smoke failed twice')
  # CPU-only implementation may proceed while reset measurements run.
  while not read(self.root/'batch_revision_ready.json',{}).get('cpu_checks_passed'):
   self.status('waiting_batch_cpu_checks',M2_M3='suspended');time.sleep(10)
  out=self.out/'batch_stage_process'
  self.execute([sys.executable,str(self.repo/'prototype/tools/batch_stage.py'),'--root',str(self.root),'--config',str(self.root/'prototype_campaign_reset_batch.json')],out,timeout=7*24*3600)
