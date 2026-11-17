"""Measured arrivals select the merge count; deadline feedback only changes duty."""
from adaptive_feedback import Feedback as Base
import json,time
from pathlib import Path

def safe_count(load,table):
 points=sorted(table,key=lambda x:x['new_load'])
 if load<=points[0]['new_load']:return points[0]['safe_experts']
 for left,right in zip(points,points[1:]):
  if abs(load-right['new_load'])<1e-12:return right['safe_experts']
  if left['new_load']<load<right['new_load']:return min(left['safe_experts'],right['safe_experts'])
 return points[-1]['safe_experts']

class Feedback(Base):
 def __init__(self,root,cfg):
  super().__init__(root);self.mode=cfg['controller'];self.experts=cfg['initial_experts'];self.table=cfg['safe_table'];self.service_ms=cfg['reference_service_ms']
 def update(self,out):
  path=Path(out)/'feedback_windows.jsonl'
  if not path.exists():return
  with path.open() as f:
   f.seek(self.offset)
   while True:
    pos=f.tell();line=f.readline()
    if not line.endswith('\n'):f.seek(pos);break
    data=json.loads(line);old_m,old_d=self.experts,self.duty;rate=data['arrivals_observed']/data['window_seconds'];load=rate*self.service_ms/1000
    if self.mode=='feedforward':self.experts=safe_count(load,self.table)
    self.duty=max(0.,min(1.,self.duty-5.*(data['timeout_rate']-.01)));self.last_window=data['window_end_ns']
    row=dict(data,observed_ns=time.monotonic_ns(),controller=self.mode,measured_arrival_rate=rate,measured_load=load,experts_before=old_m,experts_after=self.experts,duty_before=old_d,duty_after=self.duty,reason='arrival_table_and_duty_feedback' if self.mode=='feedforward' else 'fixed_count_and_duty_feedback',target_timeout_rate=.01,gain=5.,reference_service_ms=self.service_ms)
    with (Path(out)/'controller.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
   self.offset=f.tell()
  self.pause_s=None if self.duty==0 else (self.step_ema or .27)*(1-self.duty)/self.duty
