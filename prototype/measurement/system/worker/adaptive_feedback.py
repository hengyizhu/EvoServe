"""Five-second deadline feedback; count decisions are applied at step boundaries."""
import json, os, time
from pathlib import Path

class Rule:
 def __init__(self, mode='adaptive', count=1):
  self.mode=mode;self.experts=count;self.duty=1.;self.zero_streak=0
 def observe(self, miss):
  old_m,old_d=self.experts,self.duty;reason='duty_feedback'
  if self.mode=='adaptive' and miss>.01 and self.experts>1:
   self.experts=max(1,self.experts//2);reason='halve_experts'
  else:self.duty=max(0.,min(1.,self.duty-5.*(miss-.01)))
  self.zero_streak=self.zero_streak+1 if miss==0 and old_d==1. else 0
  if self.mode=='adaptive' and self.zero_streak>=3:
   self.experts=min(8,self.experts*2);self.zero_streak=0;reason='double_experts' if self.experts!=old_m else 'at_maximum'
  return dict(experts_before=old_m,experts_after=self.experts,duty_before=old_d,duty_after=self.duty,zero_timeout_streak=self.zero_streak,reason=reason)

class Feedback(Rule):
 def __init__(self,root):
  super().__init__(os.environ.get('CONTROL_MODE','adaptive'),int(os.environ.get('INITIAL_EXPERTS','1')))
  self.root=Path(root);self.step_ema=None;self.last_finish_ns=0;self.offset=0;self.last_window=0;self.pause_s=0.
 def update(self,out):
  path=Path(out)/'feedback_windows.jsonl'
  if not path.exists():return
  # Consume every published window, including windows that expired during a long step.
  with path.open() as f:
   f.seek(self.offset)
   while True:
    pos=f.tell();line=f.readline()
    if not line.endswith('\n'):f.seek(pos);break
    data=json.loads(line);decision=self.observe(data['timeout_rate']);self.last_window=data['window_end_ns']
    row=dict(data,**decision,observed_ns=time.monotonic_ns(),target_timeout_rate=.01,gain=5.,mps_limit_percent=100)
    with (Path(out)/'controller.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
   self.offset=f.tell()
  self.pause_s=None if self.duty==0 else (self.step_ema or .27)*(1-self.duty)/self.duty
 def ready(self,out):
  self.update(out)
  return self.duty>0 and (not self.last_finish_ns or time.monotonic_ns()>=self.last_finish_ns+round(self.pause_s*1e9))
 def after_step(self,row,out):
  self.step_ema=row['seconds'] if self.step_ema is None else .8*self.step_ema+.2*row['seconds'];self.last_finish_ns=row['finish_ns'];self.update(out)
  self.pause_s=None if self.duty==0 else self.step_ema*(1-self.duty)/self.duty
