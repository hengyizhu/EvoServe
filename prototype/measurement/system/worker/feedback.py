"""Deadline-matured miss-rate feedback, no minimum training duty."""
import json,time
from pathlib import Path
from cache_engine import append
class Feedback:
 def __init__(self,root):
  self.root=Path(root);self.last_window=0;self.step_ema=None;self.duty=1.;self.pause_s=0.;self.last_finish_ns=0
 def update(self,out):
  p=self.root/'feedback.json'
  if not p.exists():return
  data=json.loads(p.read_text())
  if data['window_end_ns']<=self.last_window:return
  self.last_window=data['window_end_ns'];old=self.duty;miss=data['timeout_rate'];target=.01
  self.duty=max(0.,min(1.,self.duty-5.*(miss-target)))
  self.pause_s=None if self.duty==0 else (self.step_ema or .27)*(1-self.duty)/self.duty
  append(out/'controller.jsonl',dict(window_end_ns=self.last_window,observed_ns=time.monotonic_ns(),target_timeout_rate=target,measured_timeout_rate=miss,measured_p99_ms=data['p99_ms'],requests=data['requests'],timeouts=data['timeouts'],duty_before=old,duty_after=self.duty,estimated_duty=self.duty,pause_after_s=self.pause_s,step_ema_s=self.step_ema,fully_paused=self.duty==0,mps_limit_percent=100,deadline_ms=250,gain=5.))
 def ready(self,out):
  self.update(out)
  return self.duty>0 and (not self.last_finish_ns or time.monotonic_ns()>=self.last_finish_ns+round(self.pause_s*1e9))
 def after_step(self,row,out):
  self.step_ema=row['seconds'] if self.step_ema is None else .8*self.step_ema+.2*row['seconds'];self.last_finish_ns=row['finish_ns']
  self.update(out)
  self.pause_s=None if self.duty==0 else self.step_ema*(1-self.duty)/self.duty
