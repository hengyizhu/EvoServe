"""Independent deadline-matured windows include every nominal arrival, even unfinished."""
import threading,time,json,math
from pathlib import Path
class Publisher:
 def __init__(self,root,out,plan,origin,end):
  self.root=Path(root);self.out=Path(out);self.plan=plan;self.origin=origin;self.end=end;self.rows={};self.lock=threading.Lock();self.error=None
  self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
 def add(self,row):
  with self.lock:self.rows[row['request_id']]=dict(row)
 def run(self):
  try:
   for i in range(math.ceil((self.end-self.origin)/5e9)):
    lo=self.origin+int(i*5e9);hi=min(self.end,lo+5_000_000_000);due=hi+250_000_000
    time.sleep(max(0,(due-time.monotonic_ns())/1e9))
    requests=[r for r in self.plan if lo<=r['nominal_release_ns']<hi]
    with self.lock:rows={r['request_id']:self.rows.get(r['request_id']) for r in requests}
    if not requests:continue
    late=sum(rows[r['request_id']] is None or rows[r['request_id']]['finish_ns']>r['deadline_ns'] for r in requests)
    lat=sorted((rows[r['request_id']]['finish_ns']-r['nominal_release_ns'])/1e6 if rows[r['request_id']] is not None else float('inf') for r in requests);p99=lat[math.ceil(.99*len(lat))-1]
    x=dict(window_start_ns=lo,window_end_ns=hi,published_ns=time.monotonic_ns(),window_seconds=(hi-lo)/1e9,requests=len(requests),timeouts=late,unfinished_at_observation=sum(v is None for v in rows.values()),timeout_rate=late/len(requests),slo_rate=1-late/len(requests),p99_ms=p99 if math.isfinite(p99) else None,p99_status='finite' if math.isfinite(p99) else 'infinite_at_observation',deadline_ms=250,target_timeout_rate=.01,denominator='all nominal releases in window, observed after all deadlines matured')
    p=self.root/'feedback.json';t=p.with_suffix('.feedback.tmp');t.write_text(json.dumps(x));t.replace(p)
    with (self.out/'feedback_windows.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
  except Exception:
   import traceback
   self.error=traceback.format_exc()
 def join(self):
  self.thread.join()
  if self.error:raise RuntimeError(self.error)
