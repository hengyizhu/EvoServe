"""Actual replay arrivals, independent of inference completion, exposed via shared mmap."""
import bisect,json,mmap,struct,threading,time
from pathlib import Path
SIZE=4096
class Arrivals:
 def __init__(self,root,out,plan,origin):
  self.path=Path(root)/'arrival_state.bin';self.file=self.path.open('w+b');self.file.truncate(SIZE);self.mm=mmap.mmap(self.file.fileno(),SIZE)
  self.out=Path(out);self.plan=plan;self.origin=origin;self.arrived=self.started=self.finished_count=self.seq=0;self.observed=[];self.lock=threading.Lock();self.error=None
  self._publish();self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
 def _publish(self):
  self.seq+=2;struct.pack_into('Q',self.mm,0,self.seq-1);struct.pack_into('5Q',self.mm,8,self.origin,self.arrived,self.started,self.finished_count,time.monotonic_ns());struct.pack_into('Q',self.mm,0,self.seq)
 def release_due(self,now):
  with self.lock:
   entries=[]
   while self.arrived<len(self.plan) and self.plan[self.arrived]['nominal_release_ns']<=now:
    req=self.plan[self.arrived];self.observed.append(now);entries.append(dict(request_id=req['request_id'],nominal_release_ns=req['nominal_release_ns'],arrival_observed_ns=now));self.arrived+=1
   if entries:
    self._publish()
    with (self.out/'arrivals.jsonl').open('a') as f:f.write(''.join(json.dumps(x)+'\n' for x in entries))
 def run(self):
  try:
   while True:
    with self.lock:
     if self.arrived>=len(self.plan):break
     due=self.plan[self.arrived]['nominal_release_ns']
    time.sleep(max(0,(due-time.monotonic_ns())/1e9));self.release_due(time.monotonic_ns())
  except Exception:
   import traceback;self.error=traceback.format_exc()
 def before_start(self,request_id):
  self.release_due(time.monotonic_ns())
  with self.lock:
   assert self.plan[self.started]['request_id']==request_id and self.started<self.arrived
   self.started+=1;self._publish()
 def complete(self):
  with self.lock:self.finished_count+=1;self._publish()
 def window(self,lo,hi):
  with self.lock:return dict(arrivals_observed=bisect.bisect_left(self.observed,hi)-bisect.bisect_left(self.observed,lo),queue_depth=self.arrived-self.started)
 def close(self):
  self.thread.join()
  if self.error:raise RuntimeError(self.error)
  self.mm.close();self.file.close()

class Reader:
 def __init__(self,root,origin):self.path=Path(root)/'arrival_state.bin';self.origin=origin;self.file=None;self.mm=None
 def read(self):
  if self.mm is None:
   if not self.path.exists() or self.path.stat().st_size<SIZE:return None
   self.file=self.path.open('rb');self.mm=mmap.mmap(self.file.fileno(),SIZE,access=mmap.ACCESS_READ)
  for _ in range(20):
   a=struct.unpack_from('Q',self.mm,0)[0]
   if a%2:continue
   origin,arrived,started,finished,updated=struct.unpack_from('5Q',self.mm,8);b=struct.unpack_from('Q',self.mm,0)[0]
   if a==b and b%2==0 and origin==self.origin and arrived>=started>=finished:return dict(arrived=arrived,started=started,finished=finished,queue_depth=arrived-started,updated_ns=updated,checked_ns=time.monotonic_ns())
  return None
 def close(self):
  if self.mm is not None:self.mm.close();self.file.close();self.mm=None
