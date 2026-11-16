import queue,threading,time,unittest
from evoserve.batching import collect_batch
class BatchTests(unittest.TestCase):
 def item(self,op='infer',id=0):return ({'op':op,'request_id':id},{},threading.Event())
 def test_mixed_versions_and_install_barrier(self):
  q=queue.Queue();a=self.item(id=1);b=self.item(id=2);barrier=self.item('install');q.put(b);q.put(barrier);q.put(self.item(id=3))
  batch,deferred=collect_batch(a,q,8,10)
  self.assertEqual([x[0]['request_id'] for x in batch],[1,2]);self.assertIs(deferred,barrier);self.assertEqual(q.qsize(),1)
 def test_cap_and_bounded_wait(self):
  q=queue.Queue()
  for i in range(5):q.put(self.item(id=i))
  batch,deferred=collect_batch(self.item(id=9),q,4,20);self.assertEqual(len(batch),4);self.assertEqual(q.qsize(),2)
  t=time.monotonic();batch,_=collect_batch(self.item(),queue.Queue(),8,10)
  self.assertLess(time.monotonic()-t,.1);self.assertEqual(len(batch),1)

class TimingClassificationTests(unittest.TestCase):
 def test_stale_camera_is_not_a_missed_emission_period(self):
  from evoserve.core import cycle_timing_flags
  self.assertEqual(cycle_timing_flags(.001,.438),dict(simulation_overrun=False,observation_stale=True))
  self.assertEqual(cycle_timing_flags(.251,.010),dict(simulation_overrun=True,observation_stale=False))
  self.assertTrue(cycle_timing_flags(0,None)['simulation_overrun'])
