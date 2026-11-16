import tempfile,time,unittest
from pathlib import Path
from evoserve.io import save
from evoserve.operators import operator_token,wait_operator_result
class RemoteOperatorTests(unittest.TestCase):
 def test_source_ids_do_not_collide(self):
  self.assertNotEqual(operator_token('/serve0',0),operator_token('/serve1',0))
 def test_deadline_and_identity(self):
  with tempfile.TemporaryDirectory() as root:
   p=Path(root)/'result.json';ref=dict(path=str(p),request_id='robot0:1')
   self.assertEqual(wait_operator_result(ref,time.monotonic())['status'],'operator_deadline_miss')
   save(p,dict(request_id='robot0:1',status='complete',actions=[[1]*7]*10))
   self.assertEqual(wait_operator_result(ref,time.monotonic()+1)['status'],'complete');self.assertFalse(p.exists())
   save(p,dict(request_id='robot1:1',status='complete'))
   with self.assertRaises(ValueError):wait_operator_result(ref,time.monotonic()+1)
