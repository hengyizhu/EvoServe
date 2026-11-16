"""CPU integration: batching must preserve each reply and exported KV identity."""
import tempfile,threading,time,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from evoserve.workers import serve

class BatchWorkerTests(unittest.TestCase):
 def test_each_request_keeps_its_reply_and_cache(self):
  registry={};exports=[];errors=[]
  class RPC:
   def __init__(self,address,handler):registry['handle']=handler
   def close(self):pass
  class Model:
   def __init__(self,*args):self.prefix=(np.zeros((1,1)),);self.static=types.SimpleNamespace(state=np.zeros((1,1)))
   def install(self,*args):pass
   def map(self,fn,value):return tuple(fn(x) for x in value)
   def infer_batch(self,messages,prefix_hook=None):
    ids=[m['seed'] for m in messages];prefix=(np.array(ids)[:,None],);n=len(ids)
    return np.array([np.full((10,7),i) for i in ids]),prefix,dict(start_ns=time.monotonic_ns(),finish_ns=time.monotonic_ns(),batch_size=n,graph_batch_size=2,prefix_ms=1,denoise_ms=1)
  class Export:
   def __init__(self,out,values):(out/'ipc').mkdir()
   def publish(self,ident,values,record,ready_event=None):exports.append((ident,np.array(values[0]).copy()));return dict(record,slot=ident)
   def close(self):pass
  with tempfile.TemporaryDirectory() as root:
   with patch.dict('sys.modules',{'evoserve.batch_backend':types.SimpleNamespace(BatchedGraphModel=Model),'evoserve.feature_ipc':types.SimpleNamespace(Export=Export)}),patch('evoserve.workers.RPCServer',RPC):
    cfg=dict(out=root,assets='unused',experts=[3,5],export_cache=True,batch_sizes=[1,2],batch_wait_ms=50,socket='unused')
    def main():
     try:serve(cfg)
     except BaseException as exc:errors.append(exc)
    worker=threading.Thread(target=main,daemon=True);worker.start()
    until=time.monotonic()+2
    while not (Path(root)/'ready.json').exists() and time.monotonic()<until:time.sleep(.001)
    self.assertIn('handle',registry);barrier=threading.Barrier(2);replies={}
    def caller(seed,expert):
     barrier.wait();replies[seed]=registry['handle'](dict(op='infer',raw={},expert=expert,version=f'{expert}:v0',seed=seed,request_id=str(seed),episode='e',frame=seed,operator=False))
    threads=[threading.Thread(target=caller,args=(11,3)),threading.Thread(target=caller,args=(22,5))]
    for thread in threads:thread.start()
    for thread in threads:thread.join(timeout=2)
    registry['handle'](dict(op='stop'));worker.join(timeout=2)
    self.assertFalse(errors);self.assertFalse(worker.is_alive());self.assertEqual(len(replies),2)
    exported=dict(exports)
    for seed,reply in replies.items():
     np.testing.assert_array_equal(reply['actions'],np.full((10,7),seed))
     self.assertEqual(reply['timing']['batch_size'],2)
     np.testing.assert_array_equal(exported[reply['cache_ref']['id']],[[seed]])
