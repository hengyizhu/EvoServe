import tempfile,threading,unittest
from pathlib import Path
from evoserve.io import RPCClient,RPCServer

class TransportTests(unittest.TestCase):
    def test_concurrent_calls_keep_responses_paired(self):
        with tempfile.TemporaryDirectory(prefix='ep_test_') as d:
            server=RPCServer(Path(d)/'rpc',lambda m:m['value']*3);client=RPCClient(Path(d)/'rpc')
            results={}
            def work(i):results[i]=client.call('test',value=i)
            threads=[threading.Thread(target=work,args=(i,)) for i in range(12)]
            try:
                for t in threads:t.start()
                for t in threads:t.join(timeout=5)
                self.assertEqual(results,{i:i*3 for i in range(12)})
            finally:client.close();server.close()
