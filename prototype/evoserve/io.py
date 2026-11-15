from pathlib import Path
import json
import os
import threading
import time
import traceback
from multiprocessing.connection import Listener,Client


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False));tmp.replace(path)


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


_log_lock=threading.Lock()
def append(path,data):
    with _log_lock:
        with Path(path).open('a') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n')


class RPCClient:
    def __init__(self,address):self.address=str(address);self.connection=None;self.lock=threading.RLock()
    def call(self,op,**payload):
        with self.lock:return self._call(op,**payload)
    def _call(self,op,**payload):
        if self.connection is None:self.connection=Client(self.address,family='AF_UNIX',authkey=b'evoserve-v0-local')
        self.connection.send(dict(op=op,**payload))
        if not self.connection.poll(600):raise TimeoutError((self.address,op))
        result=self.connection.recv()
        if result.get('error'):raise RuntimeError(result['error'])
        return result['value']
    def close(self):
        if self.connection:self.connection.close();self.connection=None


class RPCServer:
    def __init__(self,address,handler):
        self.address=Path(address);self.handler=handler;self.address.unlink(missing_ok=True)
        self.listener=Listener(str(self.address),family='AF_UNIX',authkey=b'evoserve-v0-local')
        self.thread=threading.Thread(target=self.accept,daemon=True);self.thread.start()
    def accept(self):
        while True:
            try:connection=self.listener.accept()
            except (OSError,EOFError):return
            threading.Thread(target=self.client,args=(connection,),daemon=True).start()
    def client(self,connection):
        try:
            while True:
                msg=connection.recv()
                try:reply={'value':self.handler(msg)}
                except Exception:reply={'error':traceback.format_exc()}
                connection.send(reply)
        except (EOFError,OSError):pass
        finally:connection.close()
    def close(self):self.listener.close();self.address.unlink(missing_ok=True)


def wait_file(path,processes=(),seconds=900):
    until=time.monotonic()+seconds
    while not Path(path).exists():
        for p in processes:
            if p.poll() is not None:raise RuntimeError(f'worker {p.pid} exited {p.returncode}')
        if time.monotonic()>until:raise TimeoutError(str(path))
        time.sleep(.1)
    return read(path)
