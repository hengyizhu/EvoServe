from pathlib import Path
import ctypes,os,subprocess,sysconfig
native=Path(__file__).with_name('_gil_probe_native.so')
if not native.exists():
 temp=native.with_name(native.name+f'.{os.getpid()}.tmp')
 subprocess.run(['cc','-O2','-shared','-fPIC','-I'+sysconfig.get_path('include'),str(Path(__file__).with_suffix('.c')),'-o',str(temp)],check=True)
 temp.replace(native)
lib=ctypes.CDLL(str(native));lib.sample_gil_wait_ms.restype=ctypes.c_double;lib.sample_gil_wait_ms.argtypes=[]
def sample():return lib.sample_gil_wait_ms()
