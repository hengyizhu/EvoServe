from pathlib import Path
import ctypes
lib=ctypes.CDLL(str(Path(__file__).with_name('_gil_probe_native.so')));lib.sample_gil_wait_ms.restype=ctypes.c_double;lib.sample_gil_wait_ms.argtypes=[]
def sample():return lib.sample_gil_wait_ms()
