"""Generate local EGL and LIBERO paths on the allocated compute node; no model load."""
import argparse,ctypes,json,importlib.util
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',required=True);a=p.parse_args();w=Path(a.work).resolve()
E=ctypes.CDLL('libEGL.so.1');E.eglGetProcAddress.restype=ctypes.c_void_p;E.eglGetProcAddress.argtypes=[ctypes.c_char_p]
query=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_int,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int))(E.eglGetProcAddress(b'eglQueryDevicesEXT'))
attr=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_ssize_t))(E.eglGetProcAddress(b'eglQueryDeviceAttribEXT'))
dev=(ctypes.c_void_p*64)();n=ctypes.c_int();assert query(64,dev,ctypes.byref(n));mapping=[]
for i in range(n.value):
 v=ctypes.c_ssize_t()
 if attr(dev[i],0x323A,ctypes.byref(v)):mapping.append({'egl_index':i,'cuda_device_attribute':v.value})
assert mapping,'NVIDIA EGL device mapping unavailable; load the compute-node GL driver'
spec=importlib.util.find_spec('libero');root=Path(next(iter(spec.submodule_search_locations)))/'libero'
import yaml
config={k:str(root/v) for k,v in {'assets':'assets','bddl_files':'bddl_files','benchmark_root':'.','datasets':'../datasets','init_states':'init_files'}.items()}
for name in ['vla_followup_20261020','vla_ladder_20261020']:
 (w/name/'audit/egl_device_mapping.json').write_text(json.dumps(mapping,indent=2));(w/name/'libero_config/config.yaml').write_text(yaml.safe_dump(config))
print(json.dumps({'egl':mapping,'libero':config},indent=2))
