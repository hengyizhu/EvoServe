"""CPU-only LIBERO rendering preflight; does not load a policy or touch CUDA."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from evoserve.simulation import ObservationRenderer
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv

p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
assert os.environ['MUJOCO_GL']=='osmesa' and os.environ['CUDA_VISIBLE_DEVICES']==''
suite=benchmark.get_benchmark_dict()['libero_10'](task_order_index=0);task=suite.get_task(3)
started=time.monotonic();env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
try:
 env.seed(20261108);env.reset();obs=env.set_init_state(suite.get_task_init_states(3)[0]);dur=[];renders=[];renderer=ObservationRenderer(env)
 for i in range(20):
  if i%5==0:
   t=time.monotonic();obs=renderer.observation();renders.append(time.monotonic()-t)
  t=time.monotonic();_,_,_,_=env.step([0.,0.,0.,0.,0.,0.,-1.]);dur.append(time.monotonic()-t)
 assert obs['agentview_image'].shape==(256,256,3)
 import torch
 assert not torch.cuda.is_initialized(),'CPU simulation initialized CUDA'
 result=dict(state='passed',renderer='osmesa',cuda_initialized=False,frames=20,median_step_s=float(np.median(dur)),render_median_s=float(np.median(renders)),render_frequency='one observation per 5 executed actions',p99_step_s=float(np.percentile(dur,99)),total_seconds=time.monotonic()-started,cpu_affinity=sorted(os.sched_getaffinity(0)),image_shape=list(obs['agentview_image'].shape))
 Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
finally:env.close()
