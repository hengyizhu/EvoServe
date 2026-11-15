"""CPU physics + independent CPU renderer at 20Hz/4Hz; no CUDA policy."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evoserve.simulation import AsyncRenderer
from evoserve.io import save
from libero.libero import benchmark,get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seconds',type=int,default=10);a=p.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES']==''
suite=benchmark.get_benchmark_dict()['libero_10'](task_order_index=0);task=suite.get_task(3)
bddl=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file)
env=ControlEnv(bddl_file_name=bddl,use_camera_obs=False,has_offscreen_renderer=False,hard_reset=False);renderer=AsyncRenderer(bddl)
try:
 env.seed(20261108);env.reset();env.set_init_state(suite.get_task_init_states(3)[0]);renderer.submit('e0',0,env.get_sim_state().copy())
 while renderer.get('e0') is None:time.sleep(.002)
 fresh=time.monotonic();renderer.submit('e0',0,env.get_sim_state().copy())
 while renderer.get('e0')['captured']<fresh:time.sleep(.002)
 origin=time.monotonic()+.01;rows=[]
 for tick in range(20*a.seconds):
  release=origin+tick/20;time.sleep(max(0,release-time.monotonic()))
  if tick%5==0:
   view=renderer.get('e0');rows.append(dict(request=tick//5,release=release,dispatch=time.monotonic(),frame=view['frame'],observation_age_s=release-view['captured']))
  env.step([0.,0.,0.,0.,0.,0.,-1.])
  if tick%5==1:renderer.submit('e0',tick+1,env.get_sim_state().copy())
 import torch
 assert not torch.cuda.is_initialized()
 missed=sum(x['dispatch']-x['release']>=.25 or x['observation_age_s']>=.25 for x in rows)
 save(a.out,dict(state='passed' if missed==0 else 'simulation_cannot_sustain_period',cuda_initialized=False,nominal_requests_per_s=4,requests=len(rows),overrun_cycles=missed,maximum_request_dispatch_lag_s=max(x['dispatch']-x['release'] for x in rows),maximum_observation_age_s=max(x['observation_age_s'] for x in rows),rows=rows))
 print(json.dumps(dict(requests=len(rows),simulation_overrun_cycles=missed)))
finally:renderer.close();env.close()
