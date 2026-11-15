"""Real fixed-period simulator against a CPU fake server (timing test, no quality claim)."""
import argparse,os,sys,time,tempfile,subprocess,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evoserve.io import RPCServer,save,wait_file
from evoserve.core import DeadlineLedger
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--osmesa-lib',required=True);p.add_argument('--out',required=True);p.add_argument('--seconds',type=int,default=35);p.add_argument('--operator',action='store_true');a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);sock=Path(tempfile.mkdtemp(prefix='ep_cpu_'));clock={};ledger=DeadlineLedger();episodes=[];takeovers=[];windows=[];operator_requests=0
def coord(m):
 op=m['op']
 if op=='begin_episode':
  if not clock or time.monotonic()<clock['origin']:return dict(wait=True,origin=clock.get('origin'))
  return dict(stop=time.monotonic()>=clock['end'],version='3:v0',epoch=0,**clock)
 if op=='environment':return dict(stop=time.monotonic()>=clock['end'],epoch=0,perturb=None)
 if op=='arrive':ledger.arrive(m['request_id'],m['release'],m['expert'],m['robot']);return True
 if op=='finish':ledger.finish(m['request_id'],m['finish']);return True
 if op=='end_episode':episodes.append(m);return True
 if op=='takeover':takeovers.append(m);return True
 if op=='window':windows.append(m);return dict(accepted=True)
 if op in ['drop','abort_episode']:return True
 raise ValueError(op)
def serve(m):
 global operator_requests
 operator_requests+=int(m.get('operator',False))
 time.sleep(.15)
 block=np.tile(np.array([0,0,0,0,0,0,-1],np.float32),(10,1))
 return dict(actions=block,operator_actions=block.copy() if m.get('operator') else None,cache_ref=None)
c=RPCServer(sock/'coord',coord);s=RPCServer(sock/'serve',serve)
cfg=dict(out=str(out/'robot'),coordinator=str(sock/'coord'),server=str(sock/'serve'),expert=3,task_id=3,robot=0,seed=20261108,control_hz=20,action_budget=600,cpus=[16,17],phase_s=0,operator_enabled=a.operator,stagnation_distance_m=.002)
save(out/'config.json',cfg)
env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]),CUDA_VISIBLE_DEVICES='',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LIBERO_CONFIG_PATH=str(Path(a.assets)/'libero_config'),LP_NUM_THREADS='1',GALLIUM_DRIVER='llvmpipe',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH=a.osmesa_lib)
with (out/'robot.log').open('w') as log:
 child=subprocess.Popen([sys.executable,'-u','-m','evoserve.robot','--config',str(out/'config.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
try:
 wait_file(out/'robot/ready.json',[child]);clock.update(origin=time.monotonic()+.5);clock['end']=clock['origin']+a.seconds
 rc=child.wait(timeout=a.seconds+30)
 assert rc==0,(rc,(out/'robot.log').read_text()[-3000:])
 time.sleep(.3);stat=ledger.window(clock['origin'],clock['end'],time.monotonic())
 counters=json.loads((out/'robot/final_counters.json').read_text())
 if a.operator:
  assert takeovers and operator_requests>0,'operator did not take over'
  assert windows and all(w['data_source']=='operator_correction' and w['action_sources']==['operator']*10 for w in windows),'correction alignment invalid'
  assert all(not e['autonomous_success'] for e in episodes if e['intervened'])
 save(out/'summary.json',dict(state='passed',operator_enabled=a.operator,takeovers=len(takeovers),operator_requests=operator_requests,accepted_correction_windows=len(windows),autonomous_successes=sum(e['autonomous_success'] for e in episodes),fake_server=True,no_learning_or_success_claim=True,episodes=len(episodes),counters=counters,**stat));print(json.dumps(stat))
finally:
 if child.poll() is None:child.terminate();child.wait(timeout=10)
 c.close();s.close()
