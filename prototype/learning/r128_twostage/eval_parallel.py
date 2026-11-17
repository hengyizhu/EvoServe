"""Six LIBERO simulator processes sharing one GPU policy, fixed episode seeds."""
import argparse, hashlib, json, multiprocessing as mp, os, time, traceback
from pathlib import Path
from multiprocessing.connection import wait

ROOT=Path('/home/hengyi/Workspace/01_VLA-Serving/Environment-and-Earlier-Runs/vla_followup_20261020')

def simulator(conn,task_id,gpu,worker,perturb,budget):
 try:
  os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',LIBERO_CONFIG_PATH=str(ROOT/'libero_config'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
  os.sched_setaffinity(0,{gpu*16+4+2*worker,gpu*16+5+2*worker})
  import sys;sys.path.insert(0,str(ROOT/'RLinf'))
  import numpy as np
  from libero.libero import benchmark,get_libero_path
  from libero.libero.envs import OffScreenRenderEnv
  from rlinf.envs.sim.libero.utils import quat2axisangle
  from perturbations import transform_image
  mapping=json.loads((ROOT/'audit/egl_device_mapping.json').read_text())
  os.environ['MUJOCO_EGL_DEVICE_ID']=str(next(x['egl_index'] for x in mapping if x['cuda_device_attribute']==gpu))
  suite=benchmark.get_benchmark_dict()['libero_10'](task_order_index=0);task=suite.get_task(task_id);states=suite.get_task_init_states(task_id)
  assert perturb['target_prompt']==task.language
  env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
  conn.send(dict(kind='ready',worker=worker,pid=os.getpid()))
  try:
   while True:
    cmd=conn.recv()
    if cmd['op']=='stop':break
    assert cmd['op']=='episode';index,seed=cmd['index'],cmd['seed'];np.random.seed(seed);env.seed(seed);env.reset();obs=env.set_init_state(states[index])
    for _ in range(10):obs,*_=env.step([0.,0.,0.,0.,0.,0.,-1.])
    start=time.monotonic();steps=0;success=False
    while steps<budget and not success:
     state=np.concatenate([obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']]).astype(np.float32)
     inp=dict(main_images=transform_image(np.ascontiguousarray(obs['agentview_image'][::-1,::-1])[None],perturb),wrist_images=transform_image(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])[None],perturb),states=state[None],task_descriptions=[task.language])
     conn.send(dict(kind='observation',index=index,inputs=inp));reply=conn.recv();assert reply['op']=='actions'
     for action in reply['actions']:
      obs,*_=env.step(action);steps+=1;success=bool(env.check_success())
      if success or steps>=budget:break
    conn.send(dict(kind='result',index=index,task=task_id,seed=seed,success=success,policy_steps=steps,wall_seconds=time.monotonic()-start,state_sha256=hashlib.sha256(np.asarray(states[index]).tobytes()).hexdigest()))
  finally:env.close()
 except BaseException:
  try:conn.send(dict(kind='error',traceback=traceback.format_exc()))
  except Exception:pass
  raise
 finally:conn.close()

def main(a):
 import sys;sys.path.insert(0,str(ROOT/'RLinf'))
 os.environ.update(VLA_GPU=str(a.gpu),CUDA_VISIBLE_DEVICES=str(a.gpu),LIBERO_CONFIG_PATH=str(ROOT/'libero_config'),JAX_PLATFORMS='cpu',TOKENIZER_DIR=str(ROOT/'assets/.cache/openpi'))
 import numpy as np,torch
 from omegaconf import OmegaConf
 from rlinf.models.embodiment.openpi import get_model
 from rlinf.envs.action_utils import prepare_actions_for_libero
 from common import setup_family,save
 protocol=json.loads((ROOT/'experiment.json').read_text());perturb=json.loads(Path(a.perturb).read_text());out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 expected=dict(delta=str(Path(a.delta).resolve()),task=a.task,perturb=perturb,episodes=a.episodes,workers=a.workers,renderer='egl',noise='per-episode CUDA generator, independent of batching order',smoke=a.smoke)
 manifest=out.with_suffix('.protocol.json')
 if manifest.exists():assert json.loads(manifest.read_text())==expected
 else:save(manifest,expected)
 rows=[]
 if out.exists():
  lines=out.read_text().splitlines();valid=[]
  for i,line in enumerate(lines):
   try:r=json.loads(line)
   except json.JSONDecodeError:
    assert i==len(lines)-1;out.rename(out.with_suffix(f'.truncated.{time.time_ns()}'));out.write_text('\n'.join(valid)+'\n');break
   assert r['task']==a.task and r['seed']==protocol['seed']+a.task*1000+r['index'];rows.append(r);valid.append(line)
 done={r['index'] for r in rows};assert len(done)==len(rows) and done<=set(range(a.episodes))
 pending=[i for i in range(a.episodes) if i not in done]
 if not pending:return
 cfg=OmegaConf.load(ROOT/'RLinf/examples/embodiment/config/model/pi0_5.yaml');cfg.model_path=str(ROOT/'parent');cfg.num_action_chunks=5;cfg.openpi.task='eval';cfg.openpi.action_horizon=10;cfg.openpi.discrete_state_input=False;cfg.openpi.torch_compile=False;cfg.openpi_data={'norm_stats_path':str(ROOT/'parent/physical-intelligence/libero/norm_stats.json')}
 model=get_model(cfg).cuda().eval();delta=torch.load(a.delta,map_location='cpu',weights_only=True)
 assert delta['parent']==json.loads((ROOT/'audit/parent_download.json').read_text()) and delta['eval_task_id']==a.task and delta['weights_kind']=='EMA'
 pairs=setup_family(model,delta['family']);assert set(delta['parameters'])=={n for n,p in pairs};assert not model.load_state_dict(delta['parameters'],strict=False).unexpected_keys
 model.gradient_checkpointing_disable();model.eval();del delta;torch.set_num_threads(4)
 for thread in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(thread.name),set(range(a.gpu*16,a.gpu*16+4)))
  except ProcessLookupError:pass
 budget=protocol['eval']['policy_action_budget'];context=mp.get_context('spawn');children=[];active={};generators={};latencies={}
 def assign(conn):
  if not pending:return False
  index=pending.pop(0);seed=protocol['seed']+a.task*1000+index;active[conn]=index;generators[index]=torch.Generator(device='cuda').manual_seed(seed);latencies[index]=[];conn.send(dict(op='episode',index=index,seed=seed));return True
 try:
  for worker in range(a.workers):
   conn,child=context.Pipe();proc=context.Process(target=simulator,args=(child,a.task,a.gpu,worker,perturb,budget));proc.start();child.close();children.append((conn,proc))
  for conn,proc in children:
   assert conn.poll(180),'Simulator initialization timed out';msg=conn.recv();assert msg['kind']=='ready',msg;assign(conn)
  while active:
   ready=wait(list(active),timeout=180);assert ready,'Simulation stalled for 180 seconds'
   # Brief batching window for CPU workers; episode seeds remain independent.
   time.sleep(.002);ready=list(dict.fromkeys(ready+[c for c in active if c.poll()]));observations=[]
   for conn in ready:
    msg=conn.recv()
    if msg['kind']=='error':raise RuntimeError(msg['traceback'])
    if msg['kind']=='result':
     index=active.pop(conn);assert index==msg['index'];msg.update(version=Path(a.delta).parent.name,checkpoint=str(a.delta),smoke_only=a.smoke,request_seconds=latencies.pop(index),peak_gpu_bytes=torch.cuda.max_memory_allocated(),simulator_processes=a.workers)
     del generators[index];msg.pop('kind');
     with out.open('a') as f:f.write(json.dumps(msg)+'\n');f.flush()
     done.add(index);save(out.with_suffix('.status.json'),dict(state='evaluating',completed=len(done),episodes=a.episodes));assign(conn)
    else:
     assert msg['kind']=='observation' and msg['index']==active[conn];observations.append((conn,msg))
   if observations:
    inputs={k:(sum([m['inputs'][k] for _,m in observations],[]) if k=='task_descriptions' else np.concatenate([m['inputs'][k] for _,m in observations],axis=0)) for k in observations[0][1]['inputs']}
    noise=torch.cat([torch.randn(1,model.action_horizon,model.action_dim,device='cuda',generator=generators[m['index']]) for _,m in observations],0)
    start=time.monotonic()
    with torch.inference_mode():actions,_=model.predict_action_batch(inputs,mode='eval',noise=noise)
    assert torch.isfinite(actions).all();actions=prepare_actions_for_libero(actions.clone(),'openpi').cpu().numpy();elapsed=time.monotonic()-start
    for j,(conn,msg) in enumerate(observations):latencies[msg['index']].append(elapsed);conn.send(dict(op='actions',actions=actions[j].tolist()))
  assert done==set(range(a.episodes));save(out.with_suffix('.status.json'),dict(state='complete',completed=len(done),episodes=a.episodes))
 finally:
  for conn,proc in children:
   try:conn.send(dict(op='stop'))
   except Exception:pass
  for conn,proc in children:
   proc.join(10)
   if proc.is_alive():proc.terminate();proc.join(5)
   conn.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--delta',required=True);p.add_argument('--task',type=int,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--perturb',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--episodes',type=int,default=50);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 main(a)
