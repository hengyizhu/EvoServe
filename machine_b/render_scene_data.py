"""Re-render pinned official demonstrations for physical camera/texture perturbations."""
from pathlib import Path
import argparse,json,os,sys,hashlib,io
R=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--task',choices=['camera','texture'],required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();w=Path(a.work).resolve();cfg=json.loads((w/'workspace.json').read_text());N=w/'pi05_machine_b_20261115';base=w/'vla_followup_20261020';sys.path[:0]=[str(N),str(R/'vendor/RLinf')];os.environ.update(LIBERO_CONFIG_PATH=str(base/'libero_config'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2')
import numpy as np,h5py,pyarrow as pa,pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scene_perturb import apply_scene,repair_xml
mapping=json.loads((base/'audit/egl_device_mapping.json').read_text());os.environ['MUJOCO_EGL_DEVICE_ID']=str(next(x['egl_index'] for x in mapping if x['cuda_device_attribute']==a.gpu))
split=json.loads((R/f'machine_b/tasks/new_{a.task}_render_split.json').read_text());source=json.loads((R/'machine_b/official_data_source.json').read_text());task=benchmark.get_benchmark_dict()['libero_10'](task_order_index=0).get_task(split['eval_task_id']);item=next(x for x in source['files'] if x['path'].endswith(task.name+'_demo.hdf5'));file=Path(hf_hub_download(source['repo'],item['path'],repo_type='dataset',revision=source['revision'],local_dir=w/'official_demonstrations'))
h=hashlib.sha256()
with file.open('rb') as f:
 while b:=f.read(8*1024*1024):h.update(b)
assert h.hexdigest()==item['lfs']['oid']
root=N/'scene_data'/a.task;root.mkdir(parents=True,exist_ok=True);perturb=json.loads((N/f'tasks/{a.task}.json').read_text())
for ep in {x['episode'] for x in split['replay_windows']}:
 rel=Path(f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet');dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():dest.symlink_to(Path(cfg['data'])/rel)
def png(x):
 b=io.BytesIO();Image.fromarray(np.ascontiguousarray(x)).save(b,format='PNG');return dict(bytes=b.getvalue(),path=None)
env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256);checks=[]
try:
 with h5py.File(file,'r') as f:
  for e in split['target_train']+split['target_validation']:
   d=f['data'][e['source_demo']];ep=e['episode_index'];dest=root/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet';dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():assert pq.read_metadata(dest).num_rows==e['length'];continue
   xml=d.attrs['model_file'];xml=xml.decode() if isinstance(xml,bytes) else str(xml);env.reset();env.reset_from_xml_string(repair_xml(xml));env.sim.forward();apply_scene(env,perturb);rows=[]
   for i in range(e['length']):
    obs=env.regenerate_obs_from_state(d['states'][i+1]);err=float(np.max(np.abs(obs['robot0_joint_pos']-d['obs/joint_states'][i])));assert err==0,('state/observation mismatch',ep,i,err)
    rows.append(dict(image=png(obs['agentview_image'][::-1,::-1]),wrist_image=png(obs['robot0_eye_in_hand_image'][::-1,::-1]),state=np.concatenate([d['obs/ee_states'][i],d['obs/gripper_states'][i]]).astype(np.float32).tolist(),actions=np.asarray(d['actions'][i],dtype=np.float32).tolist(),task_index=split['target_data_index'],episode_index=ep,frame_index=i,timestamp=i/20))
   tmp=dest.with_suffix('.tmp');pq.write_table(pa.Table.from_pylist(rows),tmp,compression='zstd');tmp.replace(dest);checks.append(dict(episode=ep,frames=len(rows),joint_alignment_exact=True));print(checks[-1],flush=True)
finally:env.close()
split.update(data_root=str(root),source_hdf5=str(file));(N/f'splits/new_{a.task}.json').write_text(json.dumps(split,indent=2));(root/'render_manifest.json').write_text(json.dumps(dict(source_revision=source['revision'],source_sha256=h.hexdigest(),perturbation=perturb,alignment='states[i+1] corresponds to originalobs[i]; verified exact joint positions everyframe',episodes=checks),indent=2))
