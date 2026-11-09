"""Check pixel equivalence across the machine's reversed EGL device ordering."""
import os
import sys
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.update(CUDA_VISIBLE_DEVICES='3',MUJOCO_EGL_DEVICE_ID='3',
    MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',LIBERO_CONFIG_PATH=str(ROOT/'libero_config'))
import numpy as np
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# robosuite validates a CUDA ordinal at import, but later uses it as an EGL ordinal.
# On this host EGL 0 is PCI 61:00.0 (physical NVIDIA GPU 3).
egl_index=int(sys.argv[1])
os.environ['MUJOCO_EGL_DEVICE_ID']=str(egl_index)
suite=benchmark.get_benchmark_dict()['libero_10'](task_order_index=0)
rows=[]
for task_id in (3,0):
    task=suite.get_task(task_id)
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
    try:
        for index in (30,40):
            seed=20261020+task_id*1000+index
            np.random.seed(seed);env.seed(seed);env.reset()
            obs=env.set_init_state(suite.get_task_init_states(task_id)[index])
            for _ in range(10):obs,_,_,_=env.step([0.,0.,0.,0.,0.,0.,-1.])
            rows.append({'task':task_id,'index':index,'hashes':{
                k:hashlib.sha256(np.asarray(obs[k]).tobytes()).hexdigest()
                for k in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')}})
    finally:env.close()
(ROOT/f'audit/render_egl{egl_index}.json').write_text(json.dumps(rows,indent=2))
