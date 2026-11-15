"""Fixed-state LIBERO evaluation of the pinned parent or a parameter delta."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path("/home/hengyi/Workspace/01_VLA-Serving/Environment-and-Earlier-Runs/vla_followup_20261020")
OUT = Path("/home/hengyi/pi05_machine_b_20261115")
sys.path.insert(0, str(ROOT / "RLinf"))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "libero_config"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ.get("CUDA_VISIBLE_DEVICES", "3").split(",")[0])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "4")

gpu=int(os.environ.get("VLA_GPU","0"))
os.sched_setaffinity(0,set(range(gpu*8,gpu*8+8)))
os.environ.update(OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4")
import numpy as np
import torch
from omegaconf import OmegaConf
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from rlinf.models.embodiment.openpi import get_model
from rlinf.envs.sim.libero.utils import quat2axisangle
from rlinf.envs.action_utils import prepare_actions_for_libero


def load_model():
    cfg = OmegaConf.load(ROOT / "RLinf/examples/embodiment/config/model/pi0_5.yaml")
    cfg.model_path = str(ROOT / "parent")
    cfg.num_action_chunks = 5
    cfg.openpi.task = "eval"
    cfg.openpi.action_horizon = 10
    cfg.openpi.discrete_state_input = False
    cfg.openpi.torch_compile = False
    cfg.openpi_data = {
        "norm_stats_path": str(ROOT / "parent/physical-intelligence/libero/norm_stats.json")
    }
    return get_model(cfg).cuda().eval()


def env_input(obs, language):
    state = np.concatenate([obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"].copy()),
                            obs["robot0_gripper_qpos"]])
    return {
        "main_images": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])[None],
        "wrist_images": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])[None],
        "states": state.astype(np.float32)[None],
        "task_descriptions": [language],
    }


def main():
    # CUDA3 is EGL0 on this host. robosuite's import-time ordinal check has
    # already run; its context constructor below needs the EGL ordinal.
    device_map = json.loads((ROOT / 'audit/egl_device_mapping.json').read_text())
    physical = int(os.environ['CUDA_VISIBLE_DEVICES'])
    egl_index = next(x['egl_index'] for x in device_map if x['cuda_device_attribute'] == physical)
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(egl_index)
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="parent")
    parser.add_argument("--delta")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--task", type=int)
    parser.add_argument("--brightness", type=float, default=1.0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Keep validated completed episodes and run only missing indices")
    parser.add_argument("--perturb")
    parser.add_argument("--adapt-task", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol = json.loads((ROOT / "experiment.json").read_text())
    state_manifest = json.loads((ROOT / "audit/independent_states.json").read_text())
    suite = benchmark.get_benchmark_dict()["libero_10"](task_order_index=0)
    split = json.loads((ROOT / "data_split.json").read_text())
    assert suite.get_task(3).language == split["target"]
    assert suite.get_task(0).language == split["history"]
    if args.perturb:
        pcfg=json.loads(Path(args.perturb).read_text())
        assert pcfg["target_prompt"]==suite.get_task(args.adapt_task).language
    model = load_model()
    if args.delta:
        delta = torch.load(args.delta, map_location="cpu", weights_only=True)
        parent = json.loads((ROOT / "audit/parent_download.json").read_text())
        assert delta["parent"] == parent, "Checkpoint parent identity mismatch"
        if "eval_task_id" in delta:assert delta["eval_task_id"]==args.adapt_task
        from common import setup_family
        pairs=setup_family(model,delta["family"])
        assert set(delta["parameters"])=={n for n,p in pairs}
        assert delta["weights_kind"]=="EMA"
        unexpected = model.load_state_dict(delta["parameters"], strict=False).unexpected_keys
        assert not unexpected, unexpected
    from common import pin_threads
    pin_threads()
    model.eval()
    result_dir = OUT / "results"
    result_dir.mkdir(exist_ok=True)
    result_file = result_dir / f"{args.name}{'_smoke' if args.smoke else ''}.jsonl"
    completed = set()
    if result_file.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite {result_file}")
        for line in result_file.read_text().splitlines():
            row = json.loads(line)
            key = (row['task'], row['index'])
            assert key not in completed, 'Duplicate completed episode'
            assert row['task'] in ([args.task] if args.task is not None else [args.adapt_task, 0])
            assert args.start <= row['index'] < args.end
            assert row['version'] == args.name and row['smoke_only'] == args.smoke
            assert row['target_brightness'] == args.brightness and row['target_mirror'] == args.mirror
            assert row.get('perturb') == args.perturb
            assert row['seed'] == protocol['seed'] + row['task'] * 1000 + row['index']
            completed.add(key)
        print(json.dumps({'resuming_completed_episodes': len(completed)}), flush=True)
    manifest = []
    for task_id in ([args.task] if args.task is not None else [args.adapt_task,0]):
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        indices = list(range(args.start,args.end))
        for index in indices:
            manifest.append({"task":task_id,"index":index,"state_sha256":hashlib.sha256(np.asarray(states[index]).tobytes()).hexdigest()})
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=256, camera_widths=256)
        try:
            for index in indices:
                if (task_id, index) in completed:
                    continue
                seed = protocol["seed"] + task_id * 1000 + index
                np.random.seed(seed)
                torch.manual_seed(seed)
                env.seed(seed)
                env.reset()
                if task_id == args.adapt_task and args.perturb and pcfg['kind'] in ['camera_orbit','table_texture']:
                    from scene_perturb import apply_scene
                    scene_record=apply_scene(env,pcfg)
                    (result_dir/f'{args.name}_scene.json').write_text(json.dumps(scene_record,indent=2))
                obs = env.set_init_state(states[index])
                # Match standard LIBERO reset settling; excluded from policy budget.
                for _ in range(10):
                    obs, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
                success = False
                steps = 0
                latencies = []
                start = time.monotonic()
                budget = 5 if args.smoke else protocol["eval"]["policy_action_budget"]
                while steps < budget and not success:
                    inputs = env_input(obs, task.language)
                    if task_id == args.adapt_task and args.perturb:
                        from perturbations import transform_image
                        pcfg=json.loads(Path(args.perturb).read_text())
                        for key in ['main_images','wrist_images']:inputs[key]=transform_image(inputs[key],pcfg)
                    if task_id == args.adapt_task and args.brightness != 1.0:
                        for key in ["main_images","wrist_images"]: inputs[key] = (inputs[key].astype(np.float32)*args.brightness).clip(0,255).astype(np.uint8)
                    if task_id == args.adapt_task and args.mirror:
                        for key in ["main_images","wrist_images"]: inputs[key] = np.ascontiguousarray(inputs[key][:,:,::-1,:])
                    torch.cuda.synchronize()
                    t0 = time.monotonic()
                    with torch.inference_mode():
                        actions, _ = model.predict_action_batch(inputs, mode="eval")
                    torch.cuda.synchronize()
                    latencies.append(time.monotonic() - t0)
                    assert torch.isfinite(actions).all()
                    actions = prepare_actions_for_libero(actions.clone(), "openpi")[0].cpu().numpy()
                    for action in actions:
                        obs, _, _, _ = env.step(action.tolist())
                        steps += 1
                        success = bool(env.check_success())
                        if success or steps >= budget:
                            break
                row = {"version": args.name, "task": task_id, "index": index, "seed": seed,
                       "success": success, "policy_steps": steps, "wall_seconds": time.monotonic()-start,
                       "request_seconds": latencies, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                       "smoke_only": args.smoke, "target_brightness":args.brightness,"target_mirror":args.mirror,"perturb":args.perturb,"adapt_task":args.adapt_task}
                with result_file.open("a") as output:
                    output.write(json.dumps(row) + "\n")
                print(json.dumps({k:v for k,v in row.items() if k != "request_seconds"}), flush=True)
        finally:
            env.close()
    (result_dir / f"{args.name}_eval_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    import traceback
    from common import save
    name=sys.argv[sys.argv.index('--name')+1]
    try:
        main();save(OUT/'results'/f'{name}_status.json',{'state':'complete'})
    except Exception:
        save(OUT/'results'/f'{name}_status.json',{'state':'failed','traceback':traceback.format_exc()});raise
