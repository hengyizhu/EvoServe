"""Fixed-state LIBERO evaluation of the pinned parent or a parameter delta."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "RLinf"))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "libero_config"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ.get("CUDA_VISIBLE_DEVICES", "3").split(",")[0])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "4")

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
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol = json.loads((ROOT / "experiment.json").read_text())
    state_manifest = json.loads((ROOT / "audit/independent_states.json").read_text())
    suite = benchmark.get_benchmark_dict()["libero_10"](task_order_index=0)
    split = json.loads((ROOT / "data_split.json").read_text())
    assert suite.get_task(3).language == split["target"]
    assert suite.get_task(0).language == split["history"]
    model = load_model()
    if args.delta:
        delta = torch.load(args.delta, map_location="cpu", weights_only=True)
        parent = json.loads((ROOT / "audit/parent_download.json").read_text())
        assert delta["parent"] == parent, "Checkpoint parent identity mismatch"
        parameter_manifest = json.loads(
            (ROOT / "training" / delta["tier"] / "parameters.json").read_text())
        expected = {x["name"] for x in parameter_manifest if x["requires_grad"]}
        assert set(delta["parameters"]) == expected, "Incomplete trainable-parameter checkpoint"
        unexpected = model.load_state_dict(delta["parameters"], strict=False).unexpected_keys
        assert not unexpected, unexpected
    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    result_file = result_dir / f"{args.name}{'_smoke' if args.smoke else ''}.jsonl"
    if result_file.exists():
        raise FileExistsError(f"Refusing to overwrite {result_file}")
    manifest = []
    for task_id in protocol["eval"]["tasks"]:
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        selected = state_manifest[str(task_id)]["selected"]
        indices = [x["index"] for x in selected]
        if args.smoke:
            indices = indices[:1]
        for index in indices:
            assert index < len(states)
            digest = hashlib.sha256(np.asarray(states[index]).tobytes()).hexdigest()
            assert digest == next(x["state_sha256"] for x in selected if x["index"] == index)
            manifest.append({"task": task_id, "index": index, "language": task.language,
                             "state_sha256": hashlib.sha256(np.asarray(states[index]).tobytes()).hexdigest()})
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=256, camera_widths=256)
        try:
            for index in indices:
                seed = protocol["seed"] + task_id * 1000 + index
                np.random.seed(seed)
                torch.manual_seed(seed)
                env.seed(seed)
                env.reset()
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
                       "smoke_only": args.smoke}
                with result_file.open("a") as output:
                    output.write(json.dumps(row) + "\n")
                print(json.dumps({k:v for k,v in row.items() if k != "request_seconds"}), flush=True)
        finally:
            env.close()
    (result_dir / f"{args.name}_eval_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
