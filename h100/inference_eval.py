"""Paired low-light LIBERO-10 parent evaluation with reproducible policy noise."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from model import Server
from rlinf.envs.sim.libero.utils import quat2axisangle


def darken(image):
    return (image[::-1, ::-1].astype(np.float32) * 0.05).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plan", type=Path)
    p.add_argument(
        "--trace",
        type=int,
        nargs="+",
        help="Record first 50 control steps for the specified episodes",
    )
    args = p.parse_args()
    if args.plan:
        from inference_engine import SelectedServer

        engine = SelectedServer([1], args.plan)
    else:
        engine = Server([1])
    obs_buf, noise, _, graph, result, _ = engine.buckets[1]
    suite = benchmark.get_benchmark_dict()["libero_10"](task_order_index=0)
    task = suite.get_task(3)
    states = suite.get_task_init_states(3)
    env = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ),
        camera_heights=256,
        camera_widths=256,
    )
    completed = (
        {r["episode"] for r in map(json.loads, args.output.read_text().splitlines())}
        if args.output.exists()
        else set()
    )
    try:
        for ep in range(50):
            if ep in completed or (args.trace and ep not in args.trace):
                continue
            seed = 20261020 + 3000 + ep
            np.random.seed(seed)
            env.seed(seed)
            env.reset()
            obs = env.set_init_state(states[ep])
            for _ in range(10):
                obs, *_ = env.step([0.0] * 6 + [-1.0])
            gen = torch.Generator(device="cuda").manual_seed(seed)
            steps = 0
            success = False
            trace = []
            while steps < 520 and not success:
                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"].copy()),
                        obs["robot0_gripper_qpos"],
                    )
                )
                raw = {
                    "observation/image": darken(obs["agentview_image"]),
                    "observation/wrist_image": darken(obs["robot0_eye_in_hand_image"]),
                    "observation/state": state.astype(np.float32),
                    "prompt": task.language,
                }
                prepared, _ = engine.prepare(raw)
                with torch.inference_mode(), torch.cuda.stream(engine.stream):
                    for a, b in zip(
                        engine.leaves(vars(obs_buf)),
                        engine.leaves(vars(prepared)),
                        strict=True,
                    ):
                        if torch.is_tensor(a):
                            a.copy_(b, non_blocking=True)
                    noise.normal_(generator=gen)
                    graph.replay()
                    actions = (
                        engine.model.decode_actions(result, obs_buf.state)[0, :5]
                        .cpu()
                        .numpy()
                    )
                for action in actions:
                    if args.trace and steps < 50:
                        trace.append(
                            {
                                "step": steps,
                                "action": action.tolist(),
                                "state": env.sim.get_state().flatten().tolist(),
                            }
                        )
                    obs, *_ = env.step(action.tolist())
                    steps += 1
                    success = bool(env.check_success())
                    if success or steps == 520:
                        break
            row = {
                "episode": ep,
                "seed": seed,
                "state_sha256": hashlib.sha256(
                    np.asarray(states[ep]).tobytes()
                ).hexdigest(),
                "steps": steps,
                "success": success,
            }
            if args.trace:
                row["trace"] = trace
            with args.output.open("a") as f:
                print(json.dumps(row), file=f)
            print(row, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
