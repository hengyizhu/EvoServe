"""Stateful LIBERO robots hosted by six simulator processes."""

from collections import deque
from pathlib import Path
import sys

robots = {}


def reset(robot, spec, episode, seed):
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "experiments/pi05_machine_b_20261115"))
    from scene_perturb import apply_scene

    if robot not in robots:
        suite = benchmark.get_benchmark_dict()["libero_10"](task_order_index=0)
        task = suite.get_task(spec["task_id"])
        env = OffScreenRenderEnv(
            bddl_file_name=str(
                Path(get_libero_path("bddl_files"))
                / task.problem_folder
                / task.bddl_file
            ),
            camera_heights=256,
            camera_widths=256,
            hard_reset=False,
        )
        robots[robot] = dict(
            env=env,
            states=suite.get_task_init_states(spec["task_id"]),
            prompt=task.language,
        )
    r = robots[robot]
    env = r["env"]
    env.seed(seed)
    np.random.seed(seed)
    env.reset()
    env.set_init_state(r["states"][episode % len(r["states"])])
    if spec["perturb"]["kind"] in ("camera_orbit", "table_texture"):
        apply_scene(env, spec["perturb"])
    for _ in range(10):
        obs, *_ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    r.update(
        spec=spec,
        positions=deque([obs["robot0_eef_pos"].copy()], maxlen=21),
        steps=0,
        taken=False,
        corrections=0,
    )
    return observation(r, obs)


def observation(r, obs):
    import numpy as np
    from perturbations import transform_image
    from rlinf.envs.sim.libero.utils import quat2axisangle

    raw = {
        "prompt": r["prompt"],
        "observation/state": np.concatenate(
            [
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"].copy()),
                obs["robot0_gripper_qpos"],
            ]
        ).astype(np.float32),
    }
    for dst, src in [
        ("observation/image", "agentview_image"),
        ("observation/wrist_image", "robot0_eye_in_hand_image"),
    ]:
        raw[dst] = transform_image(
            np.ascontiguousarray(obs[src][::-1, ::-1]), r["spec"]["perturb"]
        )
    return raw


def execute(robot, policy, operator, budget, shadow_only=False):
    import numpy as np

    r = robots[robot]
    env = r["env"]
    controller = env.env.robots[0].controller
    actions = []
    flags = []
    for p, o in zip(policy[:5], operator[:5]):
        # Compare the two OSC target displacement commands in world-space metres.
        deviation = np.linalg.norm(
            controller.scale_action(p[: controller.control_dim])[:3]
            - controller.scale_action(o[: controller.control_dim])[:3]
        )
        stuck = (
            len(r["positions"]) == 21
            and max(np.linalg.norm(x - r["positions"][0]) for x in r["positions"])
            < 0.002
        )
        r["taken"] |= bool(stuck or deviation > 0.02)
        a = o if r["taken"] and not shadow_only else p
        obs, *_ = env.step(a.tolist())
        r["positions"].append(obs["robot0_eef_pos"].copy())
        r["steps"] += 1
        r["corrections"] += int(r["taken"])
        actions.append(a)
        flags.append(r["taken"])
        success = bool(env.check_success())
        if success or r["steps"] >= budget:
            break
    return dict(
        raw=observation(r, obs),
        actions=np.asarray(actions),
        flags=flags,
        done=success or r["steps"] >= budget,
        success=success,
        autonomous=success and not r["taken"],
        steps=r["steps"],
        corrections=r["corrections"],
    )
