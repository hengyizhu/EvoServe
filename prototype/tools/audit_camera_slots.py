"""Independent paired, policy-controlled LIBERO camera rollouts, with shadow HG triggers."""

import argparse
import hashlib
import json
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from evoserve import accelerated_sim as sim
from evoserve.batch_backend import BatchedGraphModel
from evoserve.io import append, read, save
from evoserve.operators import load_operator, sha256

PROTO = Path(__file__).resolve().parents[1]


def digest(weights):
    h = hashlib.sha256()
    for key, value in sorted(weights.items()):
        x = value.detach().cpu().contiguous()
        h.update(str((key, str(x.dtype), tuple(x.shape))).encode())
        h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    spec = next(
        s
        for s in read(PROTO / "configs/accelerated.machine_a.json")
        if s["name"] == "camera"
    )
    assets = Path(read(PROTO / "configs/machine_a.example.json")["assets"])
    model = BatchedGraphModel(assets, (1, 2, 4))
    norm = assets / "parent/physical-intelligence/libero/norm_stats.json"
    checks = {}
    for role, path in [("candidate", args.candidate), ("production", args.production)]:
        if path is None:
            continue
        ck = torch.load(path, map_location="cpu", weights_only=True)
        model.install(0, role, path)
        weights = ck["parameters"]
        expected = {n: v.to(model.pairs[n].dtype) for n, v in weights.items()}
        checks[role] = dict(
            file_sha256=sha256(path),
            step=ck["step"],
            kind=ck["weights_kind"],
            file_parameters=digest(weights),
            expected_slot=digest(expected),
            actual_slot=digest(model.slots[0, role]),
            norm_sha256=sha256(norm),
            norm_path=str(norm),
            tensors=len(weights),
        )
    save(out / "weights.json", checks)
    assert all(
        c["kind"] == "EMA" and c["actual_slot"] == c["expected_slot"]
        for c in checks.values()
    )
    if args.candidate:
        assert checks["production"]["actual_slot"] != checks["candidate"]["actual_slot"]
    with torch.cuda.device(1):
        operator = BatchedGraphModel(assets, (1, 2, 4))
        weights, _ = load_operator(spec["checkpoint"], operator.pairs, spec["task_id"])
        operator.slots[0, "operator"] = {
            n: v.to(device="cuda", dtype=operator.pairs[n].dtype)
            for n, v in weights.items()
        }

    def actions(model, device, messages):
        with torch.cuda.device(device):
            return model.infer_batch(messages)[0]

    seed = 20261108 + (
        checks["candidate"]["step"] if args.candidate else args.state_schedule_step
    )  # Original run used step-seeded validation.
    states = list(range(50))
    random.Random(seed).shuffle(states)
    workers = [
        ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
        for _ in range(4)
    ]
    totals = {role: [] for role in checks}
    try:
        with ThreadPoolExecutor(1) as thread:
            stride = 2 if args.candidate else 4
            for j in range(0, 2 if args.preflight else 50, stride):
                robots = list(range(4 if args.candidate else min(4, 50 - j)))
                role = {
                    r: ("candidate" if args.candidate and r % 2 == 0 else "production")
                    for r in robots
                }
                pair = {r: j + (r // 2 if args.candidate else r) for r in robots}
                jobs = {
                    r: workers[r].submit(
                        sim.reset, r, spec, states[pair[r]], seed + pair[r]
                    )
                    for r in robots
                }
                obs = {r: f.result() for r, f in jobs.items()}
                for r in (0, 2) if args.candidate else ():
                    assert all(
                        np.array_equal(v, obs[r + 1][k]) for k, v in obs[r].items()
                    ), ("initial observation differs", pair[r])
                block = 0
                while obs:
                    active = list(obs)
                    msgs = [
                        dict(
                            expert=0,
                            raw=obs[r],
                            version=role[r],
                            seed=(seed + pair[r]) * 1024 + block,
                        )
                        for r in active
                    ]
                    teacher = thread.submit(
                        actions,
                        operator,
                        1,
                        [dict(m, version="operator") for m in msgs],
                    )
                    policy = actions(model, 0, msgs)
                    if block == 0 and j == 0:
                        # Check the per-row parameters actually bound to the captured graph.
                        packed = model.buckets[4]["params"]
                        for i, r in enumerate(active):
                            assert (
                                digest({n[5:]: v[i] for n, v in packed.items()})
                                == checks[role[r]]["actual_slot"]
                            )
                        save(
                            out / "batch_binding.json",
                            dict(
                                batched_rows_matched=4, roles=[role[r] for r in active]
                            ),
                        )
                    op = teacher.result()
                    jobs = {
                        r: workers[r].submit(
                            sim.execute, r, policy[i], op[i], 520, True
                        )
                        for i, r in enumerate(active)
                    }
                    for r, f in jobs.items():
                        row = f.result()
                        if row["done"]:
                            rec = dict(
                                pair=pair[r],
                                state=states[pair[r]],
                                seed=seed + pair[r],
                                role=role[r],
                                policy_success=row["success"],
                                autonomous_under_HG=row["autonomous"],
                                steps=row["steps"],
                            )
                            append(out / "episodes.jsonl", rec)
                            totals[role[r]].append(rec)
                            del obs[r]
                        else:
                            obs[r] = row["raw"]
                    block += 1
        save(
            out / "result.json",
            {
                role: dict(
                    episodes=len(rows),
                    policy_success=sum(r["policy_success"] for r in rows),
                    autonomous_under_HG=sum(r["autonomous_under_HG"] for r in rows),
                    first10_autonomous=sum(
                        r["autonomous_under_HG"] for r in rows if r["pair"] < 10
                    ),
                )
                for role, rows in totals.items()
            },
        )
    finally:
        for w in workers:
            w.shutdown(cancel_futures=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path)
    p.add_argument("--state-schedule-step", type=int)
    for name in ("production", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--preflight", action="store_true")
    run(p.parse_args())
