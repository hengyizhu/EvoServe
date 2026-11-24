"""Calibrate on this GPU, select numerically valid stages, then measure at capacity."""

import argparse
import csv
import fcntl
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from external_load import ExternalLoad

HERE = Path(__file__).resolve().parent


def execute(script, args, log, env, timeout=None):
    with log.open("w") as f:
        try:
            return (
                subprocess.run(
                    [sys.executable, str(HERE / script), *map(str, args)],
                    env=env,
                    check=False,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                ).returncode
                == 0
            )
        except subprocess.TimeoutExpired:
            print("Timed out", file=f)
            return False


def candidate(out, n, pre, den):
    path = out / f"{n}-{pre}-{den}.json"
    if not path.exists():
        ok = execute(
            "inference_engine.py",
            [n, pre, den, path],
            path.with_suffix(".log"),
            os.environ,
            1800,
        )
        if not ok:
            if path.exists():
                path.replace(path.with_suffix(".incomplete"))
            return None
    if path.exists():
        try:
            r = json.loads(path.read_text())
            if (r["batch"], r["prefix"], r["denoise"]) != (n, pre, den):
                return None
            if len(r["median_ms"]["candidate"]) != 2 or not all(
                math.isfinite(x) and x > 0 for x in r["median_ms"]["candidate"]
            ):
                return None
            if {"finite", "mean_abs", "max_abs"}.issubset(r):
                return r
        except (ValueError, TypeError, KeyError):
            return None
    return None


def calibrate(out):
    # Each behavior record is a paired, 50-episode test of this exact
    # prefix/denoising recipe on the pinned parent at batch one.
    evidence = json.loads(
        (HERE.parent / "reference/inference-behavior.json").read_text()
    )
    plan = {}
    audit = []
    for n in (1, 2, 4, 8):
        eligible = []
        for pre, den in (
            ("graph", "graph"),
            ("graph", "manual"),
            ("graph", "compile"),
        ):
            row = candidate(out, n, pre, den)
            if not row:
                audit.append(
                    {
                        "batch": n,
                        "prefix": pre,
                        "denoise": den,
                        "status": "trial failed",
                    }
                )
                continue
            behavior = evidence.get(f"{pre}-{den}", {})
            numeric_ok = row["finite"]
            allowed = numeric_ok and behavior.get("passed") is True
            audit.append(
                {
                    "batch": n,
                    "prefix": pre,
                    "denoise": den,
                    "max_abs": row["max_abs"],
                    "mean_abs": row["mean_abs"],
                    "behavior": behavior.get("passed"),
                    "eligible": allowed,
                    "median_ms": row["median_ms"],
                }
            )
            if allowed:
                eligible.append(row)
        (out / "selection.json").write_text(json.dumps(audit, indent=2))
        if not eligible:
            plan[n] = ["graph", "original"]
            continue
        best = min(eligible, key=lambda row: sum(row["median_ms"]["candidate"]))
        plan[n] = [best["prefix"], best["denoise"]]
    path = out / "plan.json"
    path.write_text(json.dumps(plan, indent=2))
    return path


def measure(out, plan, robots, mode, experts, seconds):
    tag = f"{robots}-{mode}-{experts}-{seconds}s"
    path = out / (tag + ".json")
    if not path.exists():
        with tempfile.TemporaryDirectory(prefix="evo-infer-opt-") as tmp:
            mps = (
                mode == "cache"
                and (Path(os.environ["SUITE_HOME"]) / "transport").read_text().strip()
                == "mps"
            )
            env = dict(
                os.environ,
                EVOSERVE_INFERENCE_PLAN=str(plan),
                CUDA_MPS_PIPE_DIRECTORY=tmp,
                CUDA_MPS_LOG_DIRECTORY=tmp,
                EVOSERVE_TRANSPORT="mps" if mps else "streams",
            )
            if mps:
                subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)
            try:
                ok = execute(
                    "supplement.py",
                    [
                        "--result",
                        path,
                        "--robots",
                        robots,
                        "--mode",
                        mode,
                        "--experts",
                        experts,
                        "--seconds",
                        seconds,
                    ],
                    path.with_suffix(".log"),
                    env,
                )
                if not ok:
                    raise RuntimeError(f"{tag} failed; see {path.with_suffix('.log')}")
            finally:
                if mps:
                    subprocess.run(
                        ["nvidia-cuda-mps-control"],
                        check=True,
                        input="quit\n",
                        text=True,
                        env=env,
                        timeout=30,
                    )
    result = json.loads(path.read_text())
    p = result["phases"][0]
    return {
        "robots": robots,
        "mode": mode,
        "experts": experts,
        "requests": p["requests"],
        "slo": p["slo"],
        "p99_ms": p["p99_ms"],
        "expert_steps_s": p["expert_steps_s"],
    }


class FallbackRequired(Exception):
    pass


def run(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "hardware.txt").write_text(
        subprocess.check_output(["nvidia-smi"], text=True)
    )
    # Verify the pinned reference server in an independent process. A failure
    # here affects the reference environment as well as the new candidates.
    health = [sys.executable, "-c", "from model import Server; Server([1]).serve(1)"]
    env = dict(
        os.environ, PYTHONPATH=str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    subprocess.run(health, env=env, check=True)
    plan = calibrate(out) if not a.plan else a.plan.resolve()
    if a.calibrate_only:
        return
    rows = []

    def failed(n, mode, experts, error):
        with (out / "failed.jsonl").open("a") as f:
            print(
                json.dumps(
                    {"robots": n, "mode": mode, "experts": experts, "error": str(error)}
                ),
                file=f,
            )
        return {"slo": 0}

    def record(n, mode, experts):
        try:
            row = measure(out, plan, n, mode, experts, a.seconds)
        except (RuntimeError, subprocess.SubprocessError) as exc:
            subprocess.run(health, env=env, check=True)
            if plan.name == "fallback-plan.json":
                return failed(n, mode, experts, exc)
            # Restart the campaign in its own directory to avoid mixing plans.
            raise FallbackRequired(str(exc), plan) from exc
        rows.append(row)
        with (out / "summary.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(rows)
        return row

    monitor = ExternalLoad(out)
    try:
        passed = []
        last = a.last
        if (
            a.a6000
            and a.last > a.first
            and all(d != "original" for _, d in json.loads(plan.read_text()).values())
        ):
            # Use the fastest GPU-only batch in the actual calibration as an
            # optimistic service-rate bound. Retain the point beyond the bound.
            best_rates = []
            for batch, (pre, den) in json.loads(plan.read_text()).items():
                r = json.loads((plan.parent / f"{batch}-{pre}-{den}.json").read_text())
                fastest_ms = min(sum(x) for x in r["samples_ms"]["candidate"])
                best_rates.append(int(batch) * 1000 / fastest_ms)
            last = min(last, math.ceil(max(best_rates) / (4 * 0.99)) + 1)
        for n in range(a.first, last + 1):
            if record(n, "infer", 1)["slo"] >= 0.99:
                passed.append(n)
        if a.a6000 and a.last > a.first:
            while last in passed:
                last += 1
                if record(last, "infer", 1)["slo"] >= 0.99:
                    passed.append(last)
        if not passed:
            any_infer = any(row["mode"] == "infer" for row in rows)
            (out / "capacity.json").write_text(
                json.dumps(
                    {
                        "n_star": 0 if any_infer else None,
                        "tested": [a.first, last],
                        "near": [],
                    }
                )
            )
            return
        near = sorted(passed, reverse=True)[: 1 if a.a6000 else 2]
        if not a.a6000 and len(near) < 2 and a.last > a.first:
            near.append(near[0] + 1 if near[0] < a.last else near[0] - 1)
        (out / "capacity.json").write_text(
            json.dumps(
                {"n_star": max(passed), "tested": [a.first, last], "near": near},
                indent=2,
            )
        )
        for n in near:
            for width in (1, 2, 4, 8):
                record(n, "cache", width)
            record(n, "recompute", 1)
    finally:
        monitor.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results") / time.strftime("inference-opt-%Y%m%d-%H%M%S"),
    )
    p.add_argument("--plan", type=Path)
    p.add_argument("--first", type=int, default=3)
    p.add_argument("--last", type=int, default=10)
    p.add_argument("--seconds", type=int, default=300)
    p.add_argument("--a6000", action="store_true")
    p.add_argument("--calibrate-only", action="store_true")
    args = p.parse_args()
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    lock = (
        Path(tempfile.gettempdir()) / f"evoserve-inference-{os.getuid()}-{device}.lock"
    )
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            run(args)
        except FallbackRequired as error:
            reason, previous = error.args
            root = args.out.resolve()
            fallback = {
                n: ["graph", "original"] for n in json.loads(previous.read_text())
            }
            args.plan = root / "fallback-plan.json"
            args.plan.write_text(json.dumps(fallback, indent=2))
            (root / "fallback-reason.txt").write_text(reason)
            args.out = root / "fallback"
            run(args)
