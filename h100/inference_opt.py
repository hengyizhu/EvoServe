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
        execute(
            "inference_engine.py",
            [n, pre, den, path],
            path.with_suffix(".log"),
            os.environ,
            1800,
        )
    if path.exists():
        r = json.loads(path.read_text())
        return r
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
            ("compile", "graph"),
            ("graph", "compile"),
        ):
            row = candidate(out, n, pre, den)
            if not row:
                audit.append(
                    dict(batch=n, prefix=pre, denoise=den, status="trial failed")
                )
                continue
            behavior = evidence.get(f"{pre}-{den}", {})
            numeric_ok = row["finite"] and (pre == "compile" or row["max_abs"] <= 0.001)
            allowed = numeric_ok and behavior.get("passed") is True
            audit.append(
                dict(
                    batch=n,
                    prefix=pre,
                    denoise=den,
                    max_abs=row["max_abs"],
                    mean_abs=row["mean_abs"],
                    behavior=behavior.get("passed"),
                    eligible=allowed,
                    median_ms=row["median_ms"],
                )
            )
            if allowed:
                eligible.append(row)
        (out / "selection.json").write_text(json.dumps(audit, indent=2))
        if not eligible:
            raise RuntimeError(
                f"No eligible implementation for batch {n}; see selection.json"
            )
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


def run(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "hardware.txt").write_text(
        subprocess.check_output(["nvidia-smi"], text=True)
    )
    plan = calibrate(out) if not a.plan else a.plan.resolve()
    if a.calibrate_only:
        return
    rows = []

    def record(n, mode, experts):
        row = measure(out, plan, n, mode, experts, a.seconds)
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
        if a.a6000 and a.last > a.first:
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
            raise RuntimeError("No tested robot count met the 99% capacity gate")
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
        run(args)
