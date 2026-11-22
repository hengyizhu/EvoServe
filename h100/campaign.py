"""Isolate attempts, retain failures, derive calibrated dependent cases, then archive."""

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
import time
from external_load import ExternalLoad

ROOT = Path(__file__).resolve().parents[1]


def atomic(p, data):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(data)
    tmp.replace(p)


def command(args):
    p = subprocess.run(args, text=True, capture_output=True)
    return (p.stdout + p.stderr).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    out = (
        a.out
        or ROOT / "results" / datetime.datetime.now().strftime("h100-%Y%m%d-%H%M%S")
    ).resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        gpu=command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,power.limit,memory.total",
                "--format=csv",
            ]
        ),
        driver_cuda=command(["nvidia-smi"]),
        nvcc=command(["nvcc", "--version"]) if shutil.which("nvcc") else None,
        cpu=command(["lscpu"]),
        available_cores=sorted(os.sched_getaffinity(0)),
        quick=a.quick,
        torch=command(
            [
                sys.executable,
                "-c",
                "import torch;print(torch.__version__,torch.version.cuda)",
            ]
        ),
        pins=command([sys.executable, "-m", "pip", "freeze"]),
    )
    (out / "hardware.json").write_text(json.dumps(metadata, indent=2))
    monitor = ExternalLoad(out)
    external = {}
    mps_allowed = (Path(os.environ["SUITE_HOME"])/"transport").read_text().strip()=="mps"
    rows = {}
    results = {}

    def refresh():
        text = [
            "# Single GPU measurements",
            "",
            metadata["gpu"],
            "PyTorch and runtime CUDA: "+metadata["torch"]+"; inference CPU threads: 1",
            "",
            "| Case | Status | Measurements | External GPU load |",
            "|---|---|---|---|",
        ]
        for name, status in rows.items():
            r = results.get(name, {})
            visible = {
                k: v
                for k, v in r.items()
                if k not in ("samples_ms", "tested", "seconds")
            }
            text.append(
                f"| {name} | {status} | {json.dumps(visible, ensure_ascii=False)} | {external.get(name, chr(8212))} |"
            )
        atomic(out / "summary.md", "\n".join(text) + "\n")
        atomic(out / "index.json", json.dumps(rows, indent=2))

    def case(name, kind, why=None, **opts):
        if why:
            rows[name] = "skipped: " + why
            refresh()
            return {}
        nonlocal mps_allowed
        for i in (1, 2):
            d = out / name / f"attempt-{i}"
            attempt_start = time.time()
            d.mkdir(parents=True)
            config = dict(kind=kind, out=str(d), **opts)
            (d / "case.json").write_text(json.dumps(config))
            env = dict(os.environ)
            mps = mps_allowed and kind in ("corun", "capacity_corun") and opts.get("mode") == "cache"
            env["EVOSERVE_TRANSPORT"]="mps" if mps else "streams"
            try:
                with tempfile.TemporaryDirectory(prefix="evo-mps-") as pipe:
                    if mps:
                        env.update(
                            CUDA_MPS_PIPE_DIRECTORY=pipe,
                            CUDA_MPS_LOG_DIRECTORY=pipe,
                            CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="100",
                        )
                        subprocess.run(
                            ["nvidia-cuda-mps-control", "-d"],
                            env=env,
                            check=True,
                            timeout=15,
                        )
                    try:
                        with (d / "run.log").open("w") as log:
                            proc = subprocess.Popen(
                                [
                                    sys.executable,
                                    ROOT / "h100/measure.py",
                                    d / "case.json",
                                ]
                                + (["--quick"] if a.quick else []),
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                            try:
                                if proc.wait(
                                    timeout=(
                                        540
                                        if a.quick
                                        else (2700 if kind == "capacity" else 1400)
                                    )
                                ):
                                    raise RuntimeError("measurement exited nonzero")
                            except BaseException:
                                import signal

                                try:
                                    os.killpg(proc.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                                proc.wait()
                                raise
                    finally:
                        if mps:
                            subprocess.run(
                                ["nvidia-cuda-mps-control"],
                                input="quit\n",
                                text=True,
                                env=env,
                                timeout=20,
                            )
                results[name] = json.loads((d / "result.json").read_text())
                if kind in ("corun","capacity_corun"):
                    results[name]["transport"]=env["EVOSERVE_TRANSPORT"]
                    atomic(d/"result.json",json.dumps(results[name],indent=2))
                external[name] = monitor.summary(attempt_start)
                rows[name] = "passed"
                break
            except Exception:
                if mps and i == 1:
                    log_path = d / "run.log"
                    detail = traceback.format_exc() + (log_path.read_text(errors="replace")[-12000:] if log_path.exists() else "")
                    if any(marker in detail.lower() for marker in ("nvidia-cuda-mps-control", "mps did not confirm", "mps server", "mps client", "mps connection")):
                        mps_allowed = False
                        atomic(out / "transport_fallback.txt", detail)
                (d / "error.txt").write_text(traceback.format_exc())
                external[name] = monitor.summary(attempt_start)
                rows[name] = f"failed attempt {i}"
        refresh()
        return results.get(name, {})

    refresh()
    case("01-gradient", "correctness")
    for n in [1, 2, 4, 8, 16]:
        case(f"02-static-{n}", "static", batch=n)
    ms = results.get("02-static-1", {}).get("service_ms")
    # Fastest observed pure GPU time per sample gives an optimistic scan boundary.
    # Profile all integers through that bound so batching islands are included.
    static = [
        (n, results[f"02-static-{n}"])
        for n in (1, 2, 4, 8, 16)
        if f"02-static-{n}" in results
    ]
    import math

    upper = (
        math.ceil(
            1.25
            * max(
                (
                    1000 * n / min(pre + den for pre, den, _ in r["samples_ms"]) / 4
                    for n, r in static
                ),
                default=8,
            )
        )
        + 1
    )
    capacities = {
        k: case("02-capacity-" + k, "capacity", alignment=k, upper=upper).get("n_star")
        for k in ("staggered", "sync")
    }
    for rank in (32, 128):
        for mode in ("visual", "recompute", "cache"):
            for n in (1, 2, 4, 8, 16) if mode == "cache" else (1,):
                case(
                    f"03-{mode}-r{rank}-m{n}", "train", mode=mode, rank=rank, experts=n
                )
    safe = {}
    unknown = False

    def load(name, kind="corun", why=None, **kw):
        return case(
            name,
            kind,
            why=why or (None if ms else "missing B1 calibration"),
            service_ms=ms,
            **kw,
        )

    for l in (0.70, 0.80, 0.85):
        ok = []
        for n in (1, 2, 4, 8):
            r = load(
                f"04-fixed-{l}-m{n}", mode="cache", experts=n, loads=[l], feedback=True
            )
            if r:
                s = r["phases"][0]
                if s["slo"] >= 0.995 and s["p99_ms"] <= 220 and s["expert_steps_s"] > 0:
                    ok.append(n)
            else:
                unknown = True
        safe[l] = max(ok, default=0)
    (out / "feedforward.json").write_text(
        json.dumps(dict(service_ms=ms, table=safe, incomplete=unknown), indent=2)
    )
    load(
        "04-variable",
        why="missing characterization" if unknown else None,
        mode="cache",
        maximum=8,
        loads=[0.70, 0.85, 0.70],
        table=safe,
        feedback=True,
    )
    for l in (0.70, 0.85):
        load(f"04-recompute-{l}", mode="recompute", loads=[l])
    for alignment, n in capacities.items():
        for mode in ("cache", "recompute"):
            load(
                f"04-Nstar-{alignment}-{mode}",
                "capacity_corun",
                why=None if n else "no measured feasible N*",
                mode=mode,
                robots=n,
                alignment=alignment,
                maximum=8 if mode == "cache" else 1,
                table=safe if mode == "cache" else {},
                feedback=mode == "cache",
                seconds=180,
            )
    case("05-visual-solo", "train", mode="visual")
    load("05-visual-70", mode="visual", loads=[0.70], seconds=180)
    case("05-rebuild-solo", "train", mode="rebuild")
    for l in (0.50, 0.70):
        load(f"05-rebuild-{l}", mode="rebuild", loads=[l], seconds=180)
    for mode, n in [("cache", 1), ("cache", 8), ("recompute", 1), ("visual", 1)]:
        case(f"06-kernel-{mode}-{n}", "profile", mode=mode, experts=n)
    monitor.close()
    archive = out.with_name(out.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out, arcname=out.name)
    print(out / "summary.md", archive, flush=True)
    return int(any(v.startswith("failed") for v in rows.values()))


if __name__ == "__main__":
    sys.exit(main())
