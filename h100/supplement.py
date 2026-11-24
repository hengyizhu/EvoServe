"""Fixed robot-count supplement. Each measured configuration gets its own process."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

from campaign import atomic
from external_load import ExternalLoad


def measure(a):
    from load import Load
    from model import CACHE, Server, inputs
    from sparse_export import SparseExport, selected_requests
    import torch

    if a.mode == "cache" and not CACHE.exists():
        cache_server = Server([1])
        torch.save([cache_server.serve(1, True)[2] for _ in inputs()], CACHE)
        del cache_server
        torch.cuda.empty_cache()
    import load as load_module

    sys.path.insert(0, str(Path(__file__).parent / "supplement"))
    from optimized_worker import worker
    from robot_controls import QueueSlack, instrument

    sizes = (1,) if a.quick else (1, 2, 4, 8, 16)
    plan = os.environ.get("EVOSERVE_INFERENCE_PLAN")
    if plan:
        from inference_engine import SelectedServer
        sizes = tuple(int(n) for n in json.loads(Path(plan).read_text()))
        load_module.Server = lambda sizes: SelectedServer(sizes, plan)
    train = a.mode in ("cache", "recompute")
    if train:
        # Width-specific winners in the published H100 exact-trajectory comparison.
        config = dict(path="graph", threads=4 if a.experts > 1 else 1)
        metadata = a.result.with_suffix("")
        metadata.mkdir(exist_ok=True)
        os.environ["EVOSERVE_FAST_TRAIN"] = json.dumps(config)
        os.environ["EVOSERVE_ROBOT_OUT"] = str(metadata)
        load_module.worker = worker
    load = Load(a.mode if train else None, a.experts, sizes=sizes)
    try:
        export = a.mode in ("export", "cache")
        if export:
            load.server.sparse_export = SparseExport(load.server)
            # Outside the timing window, verify the actual selected row and the
            # pinned destination against a synchronous copy of the same graph output.
            count = min(a.robots, max(sizes))
            _, _, reference = load.server.serve(
                count, save_cache=True, export_indices=[count - 1]
            )
            exporter = load.server.sparse_export
            exporter.stream.synchronize()
            for dest, src in zip(
                exporter.buffers[0], load.server.leaves(reference), strict=True
            ):
                if not dest.is_pinned() or not torch.equal(
                    dest, src[count - 1 : count]
                ):
                    raise RuntimeError(
                        "Asynchronous KV export failed the row-content check"
                    )
            del reference
        calibration = a.result.parent / f"{a.robots}-service-ms.json"
        if not calibration.exists():
            services = {}
            for size in sizes:
                load.server.serve(size)
                times = []
                for _ in range(5):
                    start = time.monotonic()
                    load.server.serve(size)
                    times.append(1000 * (time.monotonic() - start))
                services[size] = sorted(times)[2]
            atomic(calibration, json.dumps(services))
        else:
            services = {
                int(k): v for k, v in json.loads(calibration.read_text()).items()
            }
        instrument(load, QueueSlack(services) if train else None)
        serve = load.server.serve

        def run(seconds):
            selected = selected_requests(a.robots, seconds * 4) if export else set()
            cursor = 0

            def dispatch(count):
                nonlocal cursor
                indices = [j for j in range(count) if cursor + j in selected]
                cursor += count
                return serve(count, export_indices=indices)

            load.server.serve = dispatch
            if train:
                load.slack = QueueSlack(services)
            if torch.get_num_threads() != 1:
                raise RuntimeError("Inference must run with one CPU thread")
            start = load.server.sparse_export.cursor if export else 0
            result = load.run([a.robots * 4], seconds, n=a.experts, feedback=train)
            load.check()
            if export:
                load.server.sparse_export.stream.synchronize()
                assert load.server.sparse_export.cursor - start == len(selected)
            assert cursor == a.robots * 4 * seconds
            return dict(
                **result,
                export_requests=sorted(selected),
                kv_bytes_per_export=sum(
                    t.numel() * t.element_size() for t in exporter.buffers[0]
                )
                if export
                else 0,
                service_ms=services,
                slack_threshold_ms=200,
                protected_checks=load.slack.protected_checks if train else 0,
                training=json.loads((metadata / "training-worker.json").read_text())
                if train
                else None,
                transport=os.environ.get("EVOSERVE_TRANSPORT")
                if train
                else "inference",
                inference_threads=torch.get_num_threads(),
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
            )

        run(2)
        result = run(a.seconds)
        assert result["phases"][0]["requests"] == a.seconds * a.robots * 4
        if train and not result["phases"][0]["expert_steps_s"] and a.quick:
            raise ValueError("Rehearsal completed zero training updates")
        atomic(a.result, json.dumps(result, indent=2))
    finally:
        load.close()


def reports(out, rows):
    fields = [
        "robots",
        "mode",
        "experts",
        "status",
        "requests",
        "slo",
        "p99_ms",
        "expert_steps_s",
        "mean_duty",
        "exports",
        "transport",
    ]
    with (out / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    table = {}
    for n in dict.fromkeys(r["robots"] for r in rows):
        ok = [
            r
            for r in rows
            if r["robots"] == n
            and r["status"] == "ok"
            and r["mode"] in ("cache", "recompute")
            and r["expert_steps_s"] > 0
        ]
        safe = [
            r["experts"]
            for r in ok
            if r["mode"] == "cache" and r["slo"] >= 0.995 and r["p99_ms"] <= 220
        ]
        table[n] = {"safe_experts": max(safe, default=0)}
        for mode in ("cache", "recompute"):
            valid = [r for r in ok if r["mode"] == mode and r["slo"] >= 0.99]
            best = max(valid, key=lambda r: r["expert_steps_s"], default=None)
            table[n][mode] = (
                None
                if best is None
                else {
                    k: best[k] for k in ("experts", "expert_steps_s", "slo", "p99_ms")
                }
            )
    atomic(out / "feedforward.json", json.dumps(table, indent=2))
    md = [
        "| Robots | Mode | Experts | Status | P99 ms | SLO | Expert steps/s |",
        "|---:|---|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            "| "
            + " | ".join(
                str(r.get(k, "—"))
                for k in (
                    "robots",
                    "mode",
                    "experts",
                    "status",
                    "p99_ms",
                    "slo",
                    "expert_steps_s",
                )
            )
            + " |"
        )
    atomic(out / "summary.md", "\n".join(md) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--result", type=Path)
    p.add_argument("--robots", type=int)
    p.add_argument("--mode")
    p.add_argument("--experts", type=int, default=1)
    p.add_argument("--seconds", type=int, default=300)
    a = p.parse_args()
    if a.result:
        measure(a)
        return
    out = (
        a.out or Path("results") / time.strftime("supplement-%Y%m%d-%H%M%S")
    ).resolve()
    out.mkdir(parents=True, exist_ok=False)
    print(out, flush=True)
    (out / "hardware.txt").write_text(
        subprocess.check_output(["nvidia-smi"], text=True)
        + subprocess.check_output(["lscpu"], text=True)
    )
    monitor = ExternalLoad(out)
    rows = []
    try:
        for robots in [1] if a.quick else [5, 6, 4]:
            for mode, n in [
                ("infer", 1),
                ("export", 1),
                ("cache", 1),
                ("cache", 2),
                ("cache", 4),
                ("cache", 8),
                ("recompute", 1),
            ]:
                tag = f"{robots}-{mode}-{n}"
                row = dict(robots=robots, mode=mode, experts=n, status="failed")
                mps = (
                    Path(os.environ["SUITE_HOME"]) / "transport"
                ).read_text().strip() == "mps" and mode == "cache"
                for attempt in (1, 2):
                    path = out / f"{tag}-{attempt}.json"
                    log = path.with_suffix(".log")
                    print(tag, attempt, flush=True)
                    with (
                        tempfile.TemporaryDirectory(prefix="evo-supp-") as pipe,
                        log.open("w") as f,
                    ):
                        env = dict(
                            os.environ,
                            EVOSERVE_TRANSPORT="mps" if mps else "streams",
                            CUDA_MPS_PIPE_DIRECTORY=pipe,
                            CUDA_MPS_LOG_DIRECTORY=pipe,
                        )
                        daemon = False
                        try:
                            if mps:
                                subprocess.run(
                                    ["nvidia-cuda-mps-control", "-d"],
                                    env=env,
                                    check=True,
                                    stdout=f,
                                    stderr=f,
                                )
                                daemon = True
                            cmd = [
                                sys.executable,
                                __file__,
                                "--result",
                                str(path),
                                "--robots",
                                str(robots),
                                "--mode",
                                mode,
                                "--experts",
                                str(n),
                                "--seconds",
                                str(60 if a.quick else 300),
                            ]
                            if a.quick:
                                cmd.append("--quick")
                            subprocess.run(cmd, env=env, stdout=f, stderr=f, check=True)
                            result = json.loads(path.read_text())
                            phase = result["phases"][0]
                            row.update(
                                status="ok",
                                exports=len(result["export_requests"]),
                                transport=result["transport"],
                                **{
                                    k: phase[k]
                                    for k in (
                                        "requests",
                                        "slo",
                                        "p99_ms",
                                        "expert_steps_s",
                                        "mean_duty",
                                    )
                                },
                            )
                            break
                        except (subprocess.CalledProcessError, OSError):
                            if not daemon or "MPS did not confirm" in log.read_text():
                                mps = False
                        finally:
                            if daemon:
                                subprocess.run(
                                    ["nvidia-cuda-mps-control"],
                                    env=env,
                                    input="quit\n",
                                    text=True,
                                    stdout=f,
                                    stderr=f,
                                    timeout=20,
                                )
                rows.append(row)
                reports(out, rows)
    finally:
        monitor.close()
        with tarfile.open(str(out) + ".tar.gz", "w:gz") as tar:
            tar.add(out, arcname=out.name)
    if any(r["status"] != "ok" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
