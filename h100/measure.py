import argparse
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from model import Server, Work, gradient
from load import Load


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def train(case, quick, profile):
    mode = case["mode"]
    n = case.get("experts", 1)
    w = Work(mode, case.get("rank", 32), n)
    w.step(n)
    if profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(1 if quick else 3):
                w.step(n)
        trace = Path(case["out"]) / "trace.tmp.json"
        prof.export_chrome_trace(str(trace))
        kernels = [
            e
            for e in json.loads(trace.read_text())["traceEvents"]
            if e.get("cat") == "kernel" and "dur" in e
        ]
        if not kernels:
            raise RuntimeError("CUPTI recorded no kernels")
        top = max(kernels, key=lambda e: e["dur"])
        trace.unlink()
        return dict(
            kernel=top["name"], longest_us=top["dur"], kernel_count=len(kernels)
        )
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(2 if quick else 12):
        start = time.perf_counter()
        w.step(n)
        times.append(time.perf_counter() - start)
    return dict(
        step_ms=statistics.median(times) * 1000,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        expert_steps_s=n / statistics.mean(times) if mode != "rebuild" else None,
        samples_s=4 / statistics.mean(times) if mode == "rebuild" else None,
        seconds=times,
    )


def measure(case, quick):
    kind = case["kind"]
    out = Path(case["out"])
    if kind == "correctness":
        result = gradient()
        write(out / "gradients.json", result)
        if not result["bitwise_equal"]:
            raise ValueError("matched-path gradients differ; inspect gradients.json")
        return result
    if kind == "static":
        n = case["batch"]
        server = Server([n])
        server.serve(n)
        times = []
        for _ in range(3 if quick else (50 if n == 1 else 20)):
            start = time.perf_counter()
            pre, den, _ = server.serve(n)
            times.append([pre, den, 1000 * (time.perf_counter() - start)])
        means = np.mean(times, axis=0)
        return dict(
            prefix_ms=means[0],
            denoise_ms=means[1],
            service_ms=means[2],
            samples_ms=times,
        )
    if kind in ("train", "profile"):
        return train(case, quick, kind == "profile")
    sec = case.get("seconds", 300)
    sizes = (1, 2, 4, 8, 16) if kind in ("capacity", "capacity_corun") else (1,)
    bench = Load(
        case.get("mode"),
        case.get("maximum", case.get("experts", 1)),
        case.get("rank", 32),
        sizes=sizes,
    )
    try:
        if kind == "capacity":
            tested = {}

            def test(n):
                if n not in tested:
                    result = bench.run(
                        [4 * n],
                        3 if quick else 30,
                        sync=n if case["alignment"] == "sync" else 1,
                    )
                    write(out / f"robots_{n}.json", result)
                    tested[n] = result["phases"][0]
                return tested[n]["slo"] >= 0.99

            bench.run([4], 1)
            for n in range(1, case["upper"] + 1):
                test(n)
            while test(max(tested)):
                test(max(tested) + 1)
            return dict(
                n_star=max([k for k, v in tested.items() if v["slo"] >= 0.99] or [0]),
                tested=tested,
            )
        service = case["service_ms"] / 1000
        rates = (
            [v / service for v in case["loads"]]
            if kind != "capacity_corun"
            else [4 * case["robots"]]
        )
        sync = case.get("robots", 1) if case.get("alignment") == "sync" else 1
        table = {float(k): v for k, v in case.get("table", {}).items()}
        opts = dict(
            n=case.get("experts", 1),
            sync=sync,
            table=table,
            service=service,
            feedback=case.get("feedback", False),
        )
        bench.run(rates[:1], 2, **opts)
        result = bench.run(rates, 5 if quick else sec, **opts)
        write(out / "windows.json", result)
        return dict(phases=result["phases"])
    finally:
        bench.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("case", type=Path)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    c = json.loads(a.case.read_text())
    out = Path(c["out"])
    if c["kind"] == "correctness":
        # First load is the smoke; perform a fresh independent check as the measurement.
        from model import inputs

        assert len(inputs()) >= 4
        gradient()
    elif c["kind"] == "static":
        pass  # warm-up replay in the measured graph shape
    result = measure(c, a.quick)
    write(out / "result.json", result)
