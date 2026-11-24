"""Kineto trace for one actual (uncaptured) denoiser iteration."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from model import Server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--manual", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    server = Server([1])
    obs, noise, _, graph, _, cache = server.buckets[1]
    with torch.cuda.stream(server.stream), torch.inference_mode():
        graph.replay()
        mask, kv = cache
        t = torch.ones(1, device="cuda")

        def step():
            return noise - 0.1 * server.model.velocity_from_suffix(
                server.model.run_suffix(obs, noise, t, kv, mask)
            )

        if args.manual:
            from denoise_fast import Denoiser

            times = [torch.full((1,), 1 - j / 10, device="cuda") for j in range(10)]
            den = Denoiser(server.model, obs, noise, cache, times)
            prepared = den.prepare()

            def step():
                return den.ten(noise, steps=1, prepared=prepared)

        step()
        server.stream.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profile:
            step()
        profile.export_chrome_trace(str(args.out / "trace.json"))
    trace = json.loads((args.out / "trace.json").read_text())["traceEvents"]
    for name in ("kernel", "operator", "runtime"):
        totals = defaultdict(list)
        if name == "operator":
            for e in profile.key_averages():
                if e.key.startswith("aten::") and e.self_device_time_total:
                    totals[e.key] = [e.self_device_time_total / e.count] * e.count
        else:
            cat = "kernel" if name == "kernel" else "cuda_runtime"
            for e in trace:
                if e.get("cat") == cat and e.get("ph") == "X":
                    totals[e["name"]].append(e["dur"])
        with (args.out / f"{name}.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "calls", "total_us", "mean_us"])
            for k, v in sorted(totals.items(), key=lambda item: -sum(item[1]))[:20]:
                writer.writerow([k, len(v), sum(v), sum(v) / len(v)])


if __name__ == "__main__":
    main()
