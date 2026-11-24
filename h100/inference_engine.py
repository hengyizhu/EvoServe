"""Fixed-policy LIBERO serving. Stage choices are calibrated independently per device."""

import argparse
import json
import statistics
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from model import Server


class Replay:
    def __init__(self, run, static_inputs):
        self.replay = run
        # A CUDA graph records addresses but does not own external input tensors.
        self.static_inputs = static_inputs


def capture(fn, stream):
    fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        value = fn()

    def replay():
        graph.replay()
        return value

    return replay


def build(engine, n, prefix_kind, suffix_kind):
    if prefix_kind != "graph":
        raise ValueError("Only the graph prefix is supported")
    if suffix_kind == "compile":
        # A denoising call is replayed ten times with fixed input storage. Use
        # the single-graph implementation rather than inductor's graph-tree
        # allocator, which invalidates outputs across iterative invocations.
        import torch._inductor.config as inductor_config

        inductor_config.triton.cudagraph_trees = False
    obs, noise, _, _, _, _ = engine.buckets[n]
    m = engine.model
    s = engine.stream
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    with torch.cuda.stream(s), torch.inference_mode(), ExitStack() as stack:
        # The LIBERO input transform supplies a constant blank third camera.
        dummy = obs.images.get("right_wrist_0_rgb")
        if dummy is not None:
            assert torch.all(dummy == dummy.flatten()[0])
            img_forward = m.img.forward
            constant = img_forward(dummy)

            def encode(image):
                if image is dummy:
                    return constant
                return img_forward(image)

            stack.enter_context(patch.object(m.img, "forward", encode))

        def prefix():
            _, mask, kv = m.build_prefix_cache(obs)
            return mask, kv

        pre = capture(prefix, s)
        kv_template = pre()
        # Keep a stable KV destination for downstream graph replay and async export.
        cache = engine.map(torch.empty_like, kv_template)
        del kv_template

        def refresh():
            for dst, src in zip(
                engine.leaves(cache), engine.leaves(pre()), strict=True
            ):
                dst.copy_(src)

        refresh()
        times = [torch.full((n,), 1 - j / 10, device="cuda") for j in range(10)]
        static_inputs = list(times)
        if dummy is not None:
            static_inputs.append(constant)
        idx = [0]
        if suffix_kind == "graph":
            cond = [m.embed_suffix(obs, noise, t)[3] for t in times]
            for norm in (v for v in m.modules() if hasattr(v, "ada_modulation")):
                module = norm.ada_modulation
                values = tuple(module(c.to(module.weight.dtype)) for c in cond)
                static_inputs.extend(values)
                stack.enter_context(
                    patch.object(module, "forward", lambda x, v=values: v[idx[0]])
                )

        def step(x, t):
            mask, kv = cache
            return x - 0.1 * m.velocity_from_suffix(m.run_suffix(obs, x, t, kv, mask))

        one = (
            torch.compile(step, mode="reduce-overhead", dynamic=False, fullgraph=True)
            if suffix_kind == "compile"
            else step
        )

        if suffix_kind == "compile":
            x_buffer = torch.empty_like(noise)
            time_buffer = torch.empty_like(times[0])

        def ten():
            x = noise
            for j, t in enumerate(times):
                idx[0] = j
                if suffix_kind == "compile":
                    x_buffer.copy_(x)
                    time_buffer.copy_(t)
                    x = one(x_buffer, time_buffer)
                else:
                    x = one(x, t)
            return x

        if suffix_kind in ("manual", "manual-unfused"):
            from denoise_fast import Denoiser

            fast = Denoiser(m, obs, noise, cache, times)
            static_inputs.append(fast)
            denoise = capture(lambda: fast.ten(noise, fused=suffix_kind == "manual"), s)
        else:
            denoise = capture(ten, s) if suffix_kind == "graph" else ten
        out = torch.empty_like(noise)

        def run():
            events[0].record()
            refresh()
            events[1].record()
            out.copy_(denoise())
            events[2].record()

        run()
        s.synchronize()
    return obs, noise, events, Replay(run, static_inputs), out, cache


class SelectedServer(Server):
    def __init__(self, sizes, plan):
        super().__init__(sizes)
        choices = json.loads(Path(plan).read_text())
        for n in sizes:
            pre, den = choices[str(n)]
            if pre != "graph":
                raise ValueError("This suite requires a graph prefix")
            if den != "original":
                self.buckets[n] = build(self, n, pre, den)


def trial(n, pre, den, output):
    server = Server([n])
    original = server.buckets[n]
    new = build(server, n, pre, den)
    samples = {"baseline": [], "candidate": []}
    err_sum = count = maximum = 0
    finite = True
    with torch.inference_mode(), torch.cuda.stream(server.stream):
        obs, noise = original[:2]
        for i in range(12):
            items = [
                server.prepare(server.records[(i + j) % len(server.records)]["raw"])[0]
                for j in range(n)
            ]
            prepared = server.cat_obs(items)
            for a, b in zip(
                server.leaves(vars(obs)), server.leaves(vars(prepared)), strict=True
            ):
                if torch.is_tensor(a):
                    a.copy_(b)
            noise.normal_(
                generator=torch.Generator(device="cuda").manual_seed(20261123 + i)
            )
            for tag, bucket in (("baseline", original), ("candidate", new)):
                events, graph = bucket[2:4]
                graph.replay()
                events[-1].synchronize()
                if i >= 2:
                    samples[tag].append(
                        [
                            events[0].elapsed_time(events[1]),
                            events[1].elapsed_time(events[-1]),
                        ]
                    )
            delta = (original[4] - new[4]).abs()
            finite = finite and bool(torch.isfinite(delta).all())
            maximum = max(maximum, delta.max().item())
            err_sum += delta.double().sum().item()
            count += delta.numel()
    row = {
        "batch": n,
        "prefix": pre,
        "denoise": den,
        "max_abs": maximum,
        "mean_abs": err_sum / count,
        "finite": finite,
        "samples_ms": samples,
        "median_ms": {
            k: list(map(statistics.median, zip(*v))) for k, v in samples.items()
        },
    }
    Path(output).write_text(json.dumps(row, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("batch", type=int)
    p.add_argument("prefix", choices=["graph"])
    p.add_argument("denoise", choices=["graph", "compile", "manual", "manual-unfused"])
    p.add_argument("output")
    a = p.parse_args()
    trial(a.batch, a.prefix, a.denoise, a.output)
