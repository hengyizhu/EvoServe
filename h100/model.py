"""Shared frozen π0.5, fixed demo windows, and measured CUDA work."""

import os
from pathlib import Path
import numpy as np
import torch
from evoserve.backend import GraphModel, Model, TrainModel

ASSETS = Path(os.environ["SUITE_HOME"]) / "vla_followup_20261020"
CACHE = ASSETS / "benchmark/service-kv.pt"


def inputs():
    with np.load(ASSETS / "benchmark/inputs.npz") as f:
        return [
            dict(
                raw={
                    "observation/image": im,
                    "observation/wrist_image": wi,
                    "observation/state": st,
                    "prompt": str(prompt),
                },
                actions=ac,
            )
            for im, wi, st, prompt, ac in zip(
                f["images"], f["wrists"], f["states"], f["prompts"], f["actions"]
            )
        ][:16]


class Server(GraphModel):
    """Single production policy; maintain a CUDA Graph per supported batch shape."""

    def __init__(self, sizes):
        super().__init__(ASSETS)
        self.buckets = {
            1: (
                self.static,
                self.noise,
                self.events,
                self.graph,
                self.result,
                self.prefix,
            )
        }
        for n in sizes:
            if n == 1:
                continue
            self.static = self.cat_obs([self.buckets[1][0] for _ in range(n)])
            self.noise = torch.randn(n, 10, 32, device="cuda")
            self.events = [
                torch.cuda.Event(enable_timing=True, external=True) for _ in range(12)
            ]
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream), torch.inference_mode():
                self.work()
            self.stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with (
                torch.cuda.graph(graph, stream=self.stream, pool=self.graph.pool()),
                torch.inference_mode(),
            ):
                result = self.work()
            self.buckets[n] = (
                self.static,
                self.noise,
                self.events,
                graph,
                result,
                self.prefix,
            )
        self.records = inputs()
        self.index = 0

    def serve(self, count, save_cache=False):
        size = min(n for n in self.buckets if n >= count)
        obs, noise, events, graph, result, prefix = self.buckets[size]
        items = [
            self.records[(self.index + i) % len(self.records)]["raw"]
            for i in range(count)
        ]
        self.index += count
        prepared = self.cat_obs(
            [self.prepare(raw)[0] for raw in items + [items[-1]] * (size - count)]
        )
        with torch.cuda.stream(self.stream), torch.inference_mode():
            for dest, src in zip(self.leaves(vars(obs)), self.leaves(vars(prepared))):
                if torch.is_tensor(dest):
                    dest.copy_(src, non_blocking=True)
            noise.normal_()
            graph.replay()
            self.model.decode_actions(result, obs.state)[:count].cpu()
            cache = (
                self.map(lambda x: x.detach().cpu().clone(), prefix)
                if save_cache
                else None
            )
        self.stream.synchronize()
        return (
            events[0].elapsed_time(events[1]),
            events[1].elapsed_time(events[-1]),
            cache,
        )


def prefix(m, o):
    _, mask, kv = m.model.build_prefix_cache(o)
    return mask, kv


def loss(m, o, actions, mask, kv):
    generator = torch.Generator(device="cuda").manual_seed(173)
    noise = torch.randn(actions.shape, device="cuda", generator=generator)
    t = torch.full((len(actions),), 0.45, device="cuda")
    x = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
    v = m.model.velocity_from_suffix(m.model.run_suffix(o, x, t, kv, mask))
    return (v - (noise - actions)).square().mean()


class Work:
    def __init__(self, mode, rank, experts):
        self.mode = mode
        self.items = inputs()
        self.cursor = 0
        if mode in ("visual", "rebuild", "gradient"):
            self.model = Model(ASSETS, rank=rank, visual=(mode == "visual"))
            self.params = list(self.model.pairs.values()) + list(
                self.model.visual.values()
            )
            for p in self.params:
                p.requires_grad_(mode != "rebuild")
            if mode == "visual":
                self.master = [
                    torch.nn.Parameter(p.detach().float().clone()) for p in self.params
                ]
                self.opt = torch.optim.AdamW(self.master, lr=5e-5)
        else:
            self.model = TrainModel(ASSETS, range(experts), 20261108, rank=rank)
        if mode in ("cache", "gradient"):
            for item, kv in zip(
                self.items,
                torch.load(CACHE, map_location="cpu", weights_only=True),
                strict=True,
            ):
                item["prefix"] = self.model.map(lambda x: x.pin_memory(), kv)
        self.stream = torch.cuda.Stream(priority=0)

    def batch(self, items):
        m = self.model
        obs, actions = zip(*(m.prepare(x["raw"], x["actions"]) for x in items))
        return m.obs_map(lambda x: x.cuda(), m.cat_obs(obs)), torch.cat(actions).cuda()

    def step(self, n):
        m = self.model
        batch = [self.items[(self.cursor + i) % len(self.items)] for i in range(8)]
        self.cursor = (self.cursor + 8) % len(self.items)
        if self.mode in ("cache", "recompute"):
            m.step({e: batch for e in range(n)}, recompute=self.mode == "recompute")
        else:
            with torch.cuda.stream(self.stream):
                if self.mode == "rebuild":
                    with torch.no_grad():
                        o, _ = self.batch(batch[:4])
                        m.map(lambda x: x.detach().cpu(), prefix(m, o))
                else:
                    for sub in [batch[:4], batch[4:]]:
                        o, a = self.batch(sub)
                        (loss(m, o, a, *prefix(m, o)) / 2).backward()
                    for p, q in zip(self.params, self.master):
                        q.grad = p.grad.float()
                        p.grad = None
                    torch.nn.utils.clip_grad_norm_(self.master, 1.0)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        for p, q in zip(self.params, self.master):
                            p.copy_(q)
            self.stream.synchronize()


def gradient():
    server = Server([1])
    cache = [server.serve(1, True)[2] for _ in inputs()]
    torch.save(cache, CACHE)
    del server
    torch.cuda.empty_cache()
    w = Work("gradient", 32, 1)
    m = w.model
    items = w.items[:4]
    obs, actions = w.batch(items)

    def run(cached):
        if cached:
            parts = [m.map(lambda x: x.cuda(), i["prefix"]) for i in items]
        else:
            with torch.no_grad():
                parts = [prefix(m, w.batch([item])[0]) for item in items]
        kv = m.map(lambda *xs: torch.cat(xs), *parts)
        value = loss(m, obs, actions, *kv)
        value.backward()
        grads = [p.grad.detach().float().cpu() for p in w.params]
        for p in w.params:
            p.grad = None
        return grads

    a, b = run(True), run(False)
    diff = sum((x - y).double().square().sum().item() for x, y in zip(a, b))
    norm = sum(y.double().square().sum().item() for y in b)
    return dict(
        samples=4,
        parameters=sum(x.numel() for x in a),
        gradient_relative_l2=(diff / norm) ** 0.5,
        max_abs=max((x - y).abs().max().item() for x, y in zip(a, b)),
        bitwise_equal=all(torch.equal(x, y) for x, y in zip(a, b)),
    )
