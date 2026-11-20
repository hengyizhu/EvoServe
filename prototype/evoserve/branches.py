"""Online branch placement, vision-versioned prefix reuse, and candidate snapshots."""

import os, subprocess, sys
from collections import defaultdict, deque
from pathlib import Path
import torch
from .backend import TrainModel
from .batch_backend import BatchedGraphModel
from .io import append, read


class Branches:
    def __init__(self, assets, ids, out, specs):
        self.out = out
        self.specs = specs
        self.ids = ids
        self.locked = {}
        self.failures = defaultdict(int)
        self.vision = {}
        self.waiting = deque()
        self.job = None
        self.rebuild_refs = {}
        self.retired_visual = set()
        self.pending = set()
        self.branch_of = {"parent": "r32"}
        with torch.cuda.device(0):
            self.graph = {
                "r32": BatchedGraphModel(assets, (1, 2, 4)),
                "r128": BatchedGraphModel(assets, (1, 2, 4), rank=128, visual=True),
            }
            for e in ids:
                self.graph["r32"].install(e, "parent")
        with torch.cuda.device(1):
            self.train = {
                "r32": TrainModel(assets, ids, 20261108),
                "r128": TrainModel(
                    assets, [*ids, *[e + 2 for e in ids]], 20261108, 128, True
                ),
            }
        self.bound = {}

    def options(self, e):
        options = [self.locked[e]] if e in self.locked else ["r32", "r128"]
        if (
            e in self.vision
            and e not in self.retired_visual
            and "visual" not in options
        ):
            options.append("visual")
        return options

    def prepare_vision(self, model, tag):
        if not model.visual or self.bound.get(model, "parent") == tag:
            return
        torch.cuda.synchronize()
        with torch.no_grad():
            for k, p in model.visual.items():
                if tag == "parent":
                    if k.endswith(".B"):
                        p.zero_()
                else:
                    p.copy_(self.vision[tag][k])
        torch.cuda.synchronize()
        self.bound[model] = tag

    def tag(self, e, version):
        return e if self.branch_of[version] == "visual" else "parent"

    def predict(self, msgs):
        groups = defaultdict(list)
        ans = [None] * len(msgs)
        prefix = [None] * len(msgs)
        for i, m in enumerate(msgs):
            b = self.branch_of[m["version"]]
            groups[
                ("r32" if b == "r32" else "r128", self.tag(m["expert"], m["version"]))
            ].append(i)
        with torch.cuda.device(0):
            for (rank, tag), indices in groups.items():
                model = self.graph[rank]
                self.prepare_vision(model, tag)
                a, kv, _ = model.infer_batch([msgs[i] for i in indices])
                for j, i in enumerate(indices):
                    ans[i] = a[j]
                    prefix[i] = model.map(
                        lambda x: x[j : j + 1].detach().cpu().clone(), kv
                    )
        return ans, prefix

    def cached(self, item, tag):
        from .accelerated import make_prefix

        if item.get("vision_tag", "parent") == tag and "prefix" in item:
            return item
        cache = item.setdefault("branch_prefix", {})
        if tag not in cache:
            model = self.train["r128"]
            self.prepare_vision(model, tag)
            cache[tag] = make_prefix(model, item["raw"])
        return dict(raw=item["raw"], actions=item["actions"], prefix=cache[tag])

    def step(self, batches):
        result = {}
        with torch.cuda.device(1):
            for name in ("r32", "r128", "visual"):
                group = {}
                for e, batch in batches.items():
                    if name in self.options(e):
                        key = e + 2 if name == "visual" else e
                        group[key] = [
                            self.cached(x, e if name == "visual" else "parent")
                            for x in batch
                        ]
                if group:
                    result[name] = self.train["r32" if name == "r32" else "r128"].step(
                        group
                    )
        return result

    def snapshot(self, e, step):
        versions = []
        options = self.options(e)
        if "visual" in options:
            options = ["visual", *[b for b in options if b != "visual"]]
        for b in options:
            v = f"e{e}-s{step}-{b}"
            p = self.out / (v + ".pt")
            rank = "r32" if b == "r32" else "r128"
            with torch.cuda.device(1):
                self.train[rank].snapshot(e + 2 if b == "visual" else e, p)
            with torch.cuda.device(0):
                self.graph[rank].install(e, v, p)
            self.branch_of[v] = b
            versions.append(v)
        return versions

    def outcome(self, e, event):
        if event["decision"] == "advance":
            if event["final_branch_winner"]:
                self.locked[e] = self.branch_of[event["winner"]]
                if e in self.vision and self.locked[e] != "visual":
                    self.retired_visual.add(e)
            return
        if event["decision"] == "promote":
            self.failures[e] = 0
            self.locked[e] = self.branch_of[event["candidate_version"]]
            if e in self.vision and self.locked[e] != "visual":
                self.retired_visual.add(e)
        else:
            self.failures[e] += 1
            if self.failures[e] >= 2 and e not in self.pending and e not in self.vision:
                self.pending.add(e)
                self.waiting.append(e)

    def service_visual(self, online, smoke=False):
        if self.job:
            e, proc, path = self.job
            if proc.poll() is None:
                if (path / "freeze.json").exists() and not (
                    path / "rebuild.pt"
                ).exists():
                    self.rebuild_refs[e] = list(online[e])
                    torch.save(
                        [(i, x["raw"]) for i, x in enumerate(self.rebuild_refs[e])],
                        path / "rebuild.tmp",
                    )
                    (path / "rebuild.tmp").replace(path / "rebuild.pt")
                return
            if proc.returncode:
                raise RuntimeError(f"visual branch failed: {path}/worker.log")
            with torch.cuda.device(1):
                state = torch.load(
                    path / "frozen.pt", map_location="cpu", weights_only=True
                )
                self.vision[e] = state["vision"]
                tr = self.train["r128"]
                ex = tr.experts[e + 2]
                with torch.no_grad():
                    for (name, p), moment in zip(
                        ex["masters"].items(), state["optimizer"]
                    ):
                        p.copy_(state["masters"][name])
                        ex["ema"][name].copy_(state["ema"][name])
                        ex["optim"].state[p] = {
                            k: (v.cuda() if k != "step" else v)
                            for k, v in moment.items()
                        }
                ex["lr_step"] = state["step"]
                ex["step"] = state["step"]
                for i, item in enumerate(self.rebuild_refs.pop(e)):
                    f = path / f"prefix_{i}.pt"
                    item.setdefault("branch_prefix", {})[e] = torch.load(
                        f, map_location="cpu", weights_only=True
                    )
                    f.unlink()
                for item in online[e]:
                    self.cached(item, e)
            append(
                self.out / "placement.jsonl",
                dict(
                    event="cache_rebuilt",
                    expert=e,
                    online_windows=len(online[e]),
                    switch=read(path / "done.json"),
                ),
            )
            self.job = None
        if self.waiting and len(online[self.waiting[0]]) >= 32:
            e = self.waiting.popleft()
            path = self.out / f"visual_e{e}"
            path.mkdir()
            ex = self.train["r128"].experts[e]
            spec = dict(
                online=[{k: x[k] for k in ("raw", "actions")} for x in online[e]],
                split=self.specs[e]["split"],
                seed=20261108 + e,
                step=ex["lr_step"],
                optimizer={
                    k: {n: v.cpu() for n, v in ex["optim"].state.get(p, {}).items()}
                    for k, p in ex["masters"].items()
                },
                masters={k: v.detach().cpu() for k, v in ex["masters"].items()},
                ema={k: v.cpu() for k, v in ex["ema"].items()},
            )
            torch.save(spec, path / "input.pt")
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=os.environ.get(
                    "CUDA_VISIBLE_DEVICES", "0,1,2"
                ).split(",")[2],
            )
            with (path / "worker.log").open("a") as f:
                proc = subprocess.Popen(
                    [sys.executable, "-u", "-m", "evoserve.visual_branch", str(path)],
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
            self.job = (e, proc, path)
            append(
                self.out / "placement.jsonl",
                dict(event="visual_started", expert=e, pid=proc.pid),
            )

    def prune(self, e, keep):
        for rank, graph in self.graph.items():
            for expert, v in list(graph.slots):
                if expert == e and v != "parent" and v not in keep:
                    graph.remove(e, v)
