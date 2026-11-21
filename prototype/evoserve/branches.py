"""Online branch placement, vision-versioned prefix reuse, and candidate snapshots."""

import os, subprocess, sys, time, math, traceback
from collections import defaultdict, deque
from pathlib import Path
import torch
from .backend import TrainModel
from .batch_backend import BatchedGraphModel
from .io import append, read


class Branches:
    def __init__(self, assets, ids, out, specs, online_only=False):
        self.online_only = online_only
        self.out = out
        self.specs = specs
        self.ids = ids
        self.locked = {}
        self.failures = defaultdict(int)
        self.vision = {}
        self.waiting = deque()
        self.next_admission = 0.0
        self.preparing = None
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
        self.updated_experts = set()
        with torch.cuda.device(1):
            for name in ("r32", "r128", "visual"):
                group = {}
                for e, batch in batches.items():
                    if name not in self.options(e):
                        continue
                    try:
                        items = [
                            self.cached(x, e if name == "visual" else "parent")
                            for x in batch
                        ]
                        if name == "visual":
                            row = self.train["r128"].step({e + 2: items})
                            result.setdefault("visual", {})[e] = row
                            self.updated_experts.add(e)
                        else:
                            group[e] = items
                    except Exception as exc:
                        if name == "visual":
                            self.retired_visual.add(e)
                            self.train["r128"].experts.pop(e + 2, None)
                            if self.locked.get(e) == "visual":
                                self.locked.pop(e)
                            # Keep the immutable serving snapshot if this is the production branch.
                            event = "visual_failed"
                        else:
                            event = "parent_prefix_rebuild_failed"
                        append(
                            self.out / "placement.jsonl",
                            dict(
                                event=event,
                                expert=e,
                                stage="cached_training",
                                error=str(exc),
                                time=time.time(),
                            ),
                        )
                if group:
                    result[name] = self.train[name].step(group)
                    self.updated_experts.update(group)
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
                if self.locked[e] != "visual" and any(
                    self.branch_of[v] == "visual" for v in event["contestants"]
                ):
                    self.retired_visual.add(e)
            return
        if event["decision"] == "promote":
            self.failures[e] = 0
            self.locked[e] = self.branch_of[event["candidate_version"]]
        else:
            self.failures[e] += 1
            if self.failures[e] >= 2 and e not in self.pending and e not in self.vision:
                self.pending.add(e)
                self.waiting.append(e)

    def service_visual(self, online, smoke=False):
        try:
            self._service_visual(online)
        except Exception as exc:
            expert = self.job[0] if self.job else self.preparing
            if expert is None:
                raise
            details = traceback.format_exc()
            if self.job:
                _, proc, path = self.job
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            else:
                path = self.out / f"visual_e{expert}"
            self.job = None
            self.vision.pop(expert, None)
            refs = self.rebuild_refs.pop(expert, [])
            for item in [*refs, *online[expert]]:
                item.get("branch_prefix", {}).pop(expert, None)
            self.retired_visual.add(expert)
            if self.train.get("r128"):
                self.train["r128"].experts.pop(expert + 2, None)
            # These files are temporary transfers scoped to the failed visual job.
            for artifact in [
                *path.glob("prefix_*.pt"),
                path / "rebuild.tmp",
                path / "rebuild.pt",
            ]:
                try:
                    artifact.unlink(missing_ok=True)
                except OSError:
                    pass
            append(
                self.out / "placement.jsonl",
                dict(
                    event="visual_failed",
                    expert=expert,
                    error=str(exc),
                    traceback=details,
                    log=str(path / "worker.log"),
                    time=time.time(),
                ),
            )
        finally:
            self.preparing = None

    def _service_visual(self, online):
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
            if time.monotonic() < self.next_admission:
                return
            self.next_admission = time.monotonic() + 30
            from .gpu_inventory import inventory

            e = self.waiting[0]
            target = int(os.environ["EVOSERVE_VISUAL_CARD"])
            try:
                gpu = inventory()[target]
                peak = read(os.environ["EVOSERVE_VISUAL_PEAK"], {}) or {}
                needed = (
                    math.ceil(peak["peak_mib"] * 1.10) if peak.get("peak_mib") else None
                )
                allowed_pid = read(self.out.parent / "operator.pid", -1)
                busy = [p for p in gpu["processes"] if p["pid"] != allowed_pid]
                reason = (
                    "waiting_for_completed_peak_profile"
                    if needed is None
                    else "other_compute_process"
                    if busy
                    else "insufficient_free_memory"
                    if gpu["free_mib"] < needed
                    else None
                )
                record = dict(
                    card=target,
                    free_mib=gpu["free_mib"],
                    measured_peak_mib=peak.get("peak_mib"),
                    required_mib=needed,
                    processes=gpu["processes"],
                )
            except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
                reason = "inventory_unavailable"
                record = dict(card=target, error=str(exc))
            append(
                self.out / "placement.jsonl",
                dict(
                    event="visual_deferred" if reason else "visual_admitted",
                    reason=reason,
                    time=time.time(),
                    expert=e,
                    **record,
                ),
            )
            if reason:
                return
            self.preparing = e
            e = self.waiting.popleft()
            path = self.out / f"visual_e{e}"
            path.mkdir()
            ex = self.train["r128"].experts[e]
            spec = dict(
                online=[
                    {k: x[k] for k in ("raw", "actions", "source")} for x in online[e]
                ],
                online_only=self.online_only,
                **({} if self.online_only else {"split": self.specs[e]["split"]}),
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
                CUDA_VISIBLE_DEVICES=str(target),
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
