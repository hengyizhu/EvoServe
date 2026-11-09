"""Fixed-budget tiered SFT with fp32 AdamW master weights and delta checkpoints."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from evaluate import ROOT, load_model
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
from torch.utils._pytree import tree_map


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def atomic_save(path, value):
    if shutil.disk_usage(ROOT).free < 20 * 2**30:
        raise RuntimeError("Disk free space below 20 GiB; refusing checkpoint write")
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def fingerprint(parameters):
    return {name: hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, p in parameters}


class Samples:
    def __init__(self, model):
        self.model = model
        self.split = json.loads((ROOT / "data_split.json").read_text())
        self.tables = {}

    def get(self, episode, start):
        if episode not in self.tables:
            path = Path(self.split["data_root"]) / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
            self.tables[episode] = pq.read_table(path)
        table = self.tables[episode]
        assert start + 10 <= len(table)
        row = table.slice(start, 1).to_pylist()[0]
        assert row["episode_index"] == episode
        task = self.split["target"] if row["task_index"] == 8 else self.split["history"]
        assert row["task_index"] in (5, 8)
        raw = {
            "observation/image": np.array(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")),
            "observation/wrist_image": np.array(Image.open(io.BytesIO(row["wrist_image"]["bytes"])).convert("RGB")),
            "observation/state": np.array(row["state"], dtype=np.float32),
            "actions": np.array(table["actions"].slice(start, 10).to_pylist(), dtype=np.float32),
            "prompt": task,
        }
        processed = self.model._input_transform_fn(raw)
        batch = tree_map(lambda x: torch.as_tensor(np.asarray(x).copy())[None], processed)
        actions = batch.pop("actions").cuda()
        obs = self.model._observation_dict_to_device(batch)
        assert actions.shape == (1, 10, 32)
        return obs, actions


def make_sequence(seed):
    split = json.loads((ROOT / "data_split.json").read_text())
    rng = random.Random(seed)
    current = [(e["episode_index"], i) for e in split["target_train"]
               for i in range(e["length"] - 9)]
    replay = [(x["episode"], x["start"]) for x in split["replay_windows"]]
    batches = []
    for _ in range(1000):
        batch = [rng.choice(current) for _ in range(4)] + [rng.choice(replay) for _ in range(4)]
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def configure(model, tier):
    for p in model.parameters():
        p.requires_grad_(True)
    model.freeze_vlm()
    if tier == "L1":
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("action_out_proj."))
    if tier == "L3":
        for p in model.img.parameters():
            p.requires_grad_(True)
        # Frozen language weights remain on the autograd path to the vision trunk.
        model.gradient_checkpointing_enable({"use_reentrant": False})
    model.train()
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["L1", "L2", "L3"], required=True)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    cfg = json.loads((ROOT / "experiment.json").read_text())
    cfg["seed"] = args.seed
    out = ROOT / ("preflight" if args.preflight else "training") / args.name
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "status.json", {"state": "loading", "pid": os.getpid()})
    model = load_model()
    pairs = configure(model, args.tier)
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    manifest = [{"name": n, "numel": p.numel(), "dtype": str(p.dtype),
                 "requires_grad": p.requires_grad, "optimizer_group": 0 if p.requires_grad else None}
                for n, p in model.named_parameters()]
    atomic_json(out / "parameters.json", manifest)
    atomic_json(out / "optimizer_config.json", {
        "optimizer": "AdamW", "master_dtype": "float32",
        "betas": [0.9, 0.95], "eps": 1e-8, "lr": cfg["lr"],
        "schedule": "linear warmup 50 steps then constant",
        "weight_decay": cfg["weight_decay"], "clip_grad": cfg["clip_grad"],
        "activation_checkpointing": args.tier == "L3",
        "loss": "mean over network horizon 10 and all 32 model action dimensions",
        "training_precision": "original selective bf16 weights; fp32 optimizer masters and moments"})
    source = json.loads((ROOT / "audit/parent_download.json").read_text())
    samples = Samples(model)
    batches = make_sequence(args.seed)
    atomic_json(out / "sample_sequence.json", batches)
    atomic_json(out / "training_config.json", cfg)
    masters = [torch.nn.Parameter(p.detach().float().clone()) for _, p in pairs]
    optimizer = torch.optim.AdamW(masters, lr=cfg["lr"], betas=(0.9, 0.95),
                                eps=1e-8, weight_decay=cfg["weight_decay"], foreach=False)
    frozen_before = fingerprint(frozen) if args.preflight else None
    train_before = fingerprint(pairs) if args.preflight else None

    def checkpoint(step):
        payload = {"parameters": {n: p.detach().cpu() for n, p in pairs},
                   "step": step, "tier": args.tier, "parent": source, "training_seed": args.seed}
        atomic_save(out / f"step_{step}.pt", payload)
        return payload

    def validate(step):
        model.eval()
        records = []
        with torch.no_grad():
            for episode in samples.split["target_validation"]:
                starts = np.linspace(0, episode["length"]-10, 10, dtype=int).tolist()
                for start in starts:
                    torch.manual_seed(cfg["seed"] + episode["episode_index"]*10000 + start)
                    observation, labels = samples.get(episode["episode_index"], start)
                    loss = model.compute_loss(observation, labels, train=False)
                    assert torch.isfinite(loss).all()
                    records.append({"episode": episode["episode_index"], "start": start,
                                    "loss_all32": loss.mean().item(),
                                    "loss_valid7": loss[:, :, :7].mean().item()})
        atomic_json(out / f"validation_{step}.json", {
            "step": step, "samples": records,
            "mean_loss_all32": float(np.mean([x["loss_all32"] for x in records])),
            "mean_loss_valid7": float(np.mean([x["loss_valid7"] for x in records]))})
        model.train()

    # Round-trip check at zero updates, with fixed observations and diffusion noise.
    obs, _ = samples.get(*batches[0][0])
    noise = torch.randn((1, 10, 32), device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
    model.eval()
    with torch.no_grad():
        before = model.sample_actions(obs, num_steps=5, noise=noise).cpu()
    checkpoint(0)
    delta = torch.load(out / "step_0.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(delta["parameters"], strict=False)
    with torch.no_grad():
        after = model.sample_actions(obs, num_steps=5, noise=noise).cpu()
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-6)
    atomic_json(out / "zero_roundtrip.json", {"max_abs": (before-after).abs().max().item(),
                                            "atol": 1e-6, "rtol": 1e-6})
    del delta
    model.train()
    if not args.preflight:
        validate(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    limit = 1 if args.preflight else cfg["steps"]
    for step in range(1, limit + 1):
        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        losses = []
        for micro, (episode, start) in enumerate(batches[step-1]):
            # Same sample, augmentation, noise and diffusion-time sequence for all tiers.
            seed = cfg["seed"] + (step-1)*8 + micro
            torch.manual_seed(seed)
            np.random.seed(seed)
            obs, actions = samples.get(episode, start)
            loss = model.compute_loss(obs, actions, train=True).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at {step}/{micro}")
            (loss/8).backward()
            losses.append(loss.detach().item())
        grads = {}
        for (name, p), master in zip(pairs, masters):
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError(f"Missing/nonfinite gradient: {name}")
            if args.preflight:
                grads[name] = {"norm": p.grad.float().norm().item()}
            master.grad = p.grad.detach().float()
        norm = torch.nn.utils.clip_grad_norm_(masters, cfg["clip_grad"], error_if_nonfinite=True)
        lr = cfg["lr"] * min(step/cfg["warmup_steps"], 1.0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        with torch.no_grad():
            for (_, p), master in zip(pairs, masters):
                p.copy_(master)
                if not torch.isfinite(p).all():
                    raise RuntimeError("Nonfinite parameter after update")
        torch.cuda.synchronize()
        row = {"step": step, "loss": float(np.mean(losses)), "lr": lr,
               "grad_norm": norm.item(), "wall_seconds": time.monotonic()-started,
               "step_seconds": time.monotonic()-t0,
               "samples_per_second": 8/(time.monotonic()-t0),
               "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)
        atomic_json(out / "status.json", {"state": "training", "pid": os.getpid(), **row})
        if args.preflight:
            assert fingerprint(frozen) == frozen_before, "Frozen parameters changed"
            after_hash = fingerprint(pairs)
            changed = [n for n, _ in pairs if after_hash[n] != train_before[n]]
            assert changed
            if args.tier == "L3":
                assert any(n.startswith("img.") and grads[n]["norm"] > 0 for n in grads)
                assert any(n.startswith("img.") for n in changed)
            atomic_json(out / "update_check.json", {"frozen_unchanged": True,
                       "changed_parameters": changed, "gradients": grads})
        if step in (200, 1000) or args.preflight:
            checkpoint(step)
            reloaded = torch.load(out / f"step_{step}.pt", map_location="cpu", weights_only=True)
            assert all(torch.equal(p.detach().cpu(), reloaded["parameters"][n]) for n,p in pairs)
            model.load_state_dict(reloaded["parameters"], strict=False)
            model.eval()
            with torch.no_grad():
                outputs = model.sample_actions(obs, num_steps=5, noise=noise)
                assert torch.isfinite(outputs).all()
            model.train()
            del reloaded
            if not args.preflight:
                validate(step)
                atomic_save(out / "resume.pt", {
                    "step": step, "tier": args.tier, "parent": source,
                    "masters": [m.detach().cpu() for m in masters],
                    "master_names": [n for n, _ in pairs],
                    "optimizer": optimizer.state_dict(),
                    "parameters": {n:p.detach().cpu() for n,p in pairs}})
            atomic_json(out / f"step_{step}.ready.json", {"step": step, "reload_verified": True})
    atomic_json(out / "status.json", {"state": "complete", "steps": limit,
                                    "wall_seconds": time.monotonic()-started})


if __name__ == "__main__":
    main()
