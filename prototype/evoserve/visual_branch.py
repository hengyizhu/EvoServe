"""Visual adaptation on the placement GPU, using the requesting expert's online replay."""

import time
import argparse
import math
import random
import sys
from pathlib import Path
import torch
from torch import nn
from .io import save, append


def attach_visual(model):
    class Adapter(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.A = nn.Parameter(base.weight.new_empty(16, base.in_features))
            self.B = nn.Parameter(base.weight.new_zeros(base.out_features, 16))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        def forward(self, x):
            return self.base(x) + nn.functional.linear(
                nn.functional.linear(x.to(self.A.dtype), self.A), self.B
            ).to(x.dtype)

    pairs = {}
    for name, base in list(model.img.named_modules()):
        if name.endswith(("mlp.fc1", "mlp.fc2")) and isinstance(base, nn.Linear):
            parent, leaf = name.rsplit(".", 1)
            adapter = Adapter(base)
            setattr(model.img.get_submodule(parent), leaf, adapter)
            for k in ("A", "B"):
                pairs[name + "." + k] = getattr(adapter, k)
    assert pairs
    return pairs


def run(job):
    from .backend import Model
    from .accelerated import Offline, read, BASE, make_prefix

    sys.path.insert(0, str(BASE / "learning/r128_twostage"))
    from switch_rule import decision

    spec = torch.load(job / "input.pt", weights_only=False, map_location="cpu")
    tr = Model(
        read(BASE / "configs/machine_a.example.json")["assets"], rank=128, visual=True
    )
    offline = Offline(spec["split"])
    online = spec["online"]
    rng = random.Random(spec["seed"])
    pairs = {**tr.pairs, **tr.visual}
    for k, p in pairs.items():
        p.requires_grad_(True)
        if k in tr.pairs:
            p.data.copy_(spec["masters"][k])
    masters = {k: nn.Parameter(v.detach().float().clone()) for k, v in pairs.items()}
    ema = {k: v.cuda().clone() for k, v in spec["ema"].items()}
    opt = torch.optim.AdamW(
        list(masters.values()),
        lr=2e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        foreach=True,
    )
    for k, values in spec.get("optimizer", {}).items():
        opt.state[masters[k]] = {
            n: (v.cuda() if n != "step" else v) for n, v in values.items()
        }
    torch.manual_seed(spec["seed"])
    probes = [s["raw"] for s in rng.sample(online, 32)]

    @torch.no_grad()
    def representations():
        seq = []
        for raw in probes:
            obs, _ = tr.prepare(raw)
            hidden, mask, _ = tr.model.build_prefix_cache(
                tr.obs_map(lambda x: x.cuda(), obs)
            )
            seq.append(hidden[mask].detach().cpu())
        return seq

    ref = representations()
    history = []
    n = 0
    while True:
        n += 1
        samples = rng.choices(online, k=4) + [
            offline.sample(rng, tr, raw_only=True) for _ in range(4)
        ]
        rng.shuffle(samples)
        for start in (0, 4):
            items = [
                tr.prepare(s["raw"], s["actions"]) for s in samples[start : start + 4]
            ]
            obs = tr.cat_obs([tr.obs_map(lambda x: x.cuda(), o) for o, _ in items])
            a = torch.cat([l for _, l in items]).cuda()
            noise = torch.randn_like(a)
            t = torch.tensor(
                [rng.betavariate(1.5, 1) * 0.999 + 0.001 for _ in items],
                device="cuda",
                dtype=torch.float32,
            )
            _, mask, kv = tr.model.build_prefix_cache(obs)
            x = t[:, None, None] * noise + (1 - t[:, None, None]) * a
            v = tr.model.velocity_from_suffix(tr.model.run_suffix(obs, x, t, kv, mask))
            ((v - (noise - a)).square().mean() / 2).backward()
        if n == 1:
            assert any(
                p.grad is not None and p.grad.norm().item() > 0
                for k, p in tr.visual.items()
                if k.endswith(".B")
            )
        for k, p in pairs.items():
            masters[k].grad = p.grad.float()
            p.grad = None
        torch.nn.utils.clip_grad_norm_(
            list(masters.values()), 1.0, error_if_nonfinite=True
        )
        j = spec["step"] + n
        opt.param_groups[0]["lr"] = 2e-4 * (
            j / 50
            if j <= 50
            else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (j - 50) / 3950)))
        )
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for k, p in pairs.items():
                p.copy_(masters[k])
                if k in ema:
                    ema[k].mul_(0.999).add_(masters[k], alpha=0.001)
        if spec.get("smoke") and n == 1:
            prefix = make_prefix(tr, probes[0])
            save(
                job / "smoke.json",
                dict(visual_gradient=True, rebuilt_mask_shape=list(prefix[0].shape)),
            )
            return
        if n % 50 == 0:
            cur = representations()
            d = sum(
                (1 - nn.functional.cosine_similarity(a.float(), b.float(), dim=-1))
                .mean()
                .item()
                for a, b in zip(cur, ref)
            ) / len(ref)
            row = dict(step=n, representation_cosine_distance_parent=max(0.0, d))
            history.append(row)
            append(job / "probes.jsonl", row)
            trigger, _ = decision(history, skip_initial=True)
            if trigger:
                # The cached action trainer uses the same ordered action master tensors.
                action_opt = [
                    {
                        k: (v.detach().cpu() if torch.is_tensor(v) else v)
                        for k, v in opt.state[masters[name]].items()
                    }
                    for name in tr.pairs
                ]
                torch.save(
                    dict(
                        vision={k: v.detach().cpu() for k, v in tr.visual.items()},
                        masters={k: masters[k].detach().cpu() for k in tr.pairs},
                        ema={k: v.cpu() for k, v in ema.items()},
                        optimizer=action_opt,
                        step=j,
                    ),
                    job / "frozen.pt",
                )
                save(job / "freeze.json", dict(trigger=trigger, step=j))
                while not (job / "rebuild.pt").exists():
                    time.sleep(0.1)
                for key, raw in torch.load(
                    job / "rebuild.pt", map_location="cpu", weights_only=False
                ):
                    torch.save(make_prefix(tr, raw), job / f"prefix_{key}.pt")
                save(job / "done.json", dict(trigger=trigger, step=j))
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    run(parser.parse_args().job)
