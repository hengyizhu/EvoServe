"""Full-speed HG-DAgger loop for eight robots, four independent action experts."""

import argparse, io, json, multiprocessing, os, random, sys, time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from . import accelerated_sim as sim
from .backend import TrainModel
from .batch_backend import BatchedGraphModel
from .core import ExpertState, LearningPrior, Scheduler
from .io import append, save
from .operators import load_operator

BASE = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(Path(p).read_text())


class Offline:
    def __init__(self, split):
        self.split = read(split)
        self.windows = [
            (e["episode_index"], i)
            for e in self.split["target_train"]
            for i in range(e["length"] - 9)
        ]
        self.tables = {}

    def sample(self, rng, tr):
        import numpy as np, pyarrow.parquet as pq
        from PIL import Image

        ep, i = rng.choice(self.windows)
        if ep not in self.tables:
            self.tables[ep] = pq.read_table(
                Path(self.split["data_root"])
                / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet",
                columns=["image", "wrist_image", "state", "actions"],
            )
        table = self.tables[ep]
        row = table.slice(i, 1).to_pylist()[0]
        raw = {
            "prompt": self.split["target"],
            "observation/state": np.asarray(row["state"], dtype=np.float32),
        }
        for dst, src in [
            ("observation/image", "image"),
            ("observation/wrist_image", "wrist_image"),
        ]:
            raw[dst] = np.asarray(
                Image.open(io.BytesIO(row[src]["bytes"])).convert("RGB")
            )
        obs, _ = tr.prepare(raw)
        with tr.torch.no_grad():
            _, mask, kv = tr.model.build_prefix_cache(
                tr.obs_map(lambda x: x.cuda(), obs)
            )
            prefix = tr.map(lambda x: x.detach().cpu(), (mask, kv))
        return dict(
            raw=raw,
            actions=np.asarray(
                table["actions"].slice(i, 10).to_pylist(), dtype=np.float32
            ),
            prefix=prefix,
        )


def run(args):
    import numpy as np, torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    specs = read(args.spec)
    assets = read(BASE / "configs/machine_a.example.json")["assets"]
    if args.paired:
        specs = [s for s in specs if s["name"] in ("camera", "Y")]
        assert len(specs) == 2
    from .paired import PairedScheduler

    ids = list(range(len(specs)))
    per_expert = 8 // len(specs)
    robot_ids = list(range(8))
    rng = random.Random(20261108)
    prior = read(BASE / "configs/learning_prior.json")
    scheduler_class = PairedScheduler if args.paired else Scheduler
    sched = scheduler_class(
        [
            ExpertState(e, list(range(e * per_expert, (e + 1) * per_expert)), "parent")
            for e in ids
        ],
        LearningPrior(prior["points"], prior["default_amplitude"]),
    )
    for e in ids:
        sched.change_environment(e)
    ctx = multiprocessing.get_context("spawn")
    workers = [ProcessPoolExecutor(1, mp_context=ctx) for _ in range(6)]

    def submit(r, fn, *a):
        return workers[r % 6].submit(fn, r, *a)

    offline = [Offline(s["split"]) for s in specs]
    online = [deque(maxlen=128) for e in ids]
    started = Counter()
    period_frames = Counter()
    period_takeovers = Counter()
    period_prod = {e: [] for e in ids}
    credit = Counter()
    frames = Counter()
    interventions = Counter()
    promoted = Counter()
    validated = Counter()
    episodes = Counter()
    versions = {}
    raw = {}
    pending = {}
    unassisted = {}
    paths = {}

    noise_base = {}
    block = {}
    validation_sizes = {e: [] for e in ids}

    def begin(r):
        e = r // per_expert
        ep = episodes[r]
        key = f"{r}-{ep}"
        v = sched.begin_episode(e, r, key)
        if v is None:
            return None
        versions[r] = v
        pending[r] = None
        unassisted[r] = deque(maxlen=8)
        assignment = sched.experts[e].active_episodes[key]
        state = assignment.get(
            "state", ep * per_expert + r % per_expert if args.paired else ep
        )
        seed = assignment.get("seed", 20261108 + state * 4 + e)
        noise_base[r] = seed if "pair" in assignment else None
        block[r] = 0
        return submit(r, sim.reset, specs[e], state, seed)

    init = {r: begin(r) for r in robot_ids}
    with torch.cuda.device(0):
        serving = BatchedGraphModel(assets, (1, 2, 4))
        trainer = TrainModel(assets, ids, 20261108)
        for e in ids:
            serving.install(e, "parent")
    with torch.cuda.device(1):
        operator = BatchedGraphModel(assets, (1, 2, 4))
        for e, s in enumerate(specs):
            weights, _ = load_operator(s["checkpoint"], operator.pairs, s["task_id"])
            operator.slots[e, "operator"] = {
                n: x.to(device="cuda", dtype=operator.pairs[n].dtype)
                for n, x in weights.items()
            }
    raw.update({r: f.result() for r, f in init.items()})

    def predict(model, gpu, messages, cache):
        with torch.cuda.device(gpu):
            actions, prefix, _ = model.infer_batch(messages)
            samples = (
                [
                    model.map(lambda x: x[i : i + 1].detach().cpu().clone(), prefix)
                    for i in range(len(messages))
                ]
                if cache
                else None
            )
            return actions, samples

    pool = ThreadPoolExecutor(1)
    start = time.monotonic()
    deadline = start + args.seconds if args.seconds is not None else float("inf")
    next_log = start + 1800
    tick = 0

    def summary(stage):
        for e in ids:
            st = sched.experts[e]
            recent = st.production_recent
            append(
                out / "metrics.jsonl",
                dict(
                    stage=stage,
                    elapsed_s=time.monotonic() - start,
                    expert=specs[e]["name"],
                    production=st.production,
                    production_success=sum(recent) / len(recent) if recent else None,
                    production_episodes=len(recent),
                    frames=frames[e],
                    takeover_frames=interventions[e],
                    takeover_fraction=interventions[e] / max(1, frames[e]),
                    window_frames=period_frames[e],
                    window_takeover_fraction=period_takeovers[e] / period_frames[e]
                    if period_frames[e]
                    else None,
                    window_production_episodes=len(period_prod[e]),
                    window_production_success=sum(period_prod[e]) / len(period_prod[e])
                    if period_prod[e]
                    else None,
                    steps=st.step,
                    pairs_per_finished_validation=validation_sizes[e],
                    episodes_per_finished_validation=[
                        2 * n for n in validation_sizes[e]
                    ],
                    validations_started=started[e],
                    validations=validated[e],
                    promotions=promoted[e],
                ),
            )
        period_frames.clear()
        period_takeovers.clear()
        for outcomes in period_prod.values():
            outcomes.clear()

    save(
        out / "ready.json",
        dict(
            seed=20261108,
            paired=args.paired,
            robots_per_expert=per_expert,
            robots=8,
            simulator_processes=6,
            expert_names=[s["name"] for s in specs],
            perturbation_start_s=0,
            wall_start=time.time(),
        ),
    )
    try:
        while time.monotonic() < deadline and (
            not args.until_epoch or time.time() < args.until_epoch
        ):
            tick += 1
            reset_jobs = {}
            for r in robot_ids:
                if r not in raw:
                    f = begin(r)
                    if f:
                        reset_jobs[r] = f
            raw.update({r: f.result() for r, f in reset_jobs.items()})
            active = list(raw)
            futures = {}
            current = {}
            for offset in range(0, len(active), 4):
                robots = active[offset : offset + 4]
                msgs = [
                    dict(
                        raw=raw[r],
                        expert=r // per_expert,
                        version=versions[r],
                        seed=noise_base[r] * 1024 + block[r]
                        if noise_base[r] is not None
                        else 20261108 + r + tick * 8,
                    )
                    for r in robots
                ]
                teacher = pool.submit(
                    predict,
                    operator,
                    1,
                    [dict(m, version="operator") for m in msgs],
                    False,
                )
                acts, kv = predict(serving, 0, msgs, True)
                op, _ = teacher.result()
                for i, r in enumerate(robots):
                    block[r] += 1
                    current[r] = dict(raw=raw[r], prefix=kv[i])
                    futures[r] = submit(r, sim.execute, acts[i], op[i], 520)
            for r, f in futures.items():
                row = f.result()
                e = r // per_expert
                st = sched.experts[e]
                frames[e] += len(row["flags"])
                interventions[e] += sum(row["flags"])
                period_frames[e] += len(row["flags"])
                period_takeovers[e] += sum(row["flags"])
                if pending[r]:
                    sample, actions, flags = pending[r]
                    label = np.concatenate([actions, row["actions"]])
                    sources = flags + row["flags"]
                    if len(label) == 10:
                        sample["actions"] = label
                        if all(sources):
                            online[e].append(sample)
                            credit[e] += 1
                        elif not any(sources):
                            unassisted[r].append(sample)
                pending[r] = (current[r], row["actions"], row["flags"])
                raw[r] = row["raw"]
                if row["done"]:
                    if row["autonomous"]:
                        online[e].extend(unassisted[r])
                        credit[e] += len(unassisted[r])
                    append(
                        out / "episodes.jsonl",
                        dict(
                            expert=e,
                            pairing={
                                k: v
                                for k, v in st.active_episodes[
                                    f"{r}-{episodes[r]}"
                                ].items()
                                if k
                                in ("arm", "pair", "state", "seed", "trial_started")
                            },
                            robot=r,
                            episode=episodes[r],
                            version=versions[r],
                            success=row["success"],
                            autonomous=row["autonomous"],
                            frames=row["steps"],
                            takeover_frames=row["corrections"],
                        ),
                    )
                    if versions[r] == st.production:
                        period_prod[e].append(int(row["autonomous"]))
                    if args.paired:
                        event = sched.end_scored_episode(
                            e,
                            f"{r}-{episodes[r]}",
                            row["autonomous"],
                            row["corrections"],
                            row["steps"],
                        )
                    else:
                        event = sched.end_episode(
                            e, f"{r}-{episodes[r]}", row["autonomous"]
                        )
                    if event:
                        event["elapsed_s"] = time.monotonic() - start
                        validation_sizes[e].append(event["episodes_per_arm"])
                        validated[e] += 1
                        promoted[e] += int(event["decision"] == "promote")
                        append(out / "validation.jsonl", event)
                        if (
                            event["decision"] == "promote"
                            and sum(promoted.values()) == 1
                        ):
                            save(out / "first_promotion.json", event)
                        if event["decision"] == "discard":
                            trainer.discard(e, paths.get(st.production))
                        unused = (
                            event["old_version"]
                            if event["decision"] == "promote"
                            else event["candidate_version"]
                        )
                        if unused != "parent":
                            serving.remove(e, unused)
                            paths.pop(unused).unlink(missing_ok=True)
                    episodes[r] += 1
                    del raw[r]
                    pending[r] = None
                    unassisted[r].clear()
            selected = [e for e in sched.ranked() if credit[e] >= 4 and online[e]][:2]
            if selected:
                batches = {
                    e: [offline[e].sample(rng, trainer) for _ in range(4)]
                    + rng.choices(online[e], k=4)
                    for e in selected
                }
                result = trainer.step(batches)
                append(out / "updates.jsonl", result)
                for e in selected:
                    credit[e] -= 4
                    st = sched.experts[e]
                    st.step += 1
                    if sched.should_validate(e) and st.production_recent:
                        v = f"e{e}-s{st.step}"
                        path = out / (v + ".pt")
                        trainer.snapshot(e, path)
                        paths[v] = path
                        serving.install(e, v, path)
                        sched.start_validation(e, v, time.monotonic() - start)
                        started[e] += 1
                        append(
                            out / "validation.jsonl",
                            dict(
                                event="prediction",
                                elapsed_s=time.monotonic() - start,
                                expert=e,
                                step=st.step,
                                prediction=st.trial["prediction"],
                                candidate=v,
                            ),
                        )
            if tick % 10 == 0:
                save(
                    out / "progress.json",
                    dict(
                        elapsed_s=time.monotonic() - start,
                        steps={e: sched.experts[e].step for e in ids},
                        eligible_online={e: len(online[e]) for e in ids},
                        validations_started=started,
                        validations=validated,
                        promotions=promoted,
                        trials={
                            e: [
                                len(st.trial["candidate_outcomes"]),
                                len(st.trial["production_outcomes"]),
                            ]
                            for e, st in sched.experts.items()
                            if st.trial
                        },
                    ),
                )
            if time.monotonic() >= next_log:
                summary("30min")
                next_log += 1800
        summary("end")
        save(
            out / "result.json",
            dict(
                seconds=time.monotonic() - start,
                steps={e: sched.experts[e].step for e in ids},
                validations=validated,
                promotions=promoted,
                smoke_gate_passed=bool(sum(promoted.values())),
            ),
        )
    finally:
        pool.shutdown()
        for w in workers:
            w.shutdown(cancel_futures=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--paired", action="store_true")
    p.add_argument("--seconds", type=int, help="Optional bounded standalone run")
    p.add_argument("--until-epoch", type=float)
    run(p.parse_args())
