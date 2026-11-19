"""Wait for the owned latency campaign, then gate the overnight run on a real promotion."""

import argparse, os, signal, subprocess, sys, time, json
from pathlib import Path


proto = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proto))
from evoserve.io import read, save


def diagnosis(out, names):
    progress = read(out / "progress.json", {}) or {}
    events = []
    path = out / "validation.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows = []
    for e, name in enumerate(names):
        related = [v for v in events if v["expert"] == e]
        completed = [v for v in related if v.get("event") == "validation_finished"]
        counts = progress.get("trials", {}).get(str(e))
        if not related:
            cause = "validation_not_triggered"
        elif related[-1]["event"] == "prediction":
            cause = "collecting_validation_episodes"
        else:
            cause = "candidate_did_not_improve"
        rows.append(
            dict(
                expert=name,
                reason=cause,
                validation_counts=counts,
                steps=progress.get("steps", {}).get(str(e)),
                completed=completed,
            )
        )
    return dict(
        ready=(out / "ready.json").exists(), latest_progress=progress, experts=rows
    )


def main(root, paired=False):
    state = root / "accelerated_status.json"
    save(state, dict(state="waiting_for_latency", morning_cutoff=None))
    while (
        not paired
        and (read(root / "queue_status.json", {}) or {}).get("phase")
        != "latency_finished"
    ):
        time.sleep(10)
    machine = read(proto / "configs/machine_a.example.json")
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="0,1",
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        LIBERO_CONFIG_PATH=machine["assets"] + "/libero_config",
        LP_NUM_THREADS="1",
        GALLIUM_DRIVER="llvmpipe",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZER_DIR=machine["assets"] + "/assets/.cache/openpi",
        LD_LIBRARY_PATH=machine["osmesa_lib"] + ":" + env.get("LD_LIBRARY_PATH", ""),
        PYTHONPATH=str(proto)
        + ":"
        + str(proto.parent / "vendor/RLinf")
        + ":"
        + env.get("PYTHONPATH", ""),
    )
    spec = proto / "configs/accelerated.machine_a.json"
    names = [s["name"] for s in read(spec)]
    if paired:
        names = [name for name in names if name in ("camera", "Y")]
    start = None
    for attempt in (1, 2):
        while any(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(g),
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            for g in (0, 1)
        ):
            save(state, dict(state="waiting_for_two_gpus"))
            time.sleep(15)
        out = root / f"continuous_{attempt}"
        args = [
            sys.executable,
            "-u",
            "-m",
            "evoserve.accelerated",
            "--out",
            str(out),
            "--spec",
            str(spec),
        ]
        if paired:
            args.append("--paired")
        with (root / f"continuous_{attempt}.log").open("a") as log:
            if start is None:
                start = time.monotonic()
            p = subprocess.Popen(
                args,
                cwd=proto,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            info = dict(
                pid=p.pid, output=str(out), started_epoch=time.time(), attempt=attempt
            )
            while p.poll() is None:
                elapsed = time.monotonic() - start
                first = read(out / "first_promotion.json")
                if first:
                    phase = "formal"
                elif elapsed < 900:
                    phase = "smoke"
                else:
                    phase = "extended_smoke"
                if elapsed >= 7200 and not first:
                    # Quiesce the worker before taking its final evidence snapshot.
                    os.killpg(p.pid, signal.SIGSTOP)
                    first = read(out / "first_promotion.json")
                    if not first:
                        reason = diagnosis(out, names)
                        save(root / "no_promotion_diagnosis.json", reason)
                        os.killpg(p.pid, signal.SIGTERM)
                        os.killpg(p.pid, signal.SIGCONT)
                        try:
                            p.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(p.pid, signal.SIGKILL)
                            p.wait()
                        save(
                            state,
                            dict(
                                state="stopped_for_diagnosis", elapsed_s=elapsed, **info
                            ),
                        )
                        return
                    os.killpg(p.pid, signal.SIGCONT)
                    phase = "formal"
                save(
                    state,
                    dict(state=phase, elapsed_s=elapsed, first_promotion=first, **info),
                )
                time.sleep(5)
            if read(out / "first_promotion.json"):
                save(
                    state,
                    dict(
                        state="formal_process_exited", returncode=p.returncode, **info
                    ),
                )
                return
    save(state, dict(state="execution_failed", returncode=p.returncode, **info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--paired", action="store_true")
    options = parser.parse_args()
    main(options.root, options.paired)
