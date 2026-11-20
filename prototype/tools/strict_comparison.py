"""Preserve a live rejected camera candidate, audit it, then start two independent campaigns."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTO))
from evoserve.io import save, read


def events(old):
    return [
        json.loads(l)
        for l in (old / "validation.jsonl").read_text().splitlines()
        if l.endswith("}")
    ]


def free(card):
    return not subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(card),
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def main(root, old):
    state = root / "comparison_status.json"
    archive = root / "archive"
    archive.mkdir(exist_ok=True, parents=True)
    machine = read(PROTO / "configs/machine_a.example.json")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="2,3",
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        LIBERO_CONFIG_PATH=machine["assets"] + "/libero_config",
        LP_NUM_THREADS="1",
        GALLIUM_DRIVER="llvmpipe",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TMPDIR="/var/tmp",
        TOKENIZER_DIR=machine["assets"] + "/assets/.cache/openpi",
        LD_LIBRARY_PATH=machine["osmesa_lib"]
        + ":"
        + os.environ.get("LD_LIBRARY_PATH", ""),
        PYTHONPATH=str(PROTO) + ":" + str(PROTO.parent / "vendor/RLinf"),
    )
    status = read(old.parent / "accelerated_status.json")
    pid = status["pid"]

    def owned():
        p = Path("/proc") / str(pid) / "cmdline"
        return (
            p.is_file()
            and b"evoserve.accelerated" in p.read_bytes().split(b"\0")
            and os.fsencode(str(old)) in p.read_bytes().split(b"\0")
        )

    save(
        state,
        dict(
            phase="waiting_for_preserved_camera_candidate",
            unavailable_historical_version="e0-s6342",
        ),
    )
    for_epoch = set()
    while owned():
        hist = events(old)
        for p in old.glob("e0-s*.pt"):
            try:
                os.link(p, archive / p.name)
            except (FileNotFoundError, FileExistsError):
                pass
        triggers = [
            r
            for r in hist
            if r["expert"] == 0
            and r["event"] == "prediction"
            and r["candidate"] not in for_epoch
            and (archive / (r["candidate"] + ".pt")).exists()
        ]
        if triggers:
            trigger = triggers[-1]
            idx = hist.index(trigger)
            # Only use a candidate for which the production version at validation start is available.
            prod = next(
                (
                    r["candidate_version"]
                    for r in reversed(hist[:idx])
                    if r.get("decision") == "promote" and r["expert"] == 0
                ),
                "parent",
            )
            if prod != "parent" and (archive / (prod + ".pt")).exists():
                v = trigger["candidate"]
                for_epoch.add(v)
                finished = next(
                    (r for r in hist[idx:] if r.get("candidate_version") == v), None
                )
                if finished and finished["decision"] == "promote":
                    continue
                while not all(free(g) for g in (2, 3)):
                    time.sleep(10)
                for attempt in (1, 2):
                    out = root / f"audit_{v}_{attempt}"
                    save(
                        state,
                        dict(
                            phase="independent_50_pair_audit",
                            candidate=v,
                            production=prod,
                            output=str(out),
                        ),
                    )
                    with (root / f"audit_{v}_{attempt}.log").open("w") as log:
                        rc = subprocess.call(
                            [
                                sys.executable,
                                "-u",
                                str(PROTO / "tools/audit_camera_slots.py"),
                                "--out",
                                str(out),
                                "--candidate",
                                str(archive / (v + ".pt")),
                                "--production",
                                str(archive / (prod + ".pt")),
                            ],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                    if rc == 0:
                        break
                else:
                    save(
                        state,
                        dict(
                            phase="audit_failed",
                            output=str(out),
                            reason="inspect audit log before comparison",
                        ),
                    )
                    return
                while owned():
                    finished = next(
                        (r for r in events(old) if r.get("candidate_version") == v),
                        None,
                    )
                    if finished:
                        break
                    time.sleep(5)
                if not finished or finished["decision"] != "discard":
                    continue
                result = read(out / "result.json")
                assert all(r["episodes"] == 50 for r in result.values())
                save(
                    root / "audit_report.json",
                    dict(
                        historical_version_unavailable="e0-s6342",
                        candidate=v,
                        production=prod,
                        original_decision=finished,
                        standalone=result,
                        weight_checks=read(out / "weights.json"),
                        batch_binding=read(out / "batch_binding.json"),
                    ),
                )
                # A direction reversal can be the consequence of the verified renderer fix.
                # Require the actual interleaved-worker regression before dispatching.
                regression = read(root / "fixed_render_test.json", {})
                if not regression.get("reset_equal") or not regression.get(
                    "step_equal"
                ):
                    save(
                        state,
                        dict(
                            phase="waiting_for_renderer_fix_verification",
                            output=str(out),
                        ),
                    )
                    return
                break
        time.sleep(5)
    else:
        save(state, dict(phase="old_campaign_exited_before_auditable_rejection"))
        return
    # Old experiment owns only this verified process group. Both new runs start from parent.
    if owned() and os.getpgid(pid) == pid:
        os.killpg(pid, signal.SIGTERM)
    (root / "stop_preserving").touch()
    while not all(free(g) for g in range(4)):
        time.sleep(10)
    jobs = {}
    for mode, cards in [("sequential", ("0", "1")), ("fixed50", ("2", "3"))]:
        dest = root / mode
        dest.mkdir(exist_ok=True)
        argv = [
            sys.executable,
            "-u",
            str(PROTO / "tools/run_accelerated_queue.py"),
            "--root",
            str(dest),
            "--paired",
            "--cards",
            *cards,
        ]
        if mode == "fixed50":
            argv.append("--fixed50")
        with (dest / "supervisor.log").open("a") as log:
            jobs[mode] = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ).pid
    save(state, dict(phase="both_campaigns_dispatched", supervisors=jobs))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--old", type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.old)
