"""Wait for the owned latency campaign, then gate the overnight run on a real promotion."""

import argparse, datetime, json, os, subprocess, sys, time
from pathlib import Path
from zoneinfo import ZoneInfo

proto = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proto))
from evoserve.io import read, save


def main(root, until):
    while (read(root / "queue_status.json", {}) or {}).get(
        "phase"
    ) != "latency_finished":
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
    state = root / "accelerated_status.json"
    for label in ["smoke", "formal"]:
        for attempt in [1, 2]:
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
                for g in [0, 1]
            ):
                save(state, dict(state="waiting_for_two_gpus", phase=label))
                time.sleep(15)
            if time.time() >= until:
                save(state, dict(state="morning_cutoff", phase=label))
                return
            out = root / f"{label}_{attempt}"
            args = [
                sys.executable,
                "-u",
                "-m",
                "evoserve.accelerated",
                "--out",
                str(out),
                "--spec",
                str(proto / "configs/accelerated.machine_a.json"),
                "--seconds",
                str(900 if label == "smoke" else 86400),
                "--until-epoch",
                str(until),
            ]
            with (root / f"{label}_{attempt}.log").open("a") as log:
                p = subprocess.Popen(
                    args, cwd=proto, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                save(
                    state,
                    dict(state="running", phase=label, pid=p.pid, output=str(out)),
                )
                rc = p.wait()
            result = read(out / "result.json", {})
            if rc == 0 and result:
                break
        else:
            save(state, dict(state="execution_failed", phase=label, output=str(out)))
            return
        if label == "smoke" and not result["smoke_gate_passed"]:
            save(state, dict(state="smoke_gate_unmet", result=result, output=str(out)))
            return
    save(state, dict(state="finished", result=result))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--until", required=True)
    a = p.parse_args()
    main(
        a.root,
        datetime.datetime.fromisoformat(a.until)
        .replace(tzinfo=ZoneInfo("America/New_York"))
        .timestamp(),
    )
