"""Start the GPU 0/1 full loop, its GPU 2 placement worker and a GPU 3 baseline/operator."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTO))
from evoserve.io import read, save


def live(root, name):
    p = int((root / (name + ".pid")).read_text())
    try:
        args = Path(f"/proc/{p}/cmdline").read_bytes().split(b"\0")
        return os.fsencode(str(root / name)) in args or (
            name == "operator" and os.fsencode(str(root / "operator.sock")) in args
        )
    except FileNotFoundError:
        return False


def launch(root, name):
    machine = read(PROTO / "configs/machine_a.example.json")
    assets = machine["assets"]
    env = dict(
        os.environ,
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        LP_NUM_THREADS="1",
        GALLIUM_DRIVER="llvmpipe",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        LIBERO_CONFIG_PATH=assets + "/libero_config",
        TOKENIZER_DIR=assets + "/assets/.cache/openpi",
        LD_LIBRARY_PATH=machine["osmesa_lib"]
        + ":"
        + os.environ.get("LD_LIBRARY_PATH", ""),
        PYTHONPATH=str(PROTO) + ":" + str(PROTO.parent / "vendor/RLinf"),
        CUDA_VISIBLE_DEVICES="0,1,2" if name == "full" else "3",
    )
    if name == "operator":
        args = ["evoserve.shared_operator", str(root / "operator.sock")]
    else:
        args = [
            "evoserve.accelerated",
            "--paired",
            "--online-only",
            "--out",
            str(root / name),
            "--spec",
            str(PROTO / "configs/accelerated.machine_a.json"),
            "--operator-socket",
            str(root / "operator.sock"),
            "--start-gate",
            str(root / "start"),
        ] + (["--branches"] if name == "full" else [])
    with (root / (name + ".log")).open("a") as f:
        child = subprocess.Popen(
            [sys.executable, "-u", "-m", *args],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (root / (name + ".pid")).write_text(str(child.pid))


def main(root, existing):
    if not existing:
        root.mkdir(parents=True, exist_ok=False)
        for name in ("operator", "full", "baseline"):
            launch(root, name)
    while not (
        all((root / n / "initialized.json").exists() for n in ("full", "baseline"))
        and (root / "operator.sock").exists()
    ):
        if not all(live(root, n) for n in ("operator", "full", "baseline")):
            raise RuntimeError(
                "Initialization failed. Inspect the component logs before releasing the two runs."
            )
        time.sleep(2)
    (root / "start").touch(exist_ok=False)
    start = time.time()
    while True:
        status = {}
        for name in ("full", "baseline"):
            d = root / name
            progress = read(d / "progress.json", {})
            if not live(root, name):
                status[name] = "stopped"
                continue
            if (d / "first_promotion.json").exists():
                status[name] = "formal"
            elif time.time() - start >= 7200:
                save(
                    d / "no_promotion_diagnosis.json",
                    dict(
                        elapsed=time.time() - start,
                        progress=progress,
                        validation_log=str(d / "validation.jsonl"),
                        placement_log=str(d / "placement.jsonl"),
                    ),
                )
                pid = int((root / (name + ".pid")).read_text())
                if os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGTERM)
                status[name] = "no_promotion"
            else:
                status[name] = "before_first_promotion"
        save(
            root / "queue_status.json",
            dict(started=start, elapsed_s=time.time() - start, loops=status),
        )
        if all(v in ("stopped", "no_promotion") for v in status.values()):
            break
        time.sleep(10)
    if live(root, "operator"):
        os.kill(int((root / "operator.pid").read_text()), signal.SIGTERM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--existing", action="store_true")
    a = parser.parse_args()
    main(a.root.resolve(), a.existing)
