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
from evoserve.io import read, save, append
from evoserve.gpu_inventory import inventory


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
    layout = read(root / "gpu_layout.json")
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
        CUDA_VISIBLE_DEVICES=",".join(map(str, layout[name])),
        EVOSERVE_VISUAL_CARD=str(layout["visual"]),
        EVOSERVE_VISUAL_PEAK=layout["visual_peak"],
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


def select_layout(cards, peak):
    # The profiling worker is ours, and only its reserved card can overlap startup.
    profile_pid = (read(peak.parent / "observed.json", {}) or {}).get("pid")

    def profiler(pid):
        try:
            cmd = Path("/proc", str(pid), "cmdline").read_bytes().split(b"\0")
            return (
                pid == profile_pid
                and b"evoserve.visual_branch" in cmd
                and os.fsencode(str(peak.parent)) in cmd
            )
        except OSError:
            return False

    idle = [i for i, c in sorted(cards.items()) if not c["processes"]]
    ours = [
        i
        for i, c in cards.items()
        if c["processes"] and all(profiler(p["pid"]) for p in c["processes"])
    ]
    eligible = set(idle + ours)
    if len(idle) < 2:
        return None
    if eligible.issuperset({0, 1, 2, 3}) and {0, 1, 3}.issubset(idle):
        serve, train, vision, base = 0, 1, 2, 3
    else:
        serve, train = idle[:2]
        base = idle[-1] if len(idle) >= 3 else serve
        vision = next(
            i
            for i in sorted(cards, key=lambda i: (i not in eligible, i != 2, i))
            if i not in {serve, train, base}
        )
    op = max(
        idle,
        key=lambda i: (
            cards[i]["free_mib"],
            i == base if len(idle) > 2 else i == train,
            i != vision,
        ),
    )
    return dict(
        full=[serve, train],
        visual=vision,
        baseline=[base],
        operator=[op],
        visual_peak=str(peak),
    )


def main(root, existing, peak):
    if not existing:
        root.mkdir(parents=True, exist_ok=False)
        while True:
            cards = inventory()
            layout = select_layout(cards, peak)
            append(
                root / "startup_gpu_inventory.jsonl",
                dict(time=time.time(), cards=cards, allocation=layout),
            )
            if layout:
                break
            time.sleep(30)
        save(root / "gpu_layout.json", layout)
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
            else:
                status[name] = "before_first_promotion"
        save(
            root / "queue_status.json",
            dict(started=start, elapsed_s=time.time() - start, loops=status),
        )
        if "stopped" in status.values() or (root / "stop_requested").exists():
            for n in ("full", "baseline"):
                if live(root, n):
                    pid = int((root / (n + ".pid")).read_text())
                    if os.getpgid(pid) == pid:
                        os.killpg(pid, signal.SIGTERM)
            save(
                root / "comparison_end.json",
                dict(time=time.time(), elapsed_s=time.time() - start, states=status),
            )
            break
        time.sleep(10)
    if live(root, "operator"):
        os.kill(int((root / "operator.pid").read_text()), signal.SIGTERM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--existing", action="store_true")
    parser.add_argument("--visual-peak", type=Path, required=True)
    a = parser.parse_args()
    main(a.root.resolve(), a.existing, a.visual_peak.resolve())
