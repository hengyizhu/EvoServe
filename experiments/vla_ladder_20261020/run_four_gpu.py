"""Start verified tier jobs and evaluate published checkpoints on GPU 3."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent


def write_status(value):
    path = ROOT / "run_status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def launch(script, args, gpu, log_name):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
               TOKENIZER_DIR=str(ROOT / "assets/.cache/openpi"),
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    with (ROOT / "logs" / log_name).open("a") as log:
        return subprocess.Popen([str(ROOT / "venv/bin/python"), str(ROOT / script), *args],
                                env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)


def summarize():
    base = [json.loads(line) for line in (ROOT / "results/parent.jsonl").read_text().splitlines()]
    reference = {(x["task"], x["index"]): x for x in base}
    summary = {}
    paired = {}
    for path in (ROOT / "results").glob("*.jsonl"):
        if "smoke" in path.name:
            continue
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        if len(rows) != 40:
            continue
        summary[path.stem] = {}
        paired[path.stem] = []
        for task in (3, 0):
            subset = [x for x in rows if x["task"] == task]
            summary[path.stem][str(task)] = {
                "success": sum(x["success"] for x in subset), "attempts": len(subset),
                "all_attempt_steps": sum(x["policy_steps"] for x in subset),
                "rescue": sum(x["success"] and not reference[(task,x["index"])]["success"] for x in subset),
                "damage": sum(not x["success"] and reference[(task,x["index"])]["success"] for x in subset)}
        for row in rows:
            prior = reference[(row["task"],row["index"])]
            paired[path.stem].append({"task":row["task"],"index":row["index"],
                                     "parent_success":prior["success"],"candidate_success":row["success"],
                                     "parent_steps":prior["policy_steps"],"candidate_steps":row["policy_steps"]})
    (ROOT / "results/summary.json").write_text(json.dumps(summary, indent=2))
    (ROOT / "results/paired.json").write_text(json.dumps(paired, indent=2))


def main():
    for tier in ("L1", "L2", "L3"):
        status = json.loads((ROOT / "preflight" / tier / "status.json").read_text())
        assert status["state"] == "complete", f"{tier} preflight not complete"
        assert (ROOT / "preflight" / tier / "update_check.json").exists()
        assert (ROOT / "preflight" / tier / "step_1.ready.json").exists()
    while True:
        rows = (ROOT / "results/parent.jsonl").read_text().splitlines()
        if len(rows) == 40:
            assert len({(x["task"],x["index"]) for x in map(json.loads, rows)}) == 40
            break
        write_status({"state": "waiting_for_parent_baseline", "completed_episodes": len(rows)})
        time.sleep(10)
    summarize()
    jobs = {tier: launch("train.py", ["--tier", tier], gpu, f"train_{tier}.log")
            for gpu, tier in enumerate(("L1", "L2", "L3"))}
    pids = {tier:p.pid for tier,p in jobs.items()}
    (ROOT / "audit/training_processes.json").write_text(json.dumps(pids, indent=2))
    pending = [(tier,step) for step in (200,1000) for tier in ("L1","L2","L3")]
    evaluation = None
    active = None
    completed = []
    failed = {}
    while pending or evaluation or any(p.poll() is None for p in jobs.values()):
        for tier,p in jobs.items():
            if p.poll() not in (None, 0):
                failed[tier] = p.returncode
        if evaluation and evaluation.poll() is not None:
            if evaluation.returncode:
                failed[f"eval_{active}"] = evaluation.returncode
            else:
                completed.append(active)
                summarize()
            evaluation = None
            active = None
        if evaluation is None:
            for tier,step in list(pending):
                ready = ROOT / "training" / tier / f"step_{step}.ready.json"
                if ready.exists():
                    active = f"{tier}_{step}"
                    evaluation = launch("evaluate.py",
                        ["--name", active, "--delta", str(ready.with_name(f"step_{step}.pt"))],
                        3, f"eval_{active}.log")
                    pending.remove((tier,step))
                    break
                if tier in failed:
                    pending.remove((tier,step))
        write_status({"state": "running", "training_pids": pids,
                      "training_exit_codes": {t:p.poll() for t,p in jobs.items()},
                      "evaluation": active, "evaluation_pid": evaluation.pid if evaluation else None,
                      "evaluations_completed": completed, "pending": pending, "failed": failed})
        time.sleep(10)
    summarize()
    write_status({"state": "failed" if failed else "complete",
                  "failed": failed, "evaluations_completed": completed,
                  "scope": "fixed-tier training and closed-loop evaluation; interference experiment not started"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_status({"state": "orchestrator_failed", "traceback": traceback.format_exc()})
        raise
