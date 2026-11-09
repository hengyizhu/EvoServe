"""Independent confirmation queue; each seed remains a separate reported version."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "vla_ladder_20261020"


def status(value):
    temp = ROOT / "evaluation_status.tmp"
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(ROOT / "evaluation_status.json")


def summarize():
    base = [json.loads(s) for s in (ROOT / "results/parent.jsonl").read_text().splitlines()]
    reference = {(r['task'], r['index']): r for r in base}
    report = {}
    for path in (ROOT / 'results').glob('*.jsonl'):
        if 'smoke' in path.name:
            continue
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        if len(rows) != len(reference):
            continue
        paired = []
        tasks = {}
        for t in (3, 0):
            subset = [r for r in rows if r['task'] == t]
            tasks[str(t)] = dict(success=sum(r['success'] for r in subset), episodes=len(subset),
                rescue=sum(r['success'] and not reference[(t,r['index'])]['success'] for r in subset),
                damage=sum(not r['success'] and reference[(t,r['index'])]['success'] for r in subset),
                all_attempt_steps=sum(r['policy_steps'] for r in subset))
        for r in rows:
            paired.append({'task':r['task'],'index':r['index'],
                'parent_success':reference[(r['task'],r['index'])]['success'],
                'candidate_success':r['success']})
        report[path.stem] = {'tasks':tasks,'paired':paired}
    (ROOT / 'results/independent_summary.json').write_text(json.dumps(report,indent=2))


def main():
    manifest = json.loads((ROOT / 'audit/independent_states.json').read_text())
    expected = sum(v['count'] for v in manifest.values())
    parent_pid = json.loads((ROOT / 'audit/initial_processes.json').read_text())['parent']['pid']
    while True:
        file = ROOT / 'results/parent.jsonl'
        n = len(file.read_text().splitlines()) if file.exists() else 0
        if n == expected:
            break
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            raise RuntimeError(f'Parent evaluation exited after {n}/{expected} episodes')
        status({'state':'waiting_for_parent','episodes':n,'expected':expected})
        time.sleep(10)
    # Allow the parent to close its renderer before reusing GPU 3.
    time.sleep(5)
    summarize()
    completed = ['parent']
    queue = [('original_L1_200', OLD/'training/L1/step_200.pt'),
             ('original_L1_1000', OLD/'training/L1/step_1000.pt'),
             ('L1_seedA_1000', ROOT/'training/L1_seedA/step_1000.pt'),
             ('L1_seedB_1000', ROOT/'training/L1_seedB/step_1000.pt')]
    for name, delta in queue:
        while not delta.with_suffix('.ready.json').exists():
            train_status = delta.parent/'status.json'
            state = json.loads(train_status.read_text()) if train_status.exists() else {}
            pid = state.get('pid')
            if pid:
                os.kill(pid, 0)
            status({'state':'waiting_for_checkpoint','version':name,'completed':completed})
            time.sleep(10)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='3', MUJOCO_EGL_DEVICE_ID='3',
            TOKENIZER_DIR=str(ROOT/'assets/.cache/openpi'),OMP_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (ROOT/f'logs/eval_{name}.log').open('w') as log:
            p = subprocess.Popen(['taskset','-c','24-27',str(ROOT/'venv/bin/python'),
                str(ROOT/'evaluate.py'),'--name',name,'--delta',str(delta)],env=env,
                stdout=log,stderr=subprocess.STDOUT)
            status({'state':'evaluating','version':name,'pid':p.pid,'completed':completed})
            rc = p.wait()
        if rc:
            raise RuntimeError(f'{name} failed: exit {rc}')
        completed.append(name)
        summarize()
    status({'state':'complete','completed':completed,'episodes':expected*len(completed)})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        status({'state':'failed','traceback':traceback.format_exc()})
        raise
