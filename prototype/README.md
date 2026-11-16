# EvoServe prototype v0

This branch implements real LIBERO episode collection, π0.5 CUDA Graph serving,
serving-prefix export, independent action LoRA experts, candidate validation, and
episode-boundary deployment. GPU testing is gated on completion of the preceding
75%/80% system measurements. `prototype_task_state` records actual progress; the
presence of this code does not mean that M1/M2/M3 passed.

## Fixed protocol

- Machine A: GPU 0/1 for the experiment; M1/M2 production and training on GPU0,
  auxiliary simulated human on GPU1. M3 may borrow an idle GPU2/3 for the operator,
  otherwise GPU1 uses the lowest-priority CUDA stream. Placement is fixed across
  the four settings. Read-only checks of **225 W** power limits; continuously
  record GPU 2/3 external workload and host CPU pressure. No power-setting commands.
- Every robot has a **250 ms wall-clock request period** and 20 Hz physics.
  Latest completed camera observations are submitted at the fixed releases.
  At each period boundary, install that period's on-time response and execute
  five actions in the next period. Decode all ten model actions: on timeout use
  the old block's remainder, then hold pose (zero displacement, prior gripper
  command). Late responses are discarded. Nominal requests without responses,
  during active execution count as deadline failures. Reset and initial settling
  actions issue no requests and are excluded from SLO; each episode starts its
  own 250 ms cadence at its first formal request. Reset time is reported separately.
- CPU camera rendering uses a separate simulator state. Record observation age,
  capture frame, dispatch lag, actual request rate, and simulator overrun cycles.
  Each prefix is paired with **the ten actions actually executed after its capture
  frame**, including fallback/hold actions. Never relabel it with intended actions.
  Duplicate capture frames are not inserted twice. No padded training windows.
- Batch 1 inference, ten denoising steps, BF16, graph constants patched explicitly
  in `backend.py`; graph output is checked against eager output on startup.
- Action LoRA rank 32: q/k/v/o, gate/up/down, action adaptive-norm modulation;
  full action input/output and time-MLP projections. **20,876,320** trainable
  parameters/expert. Frozen visual backbone, no augmentation. FP32 optimizer
  masters and EMA, lr 5e-5, 50 warmup steps, cosine over 4000 steps to 10%, EMA
  .999, effective batch 8/expert (4 × accumulation 2). Each expert has a separate
  optimizer, EMA, update counter and sampling/noise seed.
- Online pool: simulated-operator corrections plus autonomous successful real
  episodes; uniformly retain at most eight eligible complete windows/episode; 512 samples and 12 GiB maximum/trainer with random replacement.
  Four recent plus four historical samples per update, with replacement. No
  offline demonstration windows are directly added; operator weights are explicitly
  pretrained on the full existing adaptation-training split. Keys: `(episode, frame, version)`;
  mismatched versions are dropped. Rejection restores production parameters and
  clears candidate optimizer state; promotion retains training state.
- Unrestricted MPS for colocated service/training, private per-GPU pipe/log dirs.
  Every five seconds use the deadline miss rate (target ≤1%): reduce merged expert
  count first, then duty; duty can reach zero. Restore duty then group size.
- Every 30 seconds rank expert `e` by

  ```text
  robots[e] * max over L in {250,500,1000,2000,4000} of
    (max(prediction(s+L), production_success)
      - max(prediction(s), production_success)) / L
  ```

  `scheduler.jsonl` stores all five budgets, selected budget and score. Validation
  uses **current** `prediction(s)`, never the future ranking budget. Trigger at
  `max(.05, 1.5*sqrt(.25/50))` predicted gain, or 1000 updates since last validation.
- Prediction uses real recent production episodes plus a fixed seven-task mean
  normalized action-only learning curve (`configs/learning_prior.json`). Fit
  amplitude and time scale after real validations; with just one distinct step
  the scale stays at its prior because two parameters are not identifiable.
  X/Y have only parent/final observations: intermediate prior values are explicitly
  interpolated. Historical curves used five denoising steps; serving here uses ten.
- Validate 50 candidate and 50 contemporaneous production episodes; allocate half
  the expert's robots to each arm (one robot alternates episodes). Promote only if
  the real candidate **autonomous** success count is higher; any episode needing
  takeover is a failure for policy prediction/validation even if the operator
  subsequently completes the task. No fabricated successes or forced
  promotion. Keep each episode's model version fixed.

## Layout and commands

Use the pinned Python 3.11.14 environment documented in `../machine_b/README.md`.
Model revision and checksum, full dependency lock, dataset downloads, vendored
RLinf/LIBERO setup, and perturbations are in `machine_b/` and `vendor/`. Prototype
assets point to the generated `work/vla_followup_20261020` directory.

```bash
# From repository root. CPU tests do not load a CUDA model.
CUDA_VISIBLE_DEVICES='' PYTHONPATH=prototype python -m unittest discover -s prototype/tests -v

# Real CPU physics + CPU rendering; requires libOSMesa.so in LD_LIBRARY_PATH.
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  LP_NUM_THREADS=1 GALLIUM_DRIVER=llvmpipe \
  python prototype/tools/probe_periodic_cpu.py --out /path/to/cpu-probe.json

# Post-measurement GPU smoke only. Synthetic labels are explicitly excluded from learning results.
CUDA_VISIBLE_DEVICES=0 python prototype/tools/gpu_preflight.py \
  --assets /path/to/work/vla_followup_20261020 --out /path/to/preflight

# One actual setting, configured by the campaign or a saved protocol.
PYTHONPATH=prototype python -m evoserve.runtime \
  --config /path/to/setting.json --out /path/to/new-run-directory

# Autonomous campaign; refuses to start GPUs without measurements_complete.json.
python prototype/run_campaign.py --root /path/to/campaign \
  --config /path/to/machine-a.json
```

Machine A's `machine_a.example.json` contains local paths; replace them when
migrating. `tools/install_osmesa_local.sh` extracts the pinned Ubuntu package into
an isolated user directory; it does not change system libraries or the driver.
Set `LIBERO_CONFIG_PATH` to the assets' `libero_config` directory. Worker GPU
visibility and MPS are handled by `runtime.py`, not by a global daemon.
CPU affinity: serving 0–3 / 8–11, training 4–7 / 12–15, two CPU cores per simulator
starting at 16, coordinator 30–31. Check these CPU indices on a different host.

## Measurement and milestone queue

Existing measurement controller runs eight cases (75/80% × 1/2/4/8 experts), each
with a 60-second smoke then ten minutes. Three isolated training updates after the
window provide longest-kernel timing; these are not concurrent kernel profiles.
A0 trace uses fixed MPS30%, feedback off, 70% load, five seconds; export format
matches A2. System traces and original logs remain outside Git.

Prototype GPU preflight checks graph inference, live-shaped prefix consumption,
a real update, a merged update with independent optimizers, and candidate loading.
Each operational stage retries once and then records the failure. Dependent long
runs do not start after a failed milestone gate; independent capacity tests continue.

- M1: ten-minute smoke, then a one-hour formal run (low-light drawer, one expert/robot).
  Passing requires an actual validated promotion and SLO≥99%; absence of improvement
  is reported as not demonstrated, not a successful milestone.
- Capacity: fixed-rate robots N=1..6, 60-second screening plus matched-arrival
  no-simulator replay; ten-minute verification of the largest passing screen and
  its next count. Report the **largest verified count in this tested range**,
  not an extrapolated maximum. Replay preserves arrival times and uses a fixed
  sixteen-observation corpus per expert; raw full-image traces are not retained.
- M2: ten-minute smoke with four experts/robots, then one hour only if it passes.
  More robots are not added if fixed-rate capacity already fails.
- M3: only after M1/M2 pass; each method has a ten-minute smoke, then **three hours
  each**, ordered ours / no learning / recompute / split. All use the same seed
  and schedule, changing drawer to low light and book to image shift at 5400 s.
  The smoke compresses the change time to 300 s. Ours must work before baseline
  long runs. Split puts all service on GPU 0 and dedicated cached training on 1.

Every run exports `summary.json`, per-robot request/episode/cycle records,
`scheduler.jsonl`, `validation_results.jsonl`, `operator_hourly.csv`, controller windows and GPU/CPU
monitoring. Campaign summaries are `prototype_summary.csv`, `原型汇总.md`, and
`milestones/*.json`. Raw images, prefix caches, weights, checkpoints and logs are
not committed. The outer queue supplies the 30-minute heartbeat and survives
terminal closure. No success-rate result is inferred from a CPU test or synthetic
GPU smoke.

`measurement/` contains a source snapshot of the running measurement controller,
worker modules, profile/export code and fixed service-time calibration. These
archived launchers retain Machine A's absolute historical dependency paths; use
`tools/init_workspace.py` and its path-remapping approach before executing them
on a different machine. They are not invoked by the portable prototype backend.
The active queue remains in the experiment directory, so editing this branch
does not restart or interrupt a measurement already in progress.

## Simulated operator correction (before M1)

`configs/operators.machine_a.json` pins the existing full-adaptation-data action
expert EMA checkpoints by SHA256 and task. The source adaptation training sets
contain 20 demonstration episodes; this means the full existing training split,
not the entire train/validation dataset. Low-light drawer uses seed08/step4000,
with historical 86% success (five-denoising-step evaluation); prototype inference
uses ten steps and measures its own outcomes. No operator weights are committed. For migration, transfer the selected EMA files
separately, retain their checksums, and set `operator_manifest` in the machine config.
The registry also audits a camera-view expert for future use; the current task
queue still uses drawer/book/microwave/X.

Take over until episode end when either condition first holds:

1. Maximum displacement from the beginning of the latest **20 control intervals**
   (21 sampled positions, 1 s at 20 Hz) is **strictly below 2 mm**.
2. The episode has used **70% of its wall-clock time limit**, without success.
   With 600 actions at 20 Hz this is 21 s of a 30 s episode.

The next fixed-period request obtains operator actions. Pending old production
commands are discarded at takeover; hold pose until a timely operator block
arrives. Production inference continues at 4 requests/s. A separate suffix-only
graph on the auxiliary GPU uses the early exported frozen VLM prefix and the
operator's action weights. Source export overhead remains in production latency;
operator computation is asynchronous and separately accounted. The legacy
same-GPU path is retained only for reproducing the original diagnostic results.
The operator does not update the learner's weights directly. GPU preflight
compares this reused-prefix path with a full operator forward and verifies that
switching back restores production output exactly.

Corrective windows must contain ten actually executed operator actions. Mixed
production/operator transitions and hold actions are excluded, and pre-takeover
failed production windows are discarded. Operator windows remain eligible even
if the episode eventually fails; their failure flag is preserved. Autonomous
windows require episode success without takeover. During candidate rescue,
shadow production inference supplies the cache and its true production version;
no candidate cache is relabeled as production. `training_windows.jsonl` and each
robot's `executed_actions.jsonl` record frame/source alignment.

Report total completions (including assistance) separately from autonomous
successes. Scheduler predictions, 50-episode validation and M1 improvement use
autonomous outcomes only. `operator_hourly.csv` reports actual takeover counts
and rates per expert; `takeovers.jsonl` records the trigger and frame. A one-hour
M1 run is a verification budget, not an asserted success-rate improvement.
The data-source mode remains simulated correction until Machine B supplies
reinforcement-learning evidence; RL is not enabled by this change.

## Auxiliary operator placement (2026-11-15 amendment)

Set `operator_gpu: 1` and `namespace: remote_operator_v1` in the campaign config.
Previous same-GPU results remain in their original run/state directories. Production
prefix completion events release KV plus state through a bounded host-memory ring
before production denoising finishes. The auxiliary worker uses an action-only
CUDA Graph; it does not recompute the VLM. Production response completion defines
its SLO independently of operator completion. Late operator actions are discarded;
operator delays and holds remain visible in simulator logs and never become false
corrective labels. Auxiliary GPU model memory, transfers and execution costs are
measured, not treated as free in the artifact.

`operator_hourly.csv` records takeovers, operator GPU seconds (H2D, parameter load,
noise generation, suffix graph, decode), H2D separately, source D2H and host-copy
seconds, and late/dropped corrections. Raw operator requests also separate denoising
from total GPU execution. GPU2/3 external workloads and 225 W limits are logged.
Identical M3 placement does not imply identical realized takeover counts: report
actual counts and GPU time for each of the four methods.

```bash
# CPU transport test deliberately includes late operator replies.
CUDA_VISIBLE_DEVICES='' python prototype/tools/probe_robot_cpu.py \
  --assets /path/to/assets --osmesa-lib /path/to/osmesa/lib \
  --out /path/to/probe --remote-operator

# Requires BOTH GPU0/1 idle; tests early KV delivery, action parity and overlap.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=prototype python prototype/tools/remote_operator_preflight.py \
  --assets /path/to/assets --operators prototype/configs/operators.machine_a.json \
  --out /path/to/remote-preflight
```

## Reset timing and dynamic batching revision

`revision: reset_batch` selects the repair queue; M2/M3 remain suspended until
reset tests, graph validation, capacity measurements and concurrent measurements
pass. `resets.jsonl` reports reset/settling boundaries separately; request deadlines
remain 250 ms once active execution begins. No threshold is relaxed.

`batch_backend.py` shares one frozen VLM and captures batches 1/2/4/8. A batch has
one VLM forward and ten denoising iterations. Expert/version-specific LoRA and
head parameters are vectorized per sample; model-install commands are barriers.
`batching.py` collects requests for at most 10 ms or until the selected cap is
full. Partial batches use the next captured size, with padding explicitly logged.
Only real requests produce replies and caches. Each request's KV slice keeps its
own identifier and version; operator prefixes are exported at the prefix event.

GPU calibration warms ten iterations and measures fifty per size. It separately
checks captured versus eager vectorized execution, mixed model versions, and
bit-exact per-request cache transport. Numerical differences from ordinary batched
execution are recorded without changing results. Pure-inference capacity includes
cache export and a pinned host-memory consumer, with no training model/update.

The capacity sweep measures staggered and synchronized 4 Hz clients. M2 uses the
largest ten-minute-verified staggered count, registering all four experts; experts
without robots can only train if they already have cached data, and cannot trigger
validation without real episodes. No artificial demonstrations are injected into
M1/M2/M3. Capacity is reported within the tested range, including failing boundary
points. The 70%/85% throughput tests use aggregate rates `4*Nmax*.70/.85` and are
labeled equivalent-robot loads, not fractional physical robots. They train one
expert using labeled demonstration windows through the actual serving cache path;
these throughput runs make no learning-quality claims.

Entry points: `tools/prepare_batch_inputs.py` (small aligned demonstration corpus),
`tools/batch_calibrate.py` (GPU graph/copy validation and batch latency),
`tools/batch_load.py` (open-loop serving/cache/feedback measurement), and
`tools/batch_stage.py` (resumable capacity, concurrent tests and gated M1→M2→M3).
All raw images, caches and logs remain outside Git. GPU0/1 power stays 225 W;
GPU2/3 workloads and host load continue to be monitored. Each failed item retries
once; independent measurements continue. No failed prerequisite starts a long run.
