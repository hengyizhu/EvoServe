# EvoServe prototype v0

This branch implements real LIBERO episode collection, π0.5 CUDA Graph serving,
serving-prefix export, independent action LoRA experts, candidate validation, and
episode-boundary deployment. GPU testing is gated on completion of the preceding
75%/80% system measurements. `prototype_task_state` records actual progress; the
presence of this code does not mean that M1/M2/M3 passed.

## Fixed protocol

- Machine A: GPU 0/1 only, read-only checks of **225 W** power limits; continuously
  record GPU 2/3 external workload and host CPU pressure. No power-setting commands.
- Every robot has a **250 ms wall-clock request period** and 20 Hz physics.
  Latest completed camera observations are submitted at the fixed releases.
  At each period boundary, install that period's on-time response and execute
  five actions in the next period. Decode all ten model actions: on timeout use
  the old block's remainder, then hold pose (zero displacement, prior gripper
  command). Late responses are discarded. Nominal requests without responses,
  including missed reset/transition periods, count as deadline failures.
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
- Online pool: successful real episodes, uniformly retain at most eight complete
  windows/episode; 512 samples and 12 GiB maximum/trainer with random replacement.
  Four recent plus four historical samples per update, with replacement. No
  offline demonstrations are secretly added. Keys: `(episode, frame, version)`;
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
  the real candidate success count is higher. No fabricated successes or forced
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

- M1: ten-minute smoke, then up to two hours (low-light drawer, one expert/robot).
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
`scheduler.jsonl`, `validation_results.jsonl`, controller windows and GPU/CPU
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
