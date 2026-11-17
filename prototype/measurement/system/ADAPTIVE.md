# Adaptive expert count and duty experiment

Entry: `adaptive_campaign.py queue --root <fresh-output-directory>` using the existing
π0.5 Python environment. For reproducibility, copy this entire `system` directory
to `<output>/code` before launching; retain its SHA256 manifest. The live campaign
is `Experiments/pi05_adaptive_experts_20261116`. Raw outputs are excluded from Git. The GIL probe builds `gil_probe.c` on first import; a C compiler and the active Python development headers are required. No prebuilt `.so` is required in the repository.

The queue first runs a 300-second smoke test (100 seconds each at 70%, 85%, 70%),
then four independent 1800-second measurements, then 80% fixed-count sensitivity
for 1/2/4/8 experts. Each failed task retries once, records the error, and advances.
The adaptive long run requires successful smoke execution. Heartbeats are written
every 30 minutes; GPU/host monitoring is every 2 seconds. Under the November16
user-approved amendment, only GPU0 must be exclusive. GPU1/2/3 activity is monitored
and does not block admission or interrupt a measurement. Power on GPU0 remains
225 W; the runner checks it and never changes it. The generic `measure.py` default
retains its historical isolation policy; this campaign explicitly selects GPU0 only.

Resume resource-interrupted tasks using
`recover_resource_interruptions.py --root <output> --resume` after ensuring the
previous waiting supervisor has exited. The recovery queue first adds an
inference-only 70%/85%/70% baseline (600 seconds each), then finishes the two missing
main comparisons and the80% sensitivity points. Each new run performs a60-second
preflight before formal measurement. The inference-only baseline retains identical
A2 export, host receipt and resident-model allocation but performs zero training
updates. GPU kernel/state checks and warmup occur outside the timed interval.

`compute_process_monitor.jsonl` records processes and CPU affinity on all cards every
5 seconds. Foreign compute on GPU0 still interrupts the measurement; own worker
PIDs and the owned MPS server are allowed. Reports mark old GPU1-idle results and
new shared-host results separately and write `external_load_comparison.csv`; a
change in external activity must not be silently treated as an equivalent condition.

Use `adaptive_campaign.py report --root <output>` to rebuild the campaign summary.
Each completed run writes `windows_5s.csv`, `stages.csv`, `result.json`, and raw
request/step/controller records. The adaptive run also emits PNG/PDF timelines.
Campaign outputs are `summary.md`, `stage_summary.csv`, and `total_expert_steps.csv`.

## Fixed protocol

- Historical inference-only reference: 142.49900977 ms. Arrival rate is target
  occupancy divided by this reference, with identical deterministic staggered
  arrivals in every comparison. Fresh isolated calibration is recorded separately.
  Each main run is 70% for 600 s, 85% for 600 s, then 70% for 600 s. Requests have
  absolute 250 ms deadlines; missing responses count as failures.
- All configurations hold eight independent action LoRA r32 experts. Selected
  groups rotate round-robin, each expert receives microbatch 4 × accumulation 2,
  four new samples plus four replay samples, no augmentation. New pool: latest512;
  replay: the existing1000 windows. A2 service prefixes use the established pinned
  host transport. Optimizer, EMA, counter and RNG state are separate per expert.
- Count choices are1/2/4/8. The adaptive group starts at1 and duty1. Every5 s,
  timeout rate >1% halves count; at count1 it reduces duty. Existing duty feedback
  gain5 is retained, with duty clamped to[0,1]. Zero-timeout windows recover duty
  by0.05. Three consecutive zero-timeout windows whose incoming duty is1 double
  the count, at most8. A window that only recovers duty to1 does not count toward
  this streak. Nonzero timeouts reset the streak. Count changes take effect only
  before the next training step. Every feedback window is consumed, including
  windows published while a training step was running.
- Baselines: fixed8 + duty feedback; fixed1 + duty feedback; fixedMPS30% + fixed1
  with no duty feedback. Other feedback runs use unrestricted MPS100%. CPU serving
  cores0–3 and training cores4–7 are fixed. The feedback target remains1% timeouts.
- The same cache-population warmup precedes each configuration. Training weights,
  optimizer/EMA/counters and per-expert RNG are restored before the measured run;
  no reset occurs at the two load transitions. Serving input sequence restarts at
  the same sample index. Kernel shapes1/2/4/8 are warmed before service starts.

## Checks and interpretation

`python test_adaptive.py` checks threshold boundaries, count hysteresis, full pause
and recovery, consumption of multiple feedback windows, and arrival continuity.
GPU initialization checks all four merged shapes and verifies that inactive
experts' weights, EMA and optimizer states are unchanged. Completed-run validation
checks each expert's optimizer/EMA counters across every measured count switch.

Throughput is completed expert updates per wall-clock second, not the requested
count times an estimated speed. Stage-crossing updates belong to their completion
stage; updates finishing after the30-minute boundary are excluded. The controller's
P99 at deadline observation can be infinite for pending responses; final window
P99 is computed from all recorded responses and explicitly includes unfinished
requests. Both are retained in the CSV. This experiment compares the measured
baselines; it does not assume that adaptive control must win or establish dominance
over every possible fixed setting.
