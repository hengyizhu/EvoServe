# r128 forced two-stage campaign

Machine-A continuation of the recalibrated feedforward experiment. This is an
experiment harness using the existing pinned pi0.5 environment and local data.
All paths in `configs/` refer to the immutable experiment snapshot. Before
migrating, remap the environment root, manifest paths, data roots, and cache roots.

The system driver `../../measurement/system/feedforward_night.py` runs latency
measurements on GPU 0, then invokes this queue. GPU 1 may be occupied by another
user: it is polled for actual compute processes and joins only after 30 seconds
without any. GPU 0 prioritizes Y and microwave; GPU 1 prioritizes lowlight and
book. An available worker can claim remaining jobs from the other queue. GPU 2/3
are monitored only. No power setting or external process is changed.

```bash
python learning_queue.py --root /path/to/immutable/learning
```

Each task runs a one-update visual smoke, one full evaluation episode using the
six-process evaluation infrastructure, and one cached continuation update with
gradient parity and optimizer/EMA preservation checks. Formal training starts
only after these pass. Failed jobs retry once; completed evaluation indices are
preserved. The queue writes a heartbeat every 30 minutes and updates
`learning_summary.csv` and `summary.md` after each completed evaluation point.

Action LoRA covers every action-expert attention q/k/v/o projection, all
gate/up/down FFN matrices, and AdaRMS modulation projections, rank/alpha 128.
Action input/output and time MLP projections train fully. Meta-device validation
finds **77,007,904 action parameters** and **4,713,984 visual parameters**.
Vision LoRA is rank/alpha 16 on all SigLIP MLP fc1/fc2. Activation checkpointing
is disabled. Both parameter groups use lr 2e-4, with 50-step warmup and cosine
to 10% over 4000 total steps; EMA .999, microbatch 4 × accumulation 2.

The forced visual stage skips the low-distance branch. Every 50 steps, the
unchanged strict signed rule checks `d(t)-d(t-100) < .1*d(t)`; two consecutive
passes freeze vision. Raw visual weights at that boundary produce the rebuilt
cache; the action optimizer, FP32 master weights, EMA, and random state continue.
Metrics at 25/50/75/100 are recorded as before. Fixed EMA evaluations occur at
1000/2000/4000, with 50 initial states and six simulator processes sharing one
batched GPU policy. Each episode has its own fixed CUDA noise generator so
worker completion order does not change its random stream. Batch arithmetic
can differ from old serial inference; the renderer, policy step budget, five
executed actions per replan, and existing evaluation denoising setting remain
unchanged. Latency experiments separately use ten denoising steps.

Generated caches live under the campaign-owned `/var/tmp` or `/dev/shm` roots,
with a 32-sample pinned LRU. Capacity is checked before population. Only generated
caches are removed after training; requested EMA checkpoints and raw metrics
are retained. Diagnostic restart snapshots retain the last committed recovery
boundary plus newer states. Training/cache time excludes loading and evaluation;
diagnostic time is a separate CSV column, with an additional inclusive total.

Machine-B action-only 4000-step comparison values (Y 68%, microwave 70%, lowlight
84%, book 98%) are **user supplied**, not newly measured on this host.
