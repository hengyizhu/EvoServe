# H100 supplemental measurements

The campaign closed at the user’s request after the staggered 4-robot, 8-expert case. The archive contains 50 original cases, 12 cache breakdowns, 34 training optimization candidates and 14 robot groups in the current matrix. Eight planned robot groups were cancelled before starting. The raw data from one earlier superseded 7-robot case is preserved under historical/. The export measurements used the full host KV export.

Single H100. Standalone cached training uses 3 warmups and 12 timed steps. H2D and compute are allowed to overlap. Clean timings and instrumented timings are reported separately.

| Rank | Experts | Cache path | Clean median step ms | Clean mean step ms | Expert steps/s | CPU % (100 = one core) | Intra/inter-op threads | OS threads sampled (active ≥10 ms) |
|---:|---:|---|---:|---:|---:|---:|---|---|
| 32 | 1 | original | 333.36 | 336.46 | 2.97 | 100.1 | 1 / 96 | 5 (2) |
| 32 | 1 | prefetch | 323.54 | 320.77 | 3.12 | 125.5 | 1 / 96 | 6 (3) |
| 32 | 1 | resident | 251.19 | 253.79 | 3.94 | 100.0 | 1 / 96 | 5 (2) |
| 32 | 8 | original | 774.53 | 760.94 | 10.51 | 100.1 | 1 / 96 | 5 (2) |
| 32 | 8 | prefetch | 681.82 | 703.88 | 11.37 | 142.6 | 1 / 96 | 6 (3) |
| 32 | 8 | resident | 614.49 | 626.95 | 12.76 | 100.1 | 1 / 96 | 5 (2) |
| 128 | 1 | original | 326.16 | 327.61 | 3.05 | 100.1 | 1 / 96 | 5 (2) |
| 128 | 1 | prefetch | 375.84 | 377.19 | 2.65 | 120.9 | 1 / 96 | 6 (3) |
| 128 | 1 | resident | 280.17 | 283.44 | 3.53 | 100.0 | 1 / 96 | 5 (2) |
| 128 | 8 | original | 609.08 | 618.17 | 12.94 | 100.1 | 1 / 96 | 5 (2) |
| 128 | 8 | prefetch | 741.30 | 760.85 | 10.51 | 150.7 | 1 / 96 | 6 (4) |
| 128 | 8 | resident | 603.71 | 613.85 | 13.03 | 100.1 | 1 / 96 | 5 (2) |

Profiled wall-clock attribution in ms per step, averaged over 2 independent step windows in the same trace. Columns through Other sum to Profiled wall. H2D includes observation, action, and prefix transfers. Cache lookup measures host references to pinned tensors. DMA reads from host memory are included in H2D. Forward includes device-side input/parameter assembly inside each forward region. Other includes Python, host waits, and unclassified packing kernels. DMA/compute overlap is reported separately. Instrumentation affects elapsed time.

| Rank | Experts | Path | Host lookup | H2D | Forward | Backward | Optimizer + EMA | Other | Profiled wall | H2D/compute overlap |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | original | 0.058 | 4.336 | 13.922 | 18.037 | 2.130 | 406.578 | 445.061 | 0.000 |
| 32 | 1 | prefetch | 0.000 | 4.414 | 13.949 | 18.048 | 2.125 | 653.670 | 692.207 | 0.163 |
| 32 | 1 | resident | 0.079 | 0.454 | 13.914 | 18.064 | 2.131 | 391.653 | 426.295 | 0.000 |
| 32 | 8 | original | 0.765 | 33.538 | 25.834 | 44.903 | 17.047 | 924.478 | 1046.565 | 0.000 |
| 32 | 8 | prefetch | 0.000 | 34.388 | 25.451 | 44.478 | 17.039 | 964.273 | 1085.630 | 2.261 |
| 32 | 8 | resident | 0.671 | 3.436 | 25.804 | 44.867 | 17.056 | 799.241 | 891.074 | 0.000 |
| 128 | 1 | original | 0.060 | 4.327 | 14.337 | 19.184 | 3.878 | 406.701 | 448.488 | 0.000 |
| 128 | 1 | prefetch | 0.000 | 4.341 | 14.348 | 19.161 | 3.881 | 663.745 | 705.476 | 0.183 |
| 128 | 1 | resident | 0.094 | 0.427 | 14.333 | 19.166 | 3.876 | 415.491 | 453.386 | 0.000 |
| 128 | 8 | original | 0.510 | 34.019 | 29.318 | 52.012 | 31.119 | 787.499 | 934.478 | 0.000 |
| 128 | 8 | prefetch | 0.000 | 34.286 | 29.091 | 51.539 | 31.116 | 1004.327 | 1150.358 | 2.252 |
| 128 | 8 | resident | 0.652 | 3.424 | 29.321 | 52.043 | 31.112 | 795.731 | 912.283 | 0.000 |

| Rank | Experts | Original pinned cache | Dataset size MiB | Prefetch speedup | Resident speedup | Largest absolute loss difference (prefetch / resident) |
|---:|---:|---|---:|---:|---:|---|
| 32 | 1 | True | 272.3 | 1.030x | 1.327x | 0 / 0 |
| 32 | 8 | True | 272.3 | 1.136x | 1.260x | 0 / 0 |
| 128 | 1 | True | 272.3 | 0.868x | 1.164x | 0 / 0 |
| 128 | 8 | True | 272.3 | 0.822x | 1.009x | 0 / 0 |

The production host cache is pinned once at initialization. Transfers use non_blocking=True on a dedicated CUDA stream; the production step synchronizes before starting the next batch. Prefetch stages the next batch on that stream using one additional host thread. In the resident reference only the prefix cache stays on device; observations and action labels still follow the original input path.

| Rank | Experts | Path | Lightweight wall ms | Forward stream ms | Backward stream ms | Optimizer stream ms | Bare host lookup ms | Profiled KV H2D ms |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | 1 | original | 309.71 | 105.69 | 165.79 | 6.76 | 0.0050 | 3.873 |
| 32 | 1 | prefetch | 359.34 | 148.51 | 201.06 | 7.10 | 0.0092 | 0.000 |
| 32 | 1 | resident | 257.43 | 103.44 | 118.54 | 6.86 | 0.0036 | 0.000 |
| 32 | 8 | original | 661.98 | 153.29 | 178.98 | 61.00 | 0.0324 | 30.123 |
| 32 | 8 | prefetch | 641.08 | 263.80 | 296.40 | 75.16 | 0.0410 | 0.000 |
| 32 | 8 | resident | 607.48 | 139.94 | 177.49 | 53.57 | 0.0280 | 0.000 |
| 128 | 1 | original | 267.31 | 107.93 | 118.93 | 7.16 | 0.0047 | 3.865 |
| 128 | 1 | prefetch | 415.60 | 218.71 | 182.74 | 10.75 | 0.0091 | 0.000 |
| 128 | 1 | resident | 252.58 | 101.70 | 113.06 | 6.95 | 0.0039 | 0.000 |
| 128 | 8 | original | 699.11 | 141.11 | 270.94 | 55.78 | 0.0309 | 30.556 |
| 128 | 8 | prefetch | 747.25 | 322.35 | 334.10 | 84.47 | 0.0388 | 0.000 |
| 128 | 8 | resident | 587.04 | 142.74 | 180.72 | 53.80 | 0.0291 | 0.000 |

CUDA stream spans include enqueue gaps. The separate clean run is used for speedups. CPU utilization includes all process threads, relative to one logical core.

The optimization matrix, CPU stack rankings, staged numerical verification and timing-selected worker configurations are in [training-optimization/summary.md](training-optimization/summary.md). The 7-robot measurement is a pure-inference baseline. The completed 4- and 5-robot training co-runs use the fastest tested exact trajectory for the corresponding training mode and expert count. Inference uses one intra-op thread in every group.

Each robot sends 4 requests/s at evenly spaced times. Every measurement window is 300 seconds. SLO is the percentage finishing within 250 ms. Cache co-runs fix the merge width at rank 32 and start at 100% commanded duty. The original 5-second deadline-miss feedback changes only duty, with pending-queue protection active. Feedforward lookup is disabled during these measurements. Duty is the controller command averaged over the wall window. Raw controls, dispatch times and actual admission widths are archived. Recompute uses the inference-priority stream and a single training expert.

| Robots | Arrival | Rate/s | Mode | Experts | Training path / threads | P99 ms | SLO % | Expert steps/s | Mean commanded duty % | CPU mean % of 8 vCPUs | CPU 1 s peak % of 8 vCPUs |
|---:|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | staggered | 16 | inference | — | — | 175.23 | 100.000 | — | — | 12.82 | 13.98 |
| 4 | staggered | 16 | cache | 1 | graph / 1 | 228.69 | 99.917 | 7.137 | 99.48 | 23.13 | 24.21 |
| 4 | staggered | 16 | cache | 2 | graph / 4 | 250.45 | 98.958 | 10.547 | 85.24 | 25.73 | 30.83 |
| 4 | staggered | 16 | cache | 4 | graph / 4 | 230.18 | 99.771 | 17.493 | 97.10 | 25.78 | 29.69 |
| 4 | staggered | 16 | cache | 8 | graph / 4 | 251.68 | 98.875 | 21.760 | 76.81 | 23.70 | 27.71 |
| 5 | staggered | 20 | inference | — | — | 198.02 | 100.000 | — | — | 12.83 | 14.04 |
| 5 | staggered | 20 | export | — | — | 278.05 | 78.533 | — | — | 12.84 | 14.12 |
| 5 | staggered | 20 | cache | 1 | graph / 1 | 266.36 | 94.783 | 0.110 | 1.75 | 13.21 | 23.08 |
| 5 | staggered | 20 | cache | 2 | graph / 4 | 275.95 | 96.167 | 0.253 | 1.92 | 13.35 | 29.04 |
| 5 | staggered | 20 | cache | 4 | graph / 4 | 295.10 | 90.383 | 0.453 | 1.86 | 13.38 | 29.51 |
| 5 | staggered | 20 | cache | 8 | graph / 4 | 306.96 | 85.050 | 0.560 | 2.13 | 13.43 | 25.17 |
| 5 | staggered | 20 | recompute | 1 | graph / 1 | 656.84 | 3.033 | 3.153 | 100.00 | 24.60 | 26.52 |
| 5 | staggered | 20 | slack | 1 | graph / 1 | 258.27 | 94.650 | 0.143 | 1.79 | 13.31 | 24.98 |
| 7 | staggered | 28 | inference | — | — | 396.90 | 35.619 | — | — | 12.83 | 13.92 |

## H100 feedforward table by robot count

Largest measured merge width satisfying both SLO ≥99.5% and P99 ≤220 ms. Values are tested before rounding. Zero means no measured width qualified; throughput is unavailable for that selection.

| Robots | Max experts | Expert steps/s | Selected P99 ms | Selected SLO % | Status |
|---:|---:|---:|---:|---:|---|
| 4 | 0 | — | — | — | no qualifying width |
| 5 | 0 | — | — | — | no qualifying width |
| 6 | — | — | — | — | cancelled before measurement |
| 7 | — | — | — | — | baseline only |

## Time-slack guard

While a request is queued, the worker defers a training update when the oldest queued wait plus the predicted service time of the current padded batch exceeds 200 ms. Predictions use the original single-GPU static end-to-end means. Feedback and stream/MPS settings match the old guard. Old-rule choices use staggered measurements at the same robot count.

| Robots | Arrival | Chosen experts | Selection basis | P99 ms | SLO % | Expert steps/s | Mean duty % |
|---:|---|---:|---|---:|---:|---:|---:|
| 5 | staggered | 1 | diagnostic width 1; no qualifying old width | 258.27 | 94.650 | 0.143 | 1.79 |

## Host cache export overhead

Every dispatched inference batch exports KV to host RAM through the suite exporter. The latest batch is retained; the timed loop performs no disk writes for KV. CPU percentages use the entire Pod CPU cgroup over the 300-second measurement, divided by eight logical cores. Peak is the maximum of one-second intervals.

| Robots | P99 baseline → export ms | Δ P99 ms | SLO baseline → export % | CPU mean baseline → export % | CPU peak baseline → export % | Host export GiB |
|---:|---|---:|---|---|---|---:|
| 5 | 198.02 → 278.05 | +80.03 | 100.000 → 78.533 | 12.83 → 12.84 | 14.04 → 14.12 | 127.89 |
