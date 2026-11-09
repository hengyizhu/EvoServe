# EvoServe — π0.5 serving / training reproduction

Private research snapshot for migrating the A6000 serving experiments to an A100 allocation. **No IG-VLA/GIST experiments, model weights, datasets, prefix caches, checkpoints, raw logs or raw result directories are included.** Curated numerical references are [`reference/summary.csv`](reference/summary.csv) and the explicitly requested summaries in [`results/`](results/). The running A6000 experiment directories were read, not modified or restarted.

## Layout

- `experiments/<original_round>/`: all experiment Python entrypoints/dependencies, `protocol*.json/txt`, deterministic sample orders, perturbation configs and train/validation splits. The dated names distinguish historically different protocols; do not mix their checkpoints.
- `vendor/RLinf/`: exact RLinf package/model configs used locally, including π0.5 prefix/suffix interfaces. Upstream commit `91a302ec50ddacd1fdabc5859b85e3661d768236`; original license retained.
- `configs/run_records/`: 53 per-run configuration snapshots, extracted without checkpoints or logs.
- `tools/`: pinned downloads, relocatable workspace creation, serving-input reconstruction, node/EGL configuration and launch wrapper.
- `environment/`: full original `pip freeze`, actual runtime versions, isolated installation script and portable dependency pins.
- `provenance/`: source checksums, model/data/tokenizer identities, input-array hashes and CLI argument inventory.
- `cluster/polaris.pbs`: single-node ALCF Polaris PBS template.
- [`docs/PORTABILITY.md`](docs/PORTABILITY.md): exact CUDA Graph adaptations, MPS/CPU portability changes and validation boundaries.

## Exact model, data and environment

| Component | Source/version |
|---|---|
| Parent | `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT` |
| Model revision | `6222623f635769bfc73c9472e29fab9b7fd8e027` |
| `model.safetensors` SHA256 | `4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683` |
| Norm stats SHA256 | `dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1` |
| Demonstrations | HF dataset `physical-intelligence/libero`, revision `9dfa69510ea9e1613fc54112bc706444b686a231` |
| Python / torch | 3.11.14 / 2.11.0+cu130; torchvision 0.26.0+cu130 |
| CUDA / driver | torch CUDA runtime 13.0 / NVIDIA 580.178.04; no `nvcc` on PATH |
| Actual transformers import | **4.53.2**, supplied by `rlinf-transformer-openpi==4.53.2` |
| openpi / LIBERO | `rlinf-openpi==0.1.1` / `rlinf-libero==0.1.3` |
| Simulation | robosuite 1.4.1; mujoco 3.8.1 |

`pip freeze` also lists `transformers==4.57.6`; that distribution's files were overlaid by the RLinf fork. **Install the fork last.** All versions, including NumPy/JAX/flash-attention, are recorded in [`environment/versions.json`](environment/versions.json) and [`environment/pip-freeze-a6000.txt`](environment/pip-freeze-a6000.txt). RLinf source is imported from the vendored tree, not an unspecified pip checkout.

Training uses full-horizon windows from the pinned LeRobot-format parquet demonstrations, with episode-separated splits recorded in `data_split.json` and later `splits/`. Evaluation uses LIBERO-10 native BDDL/assets and fixed initial states from `rlinf-libero`; no additional evaluation dataset is required. The downloader retrieves only the necessary demonstration episodes and public tokenizer. The 64 serving inputs are **reconstructed**, not committed; their four arrays were checked exactly equal to the A6000 inputs. Model/norm-stat downloads are SHA256 checked.

## Prepare an isolated reproduction

Run downloads on a network-enabled host. Use a fresh workspace for independent reruns; historical scripts intentionally refuse to overwrite many outputs.

```bash
python3.11 -m venv /path/to/vla-venv
source /path/to/vla-venv/bin/activate
bash environment/install.sh
export W=/path/to/scratch/vla-work
python tools/init_workspace.py --work "$W" --model /path/to/models/pi05-parent --data /path/to/datasets/libero
python tools/download_assets.py --work "$W"             # --only model|data|tokenizer also supported
python tools/build_inputs.py --work "$W"
# Run this on the allocated GPU node, where NVIDIA EGL is available:
python tools/configure_node.py --work "$W"
```

Check the A100 node's driver before installing a CUDA 13.0 torch wheel. `TORCH_INDEX_URL` can select a site-supported torch 2.11 wheel channel; any changed software stack must be recorded and Graph parity rerun. Do not install or change the cluster driver. `requirements-portable.txt` deliberately excludes the source host's duplicated NVIDIA/JAX CUDA wheels and architecture-specific flash-attention URL; the full historical freeze is retained separately.

All commands below use this wrapper (the shell function is only a shorthand):

```bash
run() { python tools/run.py --work "$W" --gpu 0 "$@"; }
# Change --gpu and, for a multi-GPU launcher, --allowed-gpus to your allocation.
# --dry-run prints a command without importing torch or allocating a GPU.
```

## Experiment entry commands, parameters and outputs

Run prerequisites before dependent measurements. Output paths below are relative to `$W/<round>/`. Each round's protocol records warmup, repetitions, precision and limits; learning evaluates 5 denoising steps while serving measurements use 10.

| Experiment | Entry command (after the `run` function above) | Expected output / prerequisite |
|---|---|---|
| Training step breakdown, mb1×8 | `run pi05_measure_20261107/training.py` | `training/{action,action_vision_lora}/`: parity, 10 warmup + 50 measured steps, CUDA-event timings, peak memory |
| Eager inference stages, batch 1/4/8/16 | `run pi05_measure_20261107/infer.py` | `inference/`: prefix/10 denoise stages/E2E, FLOPs, memory; run before Graph comparison |
| **CUDA Graph inference** | `run pi05_optimized_20261107/experiment1.py` | `inference/`: eager/Graph action parity and batch timings; needs eager inference summary |
| Training mb4×2 | `run pi05_optimized_20261107/training_mb4.py` | `training_mb4/`: B1/B4 gradients and training costs |
| Original eager interference | `run pi05_measure_20261107/run_interference.py 0 action` | `interference/action/`; replace `action` with `vision`; needs training breakdown |
| Graph load matrix | `run pi05_optimized_20261107/run_matrix.py 0 action` | `matrix/action/`: pure/free/MPS30/MPS50/segmented, clients 1/2/4/8, sync/staggered; needs Graph + mb4 results |
| Repeated Graph matrix | `run pi05_optimized_20261107/run_replicates.py 0 action` | `matrix_replicates/action/`; same prerequisites; `vision` selects vision-LoRA training |
| **KV export/cache transfer cost** | `run pi05_cache_20261107/cache_cost.py` | `cache_cost/`: byte layout, asynchronous D2H, 8-sample H2D, inference P99 |
| **Cached training prototype** | `run pi05_cache_20261107/cache_train.py` | `prototype/`: gradients and standalone throughput; needs mb4 parity output |
| **MPS cache/recompute comparison** | `run pi05_cache_20261107/run_mps.py 0 action` | `matrix_replicates/action/`: pure / MPS30 recompute / MPS30 cache, 3×60 s |
| **Learning variants a/b/c** | `run pi05_cache_20261107/learn.py --variant b --seed 20261107 --steps 1000 --mirror` | `learning/b_seed20261107/`: loss/config/final delta; a=augmentation, b=no augmentation, c=every-5-frame starts; historical mirror task |
| **Learning gate evaluation** | `run pi05_cache_20261107/evaluate_gate.py --name b_seed20261107 --delta "$W/pi05_cache_20261107/learning/b_seed20261107/final.pt" --mirror` | `results/`: adaptation/history 50 episodes each; omit delta for parent; `--task 3 --start 0 --end 50` restricts screening |
| Pinned-input fix + repeated MPS | `run pi05_stability_round2/run_transfer.py 0 action` | `matrix_replicates/action/`: pure/recompute/cache, 5×60 s, input transfer distributions |
| A0/A1/A2 historical sampling | `run pi05_a1_baseline_20261108/run_suite.py` | `solo/` full1000-step means + `co/` 3×60 s, A1 hits, H2D; pinned host cache and exact included sequence |
| Streaming preflight | `run pi05_streaming_20261108/run_stream_suite.py --preflight-only` | Successful smoke status required for formal streaming |
| Streaming A1/A2 | `run pi05_streaming_20261108/run_stream_suite.py` | 4/s and16/s, each20 min in4×5 min windows; new/replay hit rate, prefix cost, actual k |
| LoRA 1000-step LR screen | `run pi05_lora_lr_20261108/train_cached.py --seed 20261108 --screen --job lrscan_train08` | `training/lr*_s20261108/`: EMA1000, full module audit, cached/recompute parity; evaluate via its `eval_worker.py` |
| Tail-latency throttle sweep | `run pi05_overnight_20261108/tail_run.py` | `tail/cap*/`: unlimited/3/2/1 steps/s, each5×60 s; setup explicit configuration below; requires GPU2/3 idle |
| 2000/4000-step learning | See commands below | EMA checkpoints and per-task/per-seed evaluations, frozen-parameter and gradient checks |
| Historical S0–S4 segmented scheduling | See commands below | Calibrations, request/segment/admission timelines; inference-priority at legal training boundaries |

Legacy report builders contain historical narrative/constants and are archived as source; use newly generated JSON/CSV measurements for A100 comparisons, not their old prose.

The complete source-level argparse inventory, including diagnostics, queues, report builders and worker-only entrypoints, is [`provenance/entrypoints.json`](provenance/entrypoints.json). Do not run `matrix_worker.py`, `tail_train.py` or `server.py` by themselves: their corresponding orchestrators send the command files and manage private MPS servers. Historical queue/installer scripts are archived for completeness; use the explicit commands here rather than launching old four-GPU queues on a shared cluster allocation.

### Current 2000-step comparison and 4000-step curves

The three seeds are `20261108`, `20261109`, `20261110`. Run the following training/evaluation pair for each family and seed, with unique names. `full` uses lr1e-5; `lora32_heads` uses lr5e-5 and a 2000-step cosine horizon. All trained deltas stay outside git.

```bash
E="$W/pi05_extended_20261109"
run pi05_extended_20261109/train_cached.py --name full_s08 --family full --lr 1e-5 --seed 20261108 --steps 2000 --split "$E/splits/drawer.json" --sequence "$E/sequences/drawer_20261108_2000.json" --perturb "$E/tasks/drawer_lowlight_0.json" --checkpoints 2000
run pi05_extended_20261109/eval_worker.py --name full_s08_c2000 --adapt-task 3 --delta "$E/training/full_s08/ema_2000.pt" --perturb "$E/tasks/drawer_lowlight_0.json"
# LoRA: change --family to lora32_heads, --lr to 5e-5 and --name consistently.
# For seed09/10, change BOTH --seed and the sequence filename.
```

For the three independent curves, first screen `book` (task5) and `microwave` (task9) with `eval_worker.py --name parent_book_0 --task 5 --adapt-task 5 --perturb "$E/tasks/book_0.json"` (analogous task9/microwave). Each candidate needs **50 episodes and <25 successes**. Candidate files are tried in order, never selected by trained-model success. Then train with `--steps 4000`, the matching `splits/{drawer,book,microwave}.json`, corresponding `sequences/<task>_20261108_4000.json`, the passed perturbation and `--checkpoints 100,250,500,1000,2000,3000,4000`. Evaluate each delta with the correct `--adapt-task` and `--task` (3,5,9). The old drawer shift is an appendix, not an independent task. `--verify-2000 PATH` is only for extending an existing run: the EMA must match exactly at2000 before continuing. From scratch, a single4000-step run supplies the full curve.

For the earlier EMA/augmentation comparison, screen the parent first and construct a gate from the real episode log:

```bash
run pi05_stability_round2/eval_worker.py --name parent_lowlight --task 3 --brightness 0.05
python tools/select_config.py --work "$W" --gate-evaluation "$W/pi05_stability_round2/results/parent_lowlight.jsonl" --brightness 0.05
run pi05_stability_round2/train_worker.py --family full --variant b --seed 20261108 --name full_b_s08 --steps 1000
# --variant a enables augmentation; --cache adds the matched cache-learning branch.
# Evaluate generated EMA deltas with this round's eval_worker.py.
```

Tail sweep configuration is an input, **not a fabricated stability result**:

```bash
python tools/select_config.py --work "$W" --family full --lr 1e-5
run pi05_overnight_20261108/tail_run.py
```

### Historical segmented-runtime replay

First recreate the L2/L3 optimizer states with the original tier trainer; checkpoints are deliberately not shipped. Then invoke the original measurement entrypoints.

```bash
run vla_ladder_20261020/train.py --tier L2
run vla_ladder_20261020/train.py --tier L3
run vla_overnight_20261104/phases.py
run vla_overnight_20261104/replay.py --tier L2
run vla_overnight_20261104/replay.py --tier L3
```

The finite two-period replay lookahead is recorded in the protocol; it is not an online predictor. The preceding `vla_followup_20261020/interference.py` and `interference_parallel.py` entrypoints, their calibration input config and original L1/L2/L3 trainer/evaluator are also included. Before those, regenerate their trained tier checkpoints with `vla_followup_20261020/train.py --tier L2 --seed 20261021 --name L2_s20261021` (matching the job's requested names), rather than reusing an unrelated delta.

## MPS and CPU binding

The `run_mps.py`, `run_matrix.py`, `run_suite.py`, `run_transfer.py`, streaming and tail orchestrators start **private, per-run** MPS pipe/log directories, set training `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=30` (or50), leave inference100, and stop only their own daemon. Do not start a global daemon or change someone else's MPS service. If the cluster forbids MPS, record the condition as skipped, not as a zero-overhead result.

Equivalent manual lifecycle inside an exclusive allocation:

```bash
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)"
export CUDA_MPS_PIPE_DIRECTORY="$W/manual-mps/pipe"
export CUDA_MPS_LOG_DIRECTORY="$W/manual-mps/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
# Launch this run's inference client with percentage100 and training client with30.
# Do NOT run this manual lifecycle around a script that already manages MPS.
printf 'quit\n' | nvidia-cuda-mps-control
```

CPU slots are allocation-relative: eight per GPU, inference first4, training next4. The runner captures the original cpuset before any worker pins itself. Actual per-thread affinities and visible SM counts are recorded by workers. Cache footprints are substantial (~61 GiB for the historical1000-step sequence; ~76–104 GiB for4000-step task-specific sequences), so use at most one full-prefix cached trainer per node unless host RAM is explicitly budgeted.

## ALCF and validation

[`cluster/polaris.pbs`](cluster/polaris.pbs) uses an exclusive one-node PBS allocation and PALS `mpiexec`; set the project, queue, environment and scratch locations. [ALCF documents four A100 GPUs per Polaris node](https://docs.alcf.anl.gov/polaris/running-jobs/using-gpus/), [PBS allocation](https://docs.alcf.anl.gov/polaris/running-jobs/), and [Python environments](https://docs.alcf.anl.gov/polaris/data-science/python/). This is an A100 migration kit, **not an assertion that A100 timings have already been validated**.

```bash
python tools/validate_repository.py
python tools/run.py --work "$W" --gpu 0 --dry-run pi05_optimized_20261107/experiment1.py
```

Numerical checks execute before each measured GPU workload. Keep the original BF16/FP32 mixture and both prefix-producer paths matched; a B1/B4 numerical difference must not be silently treated as a caching error. Keep SLO/window-level outliers and host-transfer costs. `reference/summary.csv` contains measured A6000 values only, never expected A100 performance.
