# EvoServe 单卡 H100 测量套件

**云上必须在 tmux 中运行，代码、环境和结果都放在挂载好的 /workspace network volume。** 先确认挂载，进入持久会话：

```bash
findmnt -T /workspace
tmux new -s evoserve
cd /workspace
```

## 分阶段推理优化与共跑

在已挂载的 `/workspace` 和 tmux 中运行 `./run_inference_opt.sh`。入口在当前 GPU 上为 batch 1、2、4、8 分别标定前缀和十步去噪，使用相同观测与相同噪声同原整图路径对照。每组 12 个输入批次，前两个批次只用于数值校验，后十个批次同时记录 GPU Event 耗时。所有候选均用 CUDA Graph 前缀，分别测原分段图、手工优化去噪及编译去噪。手工路径为固定权重预先合并 LoRA 和 Q/K/V 投影，预计算十个时刻的 AdaRMS 参数。每次请求准备一次前缀 KV、掩码和旋转位置编码，十次迭代复用 KV，只覆盖动作后缀。BF16 动作分支使用高效 SDPA，整段十步用一个 CUDA Graph 回放。更换权重须重新构建推理图。数值误差随结果保存。选择须通过固定低照度 50 回合配对判据（成功数相差 ≤4，双侧精确 McNemar p >0.05），在合格路径中按本卡实测耗时选型。

标定使用独立进程隔离候选崩溃，失败组合记入 `selection.json`。某档没有可用候选时回退到原始整图。若组合计划在容量或共跑阶段失败，先用独立进程复查参考推理，再用原始整图在 `fallback/` 重新测量。参考环境也无法工作时终止。基线回退后个别配置仍失败会记入 `failed.jsonl` 并继续其他配置。

A6000 的十步去噪耗时、逐步优化、配对行为验证和单步 profiler CSV 见[去噪实测](reference/denoise-a6000/README.md)。

默认扫描 3–10 台错峰机器人，每台 4 Hz，250 ms 时限，每档持续 300 秒。在容量附近两个机器人数上，分别测复用缓存的 1、2、4、8 个专家，以及单专家重算。以 99% 请求满足时限为容量门槛。默认矩阵有 90 分钟正式到达窗口，另需编译、模型加载、预热与过载排空时间。各 batch 的分阶段中位数、逐元素误差在 `results/inference-opt-*/<batch>-<prefix>-<denoise>.json`，入选组合在 `plan.json`，容量和共跑明细在 `capacity.json` 和 `summary.csv`。指定新输出目录可避免复用旧测量。

## H100 机器人数量补测

在上述 tmux 和 /workspace 中执行。已有 checkout 可先运行 `git switch h100-suite && git pull --ff-only`。

```bash
git clone -b h100-suite --single-branch https://github.com/hengyizhu/EvoServe.git
cd EvoServe
./run_supplement.sh
```

按 5、6、4 台机器人依次测量，每台 4 Hz 错峰，时限 250 ms。每档包含纯推理、纯推理加按需导出、复用前缀（1/2/4/8 个专家）和每步重算（1 个专家）。21 组各 5 分钟，正式计时合计 105 分钟；预计已装环境约 2–2.5 小时，首次安装下载再预留 30–75 分钟，重试另计。单卡短版命令为 `./run_supplement.sh --quick`，7 个配置各用 1 台机器人测量 60 秒。每个测量进程先预热并跑 2 秒短测。

- 负载为固定到达回放。每台机器人每 104 次请求构成一个模拟回合（520 个控制步，每请求 5 步）。从每个完整回合均匀无放回抽取 8 次，记录截断回合在测量期内实际到达的抽样请求。复用组和纯推理加导出组采用同一随机序列。每个选中请求把自己的 KV 复制到预分配 pinned 环形缓冲区，独立 stream 等待前缀就绪事件；静态推理图下一次覆盖同一地址前等待复制完成。
- 训练共用 r32、每专家有效 batch 8（4 × 2）、GPU 组 batch、按 dtype 打包的 pinned 主机 KV 样本库、CUDA Graph 前反向、图外 AdamW 和 EMA。复用组的样本库使用事先经服务路径算出的 KV；本轮导出的 KV 用于测量导出成本。重算组在每个 microbatch 的图内执行冻结的前缀。两组直接调用 H100 已验证损失、master 权重和 EMA 逐步一致的 graph 实现，单专家训练线程数为 1，多专家为 4；推理 CPU 线程数固定为 1。
- 缓存组优先使用无配额 MPS，容器不可用时使用单进程优先级 streams。重算沿用推理优先的单进程 streams。每 5 秒按超过 250 ms 的请求比例调节占空比。两组均使用原 H100 QueueSlack：队首等待时间加预计 batch 服务耗时超过 200 ms 时暂停发起训练步。同档机器人数共享纯推理阶段标定的各 batch 耗时。每个配置只改变固定专家数，反馈不改变专家数。
- 结果在 `results/supplement-日期-时间/`，同名 `.tar.gz` 可下载。逐项 JSON 记录派发与完成时间、抽样 ID、实际 transport、P99、SLO 和吞吐。`summary.csv` 为全矩阵；`feedforward.json` 按机器人数列出满足 SLO ≥99.5% 且 P99 ≤220 ms 的最大有效专家数，以及两种训练方式各自在 SLO ≥99% 下的最高实测专家步/秒。没有合格点时分别写 0 或 null。A6000 的 GPU 2/3 外部负载记录在 `external_gpu_load.csv`。

GPU 0 的 RTX A6000（225 W）已完成全部 7 项单机器人预演，每项 240 次请求。8 专家缓存组和重算组的 SLO 分别为 97.08% 和 84.58%，均按实测值保留。详见 [A6000 预演汇总](results/a6000-supplement-rehearsal/summary.csv)。

## 原全套测量

在已配置 NVIDIA 驱动、CUDA 与 MPS 的 x86_64 Ubuntu H100 云主机上，以 root 在空目录运行以下命令。私有仓库需要有读取权限。

```bash
git clone --branch h100-suite --single-branch https://github.com/hengyizhu/EvoServe.git && cd EvoServe && ./run_all.sh
```

`run_all.sh` 自动执行 `setup.sh`，安装 Python 3.11.14、锁定依赖与 PyTorch 2.11.0（驱动 580 及以上选 cu130，525–579 选 cu128），下载数据并校验模型 SHA256。需要兼容所选 CUDA wheel 的驱动、至少 80 GB 可写磁盘、能够访问 GitHub、PyPI、PyTorch wheel 源、Hugging Face 和 Google Storage。可通过 `SUITE_HOME` 选择环境与模型所在的磁盘。使用默认 250 ms 请求时限，保留云主机的功耗设置。

**A6000 短版预演**于 2026-11-22T14:08-05:00 完成，50/50 个测量项目均写出结果，已核对汇总与归档内容。

**时间预算**为约 3–4.5 小时，属于 H100 运行前估算。12 个固定专家数共跑点为 60 分钟，负载变化 15 分钟，重算对照 10 分钟，两个容量点的两种训练方法共 12 分钟，视觉与重建共跑 9 分钟。另预留容量搜索、独立训练、profile、各进程模型加载与预热约 40–75 分钟，安装下载约 30–75 分钟。重试、下载限速及共享云主机可能延长计费时间。

每次运行独立写入 `results/h100-日期-时间/`，单项完成即刷新 `summary.md` 和 `index.json`，最终同目录旁生成 `.tar.gz`。硬件、CUDA、功耗及依赖在 `hardware.json`，测量配置、标准错误与结果在各项目的 `attempt-N/`。每项先执行预热或短测，再测正式区间。失败保留原因、重试一次、继续独立项目；依赖缺失则标为跳过。套件结束时有失败会返回非零。缩短版命令为 `./run_all.sh --quick`。

| 项目 | 协议与结果 |
|---|---|
| 梯度 | 固定四个真实 LIBERO 10 动作窗口、相同噪声和 flow 时间。独立比较服务 CUDA Graph 的 B1 缓存与训练 B1 冻结前缀重算，均拼为 B4 反向。保留逐位一致、最大绝对差与相对 L2。 |
| 推理与容量 | 单策略 CUDA Graph，10 步去噪。静态 B1/2/4/8/16 分别记录前缀和去噪 GPU Event 耗时及含预处理的墙钟服务时间。按每机器人 4 Hz 固定到达，动态等待最多 10 ms，按到达时刻计延迟；填充到下一档 batch。同步与错峰分别按所有整数机器人数量扫描至实测乐观吞吐上界加余量，最后一点通过则继续。N* 为该次扫描内 SLO≥99% 的最大实测数。 |
| 独立训练 | microbatch 4×累积 2；动作 LoRA r32/r128 含全量动作映射和时间 MLP。视觉阶段另开 SigLIP MLP fc1/fc2 r16，保留穿过 VLM 的梯度并关闭 checkpointing。冻结视觉逐步重算、从主机读取服务 KV、合并 1/2/4/8/16 个独立优化器均用相同示范窗口。记录墙钟步时、框架峰值 allocated 和 reserved 字节。 |
| 共跑 | 负载 = 合计到达率 × 该卡 B1 实测平均服务时间。默认 r32。容器无法启用 MPS 时自动回退到单进程优先级 streams，逐项记录实际 transport。缓存组优先用无固定配额 MPS，与推理分进程；步边界检查等待队列，五秒超时率反馈调占空比。70/80/85% 各测 1/2/4/8 个专家，每点 300 秒。SLO≥99.5%、P99≤220 ms 且实际有更新才选入前馈表，无可行项记 0。负载变化三段各 300 秒，按一秒窗口观测到达率，档间取保守值。重算对照用单进程高优先级推理 stream、低优先级训练 stream；另在两种 N* 的原始到达流下对比。 |
| 重计算与 profile | 实测视觉训练、B4 重建并拷回主机，以及相应 stream 共跑。Kineto/CUPTI 分离采集缓存 1/8 专家、重算及视觉训练，输出最长单个 GPU kernel 名、微秒和事件数。profile 数据不混入延迟。 |

共跑 `windows.json` 包含每批请求索引、调度/完成时刻、控制变更和各段 P99、SLO、专家步/秒、平均目标占空比。每个预定到达都计入分母。缓存组读取固定示范样本的服务缓存，测得的是预生成后的训练吞吐；重建与搬回单列。

模型 `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT`，revision `6222623f635769bfc73c9472e29fab9b7fd8e027`，权重 SHA256 `4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683`。数据 `physical-intelligence/libero` 固定到 `9dfa69510ea9e1613fc54112bc706444b686a231`。按仓库的输入清单下载相关 episodes，并使用保留完整动作标签的观测。历史实验入口保留如下。

# EvoServe — π0.5 serving / training reproduction

**Machine B learning setup:** [installation, pinned environment and1000-step book check](machine_b/README.md). This branch adds a learning-only migration path for a CPU-contended dual-A6000 host.

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
