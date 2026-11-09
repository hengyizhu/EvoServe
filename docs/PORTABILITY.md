# Portability and provenance

`experiments/` is a byte-preserving snapshot of the serving experiments; the per-file original paths and SHA256 values are in `provenance/source_manifest.json`. The IG-VLA/GIST projects are excluded. RLinf is the actual local clean checkout at `91a302ec50ddacd1fdabc5859b85e3661d768236`, with its package and model configs vendored under `vendor/RLinf` and its original license retained.

The on-disk RLinf tree has no uncommitted patch. **CUDA Graph compatibility is implemented by runtime method substitutions**, not a git diff:

| Location in `pi05_optimized_20261107/graph_serving.py` | Captured replacement |
|---|---|
| `Pi0.embed_prefix` | `torch.tensor(ar_mask, device=...)` → device `torch.zeros(..., dtype=torch.bool)` |
| `Pi0.embed_suffix` | `ar_mask[0] = True` → `ar_mask[:1].fill_(True)` |
| each Gemma attention `forward` | `torch.tensor(big_neg, ...)` → device `torch.full((), big_neg, ...)` |

These substitutions, static observation/noise buffers, external CUDA Events, graph replay, eager/action checks, prefix-KV export ordering, pinned input handling, and the LoRA adapters are all included. Do not replace them with unmodified upstream inference. Relevant dependencies: `vendor/RLinf/rlinf/models/embodiment/openpi/pi0.py`, `modules/gemma.py`, `pi05_stability_round2/pinned_serving.py`, `pi05_a1_baseline_20261108/trace_serving.py`, `pi05_cache_20261107/core_mps.py`, and later `common.py` copies.

`tools/init_workspace.py` writes a **new** runnable tree and records changed paths in `workspace.json`. It rewrites experiment/model/data/interpreter roots; replaces only literal MPS `<84`/`>=84` checks with the target device's full SM count; and translates the original virtual CPU groups through the allocation's allowed CPU list. It does not change loss, precision, batch sizes, random order, warmup, deadlines, margins, EMA, or the three CUDA Graph substitutions. `tools/configure_node.py` regenerates EGL device ordinals instead of assuming the source machine's reversed mapping. The runner refuses GPU IDs outside an explicit numeric CUDA allocation; UUID/MIG allocations need a deliberate physical-ordinal mapping.

CPU layout: GPU ordinal `g` uses virtual CPU slots `8g..8g+7`; co-running inference uses the first four, training the next four. `VLA_NODE_CPUS` is the original allocation's ordered CPU list, inherited by subprocesses even after a parent pins itself. `VLA_FULL_SM_COUNT` is queried before MPS; a 30% allocation need not expose the A6000's 24 SMs on an A100. MPS cap is an execution-resource limit, not a promise of exclusive SM ownership.

The frozen environment has both `transformers==4.57.6` metadata and `rlinf-transformer-openpi==4.53.2`; the latter overwrites the import tree. Actual `transformers.__version__` is **4.53.2**, and all its files match the RLinf fork's wheel RECORD. Install the RLinf transformer wheel last, with `--no-deps`. `pip check` may flag metadata conflicts; import/provider identity is additionally checked. No source changes were found against wheel RECORDs for `rlinf-openpi`, `rlinf-libero`, robosuite, mujoco or torch.

The original PyTorch build is 2.11.0+cu130 (CUDA runtime 13.0; driver 580.178.04). There was no `nvcc` executable on PATH. `environment/install.sh` targets that stack in a new Python 3.11 environment. On ALCF, first verify driver/runtime compatibility; do not replace the site's driver. A different site-supported torch/CUDA build is an explicitly changed replication condition, not an identical environment. Record `pip freeze`, CUDA version and Graph parity again. FlashAttention's original URL wheel is retained in the full freeze; it is CUDA/torch-specific and omitted from the portable install because these measured openpi attention paths do not require it.

ALCF examples target **Polaris**, whose nodes have four A100 GPUs and use PBS. They are templates, not a claim that this repository has been executed there. Change project, queue, filesystem names, environment module and walltime for your allocation. Sources: [GPU use](https://docs.alcf.anl.gov/polaris/running-jobs/using-gpus/), [PBS jobs](https://docs.alcf.anl.gov/polaris/running-jobs/), [Python environments](https://docs.alcf.anl.gov/polaris/data-science/python/). No Slurm assumptions are embedded.

Validation on the source host is CPU-only to leave ongoing experiments untouched: parse/compile, provenance and exclusion checks, materialization, mocked allocation/SM checks, command generation, and full-array equality of reconstructed serving inputs. **No new GPU benchmark, MPS daemon or training job was started for packaging.**
