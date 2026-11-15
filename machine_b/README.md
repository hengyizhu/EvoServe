# Machine B: two A6000s, learning only

This branch exports the current working software stack, patched π0.5/RLinf source and LIBERO perturbation code. It does not run latency experiments, start MPS, set power limits, change the driver or modify system Python. No weights, datasets, prefix caches, checkpoints or raw results are committed.

## Install

Use Linux x86_64, Python **3.11.14**, an NVIDIA driver supporting the pinned CUDA13.0 PyTorch wheel, and NVIDIA EGL for headless LIBERO. Source machine driver is580.178.04. The source machine currently has225W limits on all four GPUs; this export records that setting but never applies it to Machine B. Record Machine B's actual limits in its own outputs. Provision approximately40GB for the complete environment/assets plus dataset space, and at least100GB scratch for task caches. Host RAM requirements grow with unique sampled windows; run the initial reproduction on one GPU before starting two cached trainers.

```bash
git clone --branch machine-b-setup git@github.com:hengyizhu/EvoServe.git
cd EvoServe
export MACHINE_B_ROOT=/your/scratch/evoserve-machine-b
export PYTHON311=/path/to/python3.11  # exactly3.11.14
bash machine_b/install.sh
```

If Python3.11.14 is not available, obtain that exact interpreter with the site's Python manager (e.g. `uv python install 3.11.14`) and set `PYTHON311`. The script refuses an existing venv. It installs all305 source distribution versions with dependency resolution disabled, installs the exact CUDA wheels and flash-attention wheel, and installs the RLinf transformer overlay last. It then verifies versions, downloads pinned assets, materializes relocatable scripts and discovers this node's EGL mapping.

The source environment has **six existing dependency-metadata conflicts**, listed verbatim in `environment/pip-check-source.txt`: RLinf/LeRobot's older torch constraints, torchcodec/torchvision constraints, LIBERO's older MuJoCo constraint, and the overlaid transformers/tokenizers metadata. These are not silently resolved by upgrading packages. Effective `transformers.__version__` must be4.53.2 even though the `transformers` distribution metadata is4.57.6. `verify_environment.py` rejects unexpected new conflicts or changed package versions. `environment/pip-freeze-full.txt` is the complete original freeze, including pip/setuptools and both installed CUDA runtime package families; `manifest.json` records effective imports and direct wheel URLs.

## Assets and task code

|Asset|Pinned source|
|---|---|
|Parent|`RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT`, revision`6222623f635769bfc73c9472e29fab9b7fd8e027`|
|Weight SHA256|`4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683`|
|LeRobot demonstrations|`physical-intelligence/libero`, revision`9dfa69510ea9e1613fc54112bc706444b686a231`|
|Physical-scene demonstrations|`yifengzhu-hf/LIBERO-datasets`, revision`f13aa24a3da8c43c7225569f28c562979fa0e35a`; individual HDF5 SHA256s in`official_data_source.json`|
|Tokenizer|Pinned URL and SHA256 in`../provenance/assets.json`|
|LIBERO/robosuite/MuJoCo|`rlinf-libero==0.1.3` (includes assets/BDDL/init states),`robosuite==1.4.1`,`mujoco==3.8.1`|
|RLinf / openpi|Vendored RLinf commit`91a302ec50ddacd1fdabc5859b85e3661d768236` plus included patches;`rlinf-openpi==0.1.1`|

`tools/download_assets.py` verifies model/tokenizer hashes and downloads the needed original demonstration episodes at pinned revisions. `experiments/pi05_extended_20261109/tasks/` supplies lowlight drawer, shifted book and microwave conditions. `experiments/pi05_machine_b_20261115/tasks/` supplies X lowlight, Y25% image shift, actual camera orbit and texture perturbations. `perturbations.py` transforms both observation cameras for image shifts; `scene_perturb.py` modifies the actual simulator for camera/texture cases. Original train/validation splits and sampling sequences are included.

Camera and texture tasks require re-rendered observations with matching native actions. After installation, prepare them separately:

```bash
PY="$MACHINE_B_ROOT/venv/bin/python"
W="$MACHINE_B_ROOT/work"
"$PY" machine_b/render_scene_data.py --work "$W" --task camera --gpu 0
"$PY" machine_b/render_scene_data.py --work "$W" --task texture --gpu 1
```

The renderer downloads and hashes the official HDF5, restores each demonstration XML, applies the same perturbation as evaluation and verifies`states[i+1]` corresponds to original`obs[i]` by exact joint positions. It writes actual-camera/texture datasets and splits into`$W/pi05_machine_b_20261115/`; original-history files are symlinked from the downloaded data. Do not substitute a 2D shift for the physical camera orbit.

## Reproduction check: book, seed08,1000 updates,50 episodes

```bash
"$MACHINE_B_ROOT/venv/bin/python" machine_b/reproduce_book.py \
  --work "$MACHINE_B_ROOT/work" --gpu 0 --threads 2
# Optional: --cpus 16,17 to use an explicitly allocated small CPU set.
# Print all commands without model loading/training: add --dry-run.
```

The script first runs **one optimizer update and one full evaluation episode**, then1000 updates and50 fixed-state evaluation episodes. Action LoRA r32 (20,876,320 trainable parameters), dense action input/output/time heads, frozen vision, no augmentation, lr5e-5, EMA0.999, microbatch4×accumulation2. Task5 is shifted right25% in both cameras; evaluation uses5denoising steps and action chunk5.

**The reference is7/50 (14%) at update1000 of a4000-step cosine schedule.** The script uses exactly the first1000 batches of that4000-step sample sequence,50-step warmup and a4000-step cosine horizon. Replacing this with a1000-step cosine would be a different experiment. `book_reference.json` captures the reference. The historical reference's power limit was not independently recorded;225W is the current export-time setting, not a retroactive claim.

A failed execution retries once using a separate run name. The summary reports successes, difference from reference, actual power limits, CPU affinity and config. It does not invent a quality acceptance threshold or claim cross-machine bitwise equivalence. Output:`$W/pi05_machine_b_20261115/machine_b_book_s08_1000_summary.json`, EMA checkpoints under`training/`, episode logs under`results/`.

The CPU policy for this learning check permits a small shared allocation rather than assuming8free cores per GPU. Use`--cpus` or launch under the cluster's cpuset; the script refuses CPU IDs outside that allocation. CPU contention may lengthen the experiment, so compare learning outputs, not timings. No Machine B execution has been claimed from source-host CPU-only validation.
