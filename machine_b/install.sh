#!/usr/bin/env bash
set -euo pipefail
# Isolated install only. Never changes GPU power limits, driver, or system Python.
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${MACHINE_B_ROOT:?Set MACHINE_B_ROOT to an empty experiment directory}"
python_bin=${PYTHON311:-python3.11}
"$python_bin" -c 'import platform; assert platform.python_version()=="3.11.14", "Use Python3.11.14 (PYTHON311=/path/to/python)"'
mkdir -p "$MACHINE_B_ROOT"
venv_path="$MACHINE_B_ROOT/venv"
if test -e "$venv_path"; then echo 'Refusing to overwrite an existing environment'; exit 1; fi
"$python_bin" -m venv "$venv_path"
py="$venv_path/bin/python"
"$py" -m pip install --no-deps -r "$repo_dir/machine_b/environment/requirements-lock.txt"
"$py" -m pip install --no-deps 'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' --index-url https://download.pytorch.org/whl/cu130
"$py" -m pip install --no-deps 'flash_attn @ https://github.com/RLinf/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
# Both distributions exist on the source host; the RLinf fork must own the import files.
"$py" -m pip install --force-reinstall --no-deps 'rlinf-transformer-openpi==4.53.2'
"$py" "$repo_dir/machine_b/verify_environment.py" --out "$MACHINE_B_ROOT/environment-check.json"
"$py" "$repo_dir/tools/init_workspace.py" --work "$MACHINE_B_ROOT/work" --model "$MACHINE_B_ROOT/model" --data "$MACHINE_B_ROOT/data"
"$py" "$repo_dir/tools/download_assets.py" --work "$MACHINE_B_ROOT/work"
"$py" "$repo_dir/tools/build_inputs.py" --work "$MACHINE_B_ROOT/work"
"$py" "$repo_dir/tools/configure_node.py" --work "$MACHINE_B_ROOT/work"
printf '\nInstalled. Run: %s %s/machine_b/reproduce_book.py --work %s/work --gpu 0\n' "$py" "$repo_dir" "$MACHINE_B_ROOT"
