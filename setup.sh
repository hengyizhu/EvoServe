#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=${SUITE_HOME:-"$PWD/.h100"}
mkdir -p "$root"
nvidia-smi -L
command -v nvidia-cuda-mps-control >/dev/null || { echo 'CUDA MPS executable missing: MPS cases will report failure.' >&2; }
# Probe the control plane in a private namespace. The co-run also checks both CUDA clients.
if command -v nvidia-cuda-mps-control >/dev/null; then
    (
        mps_probe=$(mktemp -d)
        mkdir "$mps_probe/pipe" "$mps_probe/log"
        export CUDA_MPS_PIPE_DIRECTORY="$mps_probe/pipe" CUDA_MPS_LOG_DIRECTORY="$mps_probe/log"
        trap 'rm -rf -- "$mps_probe"' EXIT
        nvidia-cuda-mps-control -d || exit 1
        trap 'echo quit | timeout 15 nvidia-cuda-mps-control >/dev/null; rm -rf -- "$mps_probe"' EXIT
        echo get_server_list | timeout 15 nvidia-cuda-mps-control >/dev/null || exit 1
        echo 'Private MPS control daemon responded.'
    ) || echo 'MPS control probe failed; individual MPS cases will retry and report their status.' >&2
fi
if [[ -z "${SUITE_PYTHON:-}" ]]; then
    if [[ $(id -u) == 0 ]]; then
        apt-get update -qq
        apt-get install -y python3-venv build-essential libgl1 libglib2.0-0 libegl1 libosmesa6 libnuma1 git curl pkg-config cmake
    fi
    if ! command -v uv >/dev/null && [[ ! -x "$root/bootstrap/bin/uv" ]]; then
        python3 -m venv "$root/bootstrap"
        "$root/bootstrap/bin/pip" install 'uv==0.9.11'
    fi
    uvbin=$(command -v uv || echo "$root/bootstrap/bin/uv")
    "$uvbin" python install 3.11.14
    if [[ ! -x "$root/venv/bin/python" ]]; then "$uvbin" venv --seed --python 3.11.14 "$root/venv"; fi
    py="$root/venv/bin/python"
    if [[ ! -f "$root/environment.ok" ]]; then
        "$py" -m pip install --no-deps -r machine_b/environment/requirements-lock.txt
        "$py" -m pip install --no-deps 'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' --index-url https://download.pytorch.org/whl/cu130
        "$py" -m pip install --no-deps 'https://github.com/RLinf/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
        "$py" -m pip install --no-deps --force-reinstall 'rlinf-transformer-openpi==4.53.2'
        "$py" -c 'import torch,flash_attn; assert torch.version.cuda.startswith("13."); print(torch.__version__)'
        touch "$root/environment.ok"
    fi
else
    py="$SUITE_PYTHON"
fi
if [[ ! -e "$root/assets.ok" ]]; then
    extra=()
    if [[ -n "${SUITE_LOCAL_DATA:-}" ]]; then extra+=(--local-data "$SUITE_LOCAL_DATA"); fi
    "$py" h100/assets.py "$root" "${extra[@]}"
    touch "$root/assets.ok"
fi
