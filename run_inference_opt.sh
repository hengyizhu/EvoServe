#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SUITE_HOME=${SUITE_HOME:-"$PWD/.h100"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)}
./setup.sh
export PYTHONPATH="$PWD/prototype:$PWD/vendor/RLinf${PYTHONPATH:+:$PYTHONPATH}"
export EVOSERVE_CPU_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Setup errors are global; the Python runner isolates candidate CUDA faults in
# child processes and can start a fresh reference process after a failed replay.
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-2}
exec "${SUITE_PYTHON:-$SUITE_HOME/venv/bin/python}" h100/inference_opt.py "$@"
