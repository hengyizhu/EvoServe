#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SUITE_HOME=${SUITE_HOME:-"$PWD/.h100"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)}
./setup.sh
export PYTHONPATH="$PWD/prototype:$PWD/vendor/RLinf${PYTHONPATH:+:$PYTHONPATH}"
export EVOSERVE_CPU_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${SUITE_PYTHON:-$SUITE_HOME/venv/bin/python}" h100/inference_opt.py "$@"
