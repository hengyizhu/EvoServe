#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SUITE_HOME=${SUITE_HOME:-"$PWD/.h100"}
./setup.sh
export PYTHONPATH="$PWD/prototype:$PWD/vendor/RLinf${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${SUITE_PYTHON:-$SUITE_HOME/venv/bin/python}" h100/campaign.py "$@"
