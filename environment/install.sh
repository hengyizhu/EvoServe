#!/usr/bin/env bash
set -euo pipefail
# Run only inside a new Python 3.11 venv; never against the live experiment environment.
: "${VIRTUAL_ENV:?Activate an isolated Python 3.11 venv first}"
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python -m pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
python -m pip install -r "$repo_dir/environment/requirements-portable.txt"
# This fork owns the transformers import path. It MUST be installed last.
python -m pip install --force-reinstall --no-deps 'rlinf-transformer-openpi==4.53.2'
python -c 'import torch,transformers; assert transformers.__version__=="4.53.2"; print(torch.__version__,torch.version.cuda,transformers.__version__)'
