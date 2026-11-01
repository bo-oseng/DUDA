#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
: "${STAGE:?Set STAGE to 1-3}"
export CONTINUAL_STREAM=6p1p1p1
export CONTINUAL_STAGE="${STAGE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
python scripts/run.py evaluate "$@"
