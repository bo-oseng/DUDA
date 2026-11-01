#!/usr/bin/env bash
# Evaluate the released DUDA epoch-10 teacher; override CHECKPOINT for your own run.
# Example: CHECKPOINT=/path/to/stage1_latest.pth bash bash/test.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

python scripts/run.py evaluate "$@"
