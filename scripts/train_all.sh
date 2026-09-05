#!/usr/bin/env bash
# Train all 5 models at all 3 sparsity levels (15 runs).
# Usage: scripts/train_all.sh [--epochs N] [--ckpt-dir DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

for sparsity in 32 64 96; do
  for model in baseline se cbam simam simcbam; do
    echo "===== Training $model @ $sparsity angles ====="
    "$PYTHON" -m sparse_pa_unet.train --sparsity "$sparsity" --model "$model" "$@"
  done
done
echo "All training runs complete."