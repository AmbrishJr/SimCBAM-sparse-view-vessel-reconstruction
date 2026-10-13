#!/usr/bin/env bash
# Symlink local data + checkpoints from ~/Downloads into ./data and ./checkpoints.
# Usage: scripts/link_local_data.sh [DATA_SRC] [CKPT_SRC]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_SRC="${1:-$HOME/Downloads}"
CKPT_SRC="${2:-$HOME/Downloads/all_model_checkpoints}"

echo "Linking dataset from: $DATA_SRC"
for d in Vessel Vessel_sparse32 Vessel_sparse64 Vessel_sparse96; do
  if [ -d "$DATA_SRC/$d" ]; then
    ln -sfn "$DATA_SRC/$d" "$ROOT/data/$d"
    echo "  linked data/$d"
  else
    echo "  [warn] $DATA_SRC/$d not found"
  fi
done

echo "Linking checkpoints from: $CKPT_SRC"
if [ -d "$CKPT_SRC" ]; then
  ln -sfn "$CKPT_SRC" "$ROOT/checkpoints/all_model_checkpoints"
  cp "$CKPT_SRC"/sparsity_classifier.pth "$ROOT/checkpoints"/sparsity_classifier.pth 2>/dev/null || true
  for f in "$CKPT_SRC"/*_unet_sparse*.pth; do
    ln -sfn "$f" "$ROOT/checkpoints/$(basename "$f")"
  done
  echo "  linked checkpoints"
else
  echo "  [warn] $CKPT_SRC not found"
fi

echo "Done. Set DATA_ROOT/CKPT_ROOT to ./data ./checkpoints or use config defaults."