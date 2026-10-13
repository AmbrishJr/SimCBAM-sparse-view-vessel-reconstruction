#!/usr/bin/env bash
# Evaluate all models at all sparsity levels (or a subset via args).
# Usage: scripts/evaluate_all.sh [--sparsity 64] [--models baseline,cbam]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

"$PYTHON" -m sparse_pa_unet.evaluate "$@"