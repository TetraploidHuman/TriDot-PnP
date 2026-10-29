#!/usr/bin/env bash
# The phone app's own detector (B1 port) with a *corrected* three-point depth
# solver.  Same detections as the "b1" row, so any difference is the solver.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=research/.venv/bin/python
OUT=research/results/honest
for subset in test hard_test; do
  echo "=== [$(date -u +%H:%M:%SZ)] app_fixed grid=32 subset=$subset"
  $PY -m research.tridot_temporal.eval_v2 \
    --subset "$subset" --methods app_fixed --b1-grid 32 \
    --device cpu --n-boot 1000 --out "$OUT/app_fixed_$subset"
done
echo "=== done $(date -u +%H:%M:%SZ)"
