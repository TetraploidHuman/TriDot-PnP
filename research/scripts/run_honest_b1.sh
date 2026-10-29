#!/usr/bin/env bash
# Evaluate the phone app's own pipeline (B1 port: soft-HSV grid detector +
# FOV-estimated PnP) on the same honest harness as B0 / FindNet.
# grid=32 matches the app's 32x24 probability matrix (port default 256 is
# ~1.3 s/frame on CPU and not what the phone runs).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=research/.venv/bin/python
OUT=research/results/honest
for subset in test hard_test; do
  echo "=== [$(date -u +%H:%M:%SZ)] b1 grid=32 subset=$subset"
  $PY -m research.tridot_temporal.eval_v2 \
    --subset "$subset" --methods b1 --b1-grid 32 --b1-fov 60 \
    --device cpu --n-boot 1000 --out "$OUT/b1_$subset"
done
echo "=== done $(date -u +%H:%M:%SZ)"
