#!/usr/bin/env bash
# Honest-eval batch: learned finder (v3.12 checkpoint) on test + hard_test.
#
# Runs the fixed-decoder, no-gate evaluation of research/tridot_temporal/eval_v2.py
# for the only surviving recent learned checkpoint.  `--temporal-w 0` is pure
# per-frame inference; `--temporal-w 3` reproduces the historical temporal heat
# prior so the two can be compared honestly.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=research/.venv/bin/python
FIND=research/results/find_v3_12_data/best.pt
GEOM=research/results/find_geom_v34/best.pt
OUT=research/results/honest

run() {
  local tag=$1 subset=$2 tw=$3
  echo "=== [$(date -u +%H:%M:%SZ)] $tag subset=$subset temporal_w=$tw" >&2
  "$PY" -m research.tridot_temporal.eval_v2 \
    --methods find_geom --subset "$subset" \
    --find-ckpt "$FIND" --geom-ckpt "$GEOM" \
    --temporal-w "$tw" --device cpu --n-boot 1000 \
    --out "$OUT/$tag"
}

run find_v312_test_w0  test      0
run find_v312_test_w3  test      3
run find_v312_hard_w0  hard_test 0
run find_v312_hard_w3  hard_test 3
echo "=== done $(date -u +%H:%M:%SZ)" >&2
