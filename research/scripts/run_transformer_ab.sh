#!/usr/bin/env bash
# End-to-end honest comparison: classic B0 vs the v3.12 learned decoder vs the
# new offset-head decoder, on the official splits *and* on the fresh holdout.
#
# Everything is decoded the same way (single-frame, no hand-tuned temporal
# gates); the PnP solver is the fixed one.  Results land in
# research/results/honest/ and are rendered by transformer_ab_report.py.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=research/.venv/bin/python
FIND=research/results/find_v3_12_data/best.pt
GEOM=research/results/find_geom_v34/best.pt
OFFSET=${OFFSET_CKPT:-research/results/find_v3_25_offset/best.pt}
OUT=research/results/honest

run() {
  local subset=$1; shift
  local tag=$1; shift
  echo "=== $tag / $subset ==="
  "$PY" -m research.tridot_temporal.eval_v2 \
    --subset "$subset" --methods "$*" \
    --find-ckpt "$FIND" --geom-ckpt "$GEOM" \
    --snap-radius 14 --n-boot 1000 \
    --out "$OUT/$tag" > "/tmp/tab_${tag}_${subset}.log" 2>&1
}

for subset in test hard_test holdout_easy holdout_hard; do
  run "$subset" "ab_${subset}_b0" b0
  run "$subset" "ab_${subset}_geom" find_geom
done

for subset in test hard_test holdout_easy holdout_hard; do
  echo "=== find_offset / $subset ==="
  "$PY" -m research.tridot_temporal.eval_v2 \
    --subset "$subset" --methods find_offset \
    --find-ckpt "$OFFSET" --snap-radius 14 --n-boot 1000 \
    --out "$OUT/ab_${subset}_offset" > "/tmp/tab_offset_${subset}.log" 2>&1
done

echo "done: $(date -u +%H:%M:%SZ)"
