#!/usr/bin/env bash
# Fine-tune peak association + heat writeback on top of an existing FindNet ckpt.
# Usage:
#   bash research/scripts/run_peak_assoc_ft.sh /path/to/best.pt [out_dir] [device]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT="${1:?need find ckpt}"
OUT="${2:-$ROOT/research/results/find_v3_peak}"
DEV="${3:-cuda}"
export PYTHONPATH="$ROOT"
PY="${PY:-python}"

"$PY" -m research.tridot_temporal.train_find \
  --resume "$CKPT" --partial-init --reset-best \
  --out "$OUT" --device "$DEV" \
  --peak-assoc --peak-k 5 --heat-writeback --writeback-w 1.5 \
  --freeze-backbone \
  --frame-drop-p 0.35 --lambda-temp 1.0 --lambda-pick 2.0 \
  --lambda-xy 8 --lambda-xy0 4 --lambda-hm 6 --lambda-peak 1.0 \
  --epochs 20 --batch-size 4 --amp --cache-videos \
  --train-subset hard_train,train --val-subset hard_val,val

echo "[done] peak FT → $OUT/best.pt"
echo "eval: $PY -m research.tridot_temporal.eval_find --find-ckpt $OUT/best.pt --modes peak,vel,coords,geom --subset hard_test --device $DEV"
