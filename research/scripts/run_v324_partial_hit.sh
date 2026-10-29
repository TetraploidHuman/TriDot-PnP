#!/usr/bin/env bash
# v3.24 from v3.22. Peak-margin includes partial LEDs (vis=0.5).
# xy weight on partial stays 0.5 (v3.23 full weight hurt easy/hard, TopK flat).
# Backbone unfrozen at a small lr. Keep a checkpoint only if full-vis l2
# stays within 4% and partial Top1 hit improves.
set -euo pipefail
cd /home/lenovo/TriDotPnP
export PYTHONPATH=/home/lenovo/TriDotPnP CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/lenovo/anaconda3/envs/phi4mm/bin/python
LOG=logs/v324_partial_hit.log
mkdir -p logs research/results
exec >>"$LOG" 2>&1
echo "=== v3.24 partial-hit from v322 $(date) ==="

$PY -m research.tridot_temporal.train_find \
  --resume research/results/find_v3_22/best.pt --reset-best \
  --out research/results/find_v3_24 \
  --epochs 4 --batch-size 16 --device cuda --lr 8e-6 \
  --train-subset train,hard_train --val-subset val,hard_val \
  --max-clips 10 --xy-boost-after 999 \
  --lambda-xy 8 --lambda-xy0 20 --lambda-hm 10 --lambda-temp 1.5 \
  --lambda-peak 0.8 --peak-margin 0.25 --peak-exclude 8 \
  --hm-sigma 1.4 \
  --lambda-sep 2 --lambda-eq 0.15 \
  --cache-videos --num-workers 8 --lru-max 512 --prefetch-factor 4 --amp \
  --d-model 160 --nlayers 3 --nhead 4 --ffn-dim 320 --feat-ch 96

$PY -m research.tridot_temporal.train_find_geom \
  --find-ckpt research/results/find_v3_24/best.pt \
  --out research/results/find_geom_v324 \
  --epochs 4 --batch-size 16 --device cuda --upsample 1 --top-k 5 \
  --train-subset train,hard_train --val-subset val,hard_val --max-clips 8 \
  --cache-videos --num-workers 8 --lru-max 512

for pair in "test easy" "hard_test hard"; do
  set -- $pair
  $PY -m research.tridot_temporal.eval_find \
    --find-ckpt research/results/find_v3_24/best.pt \
    --geom-ckpt research/results/find_geom_v324/best.pt \
    --modes geom --upsample 1 --top-k 5 --temporal-w 10 --vel-w 1.0 \
    --subset "$1" --out "research/results/find_v324_${2}_cut" --device cuda
  echo "=== $2 ==="
  grep -E '^\| FindNet_geom \|' "research/results/find_v324_${2}_cut/REPORT.md" | head -1
done
echo "=== v3.24 DONE $(date) ==="
