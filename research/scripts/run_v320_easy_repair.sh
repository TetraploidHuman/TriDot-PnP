#!/usr/bin/env bash
# Repair easy blur/noise without dropping v3.19b hard gains.
# Soft factors: wide heatmap, no peak-margin. Hard factors: keep sharp peak loss.
set -euo pipefail
cd /home/lenovo/TriDotPnP
export PYTHONPATH=/home/lenovo/TriDotPnP CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/lenovo/anaconda3/envs/phi4mm/bin/python
LOG=logs/v320_easy_repair.log
mkdir -p logs research/results
exec >>"$LOG" 2>&1
echo "=== v3.20 easy repair $(date) ==="

$PY -m research.tridot_temporal.train_find \
  --resume research/results/find_v3_19b/best.pt --reset-best \
  --out research/results/find_v3_20 \
  --epochs 8 --batch-size 16 --device cuda --lr 3e-5 \
  --train-subset train,hard_train --val-subset val,hard_val \
  --max-clips 10 --xy-boost-after 999 \
  --lambda-xy 6 --lambda-xy0 22 --lambda-hm 12 \
  --lambda-peak 4 --peak-margin 0.6 \
  --hm-sigma 1.15 --hm-sigma-soft 1.4 \
  --soft-factors blur,noise,clean,light \
  --lambda-sep 2 --lambda-temp 0 --lambda-eq 0.15 \
  --cache-videos --num-workers 8 --lru-max 512 --prefetch-factor 4 --amp \
  --freeze-temporal \
  --d-model 160 --nlayers 3 --nhead 4 --ffn-dim 320 --feat-ch 96

$PY -m research.tridot_temporal.train_find_geom \
  --find-ckpt research/results/find_v3_20/best.pt \
  --out research/results/find_geom_v320 \
  --epochs 6 --batch-size 16 --device cuda --upsample 1 --top-k 5 \
  --train-subset train,hard_train --val-subset val,hard_val --max-clips 8 \
  --cache-videos --num-workers 8 --lru-max 512

eval_one() {
  local subset="$1" tag="$2"
  $PY -m research.tridot_temporal.eval_find \
    --find-ckpt research/results/find_v3_20/best.pt \
    --geom-ckpt research/results/find_geom_v320/best.pt \
    --modes geom --upsample 1 --top-k 5 --temporal-w 10 --vel-w 1.0 \
    --subset "$subset" --out "research/results/find_v3_20_${tag}" --device cuda
  echo "=== $tag ==="
  grep -E '^\| FindNet_geom \|' "research/results/find_v3_20_${tag}/REPORT.md" | head -1
  sed -n '/分因子 det@10/,/分因子 pix/p' "research/results/find_v3_20_${tag}/REPORT.md"
}
eval_one test easy
eval_one hard_test hard
echo "=== v3.20 DONE $(date) ==="
