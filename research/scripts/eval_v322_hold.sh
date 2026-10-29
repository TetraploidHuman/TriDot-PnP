#!/usr/bin/env bash
# Champion decode on v3.22: peak-only heat-hold (no blind coast).
set -euo pipefail
cd /home/lenovo/TriDotPnP
export PYTHONPATH=/home/lenovo/TriDotPnP CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/lenovo/anaconda3/envs/phi4mm/bin/python
F=research/results/find_v3_22/best.pt
G=research/results/find_geom_v322/best.pt
for pair in "hard_test hard" "test easy"; do
  set -- $pair
  $PY -m research.tridot_temporal.eval_find \
    --find-ckpt "$F" --geom-ckpt "$G" \
    --modes geom --upsample 1 --top-k 5 --temporal-w 10 --vel-w 1.0 \
    --heat-hold --heat-hold-jump 0.12 --heat-hold-frac 0.85 --heat-hold-snap 0.03 \
    --subset "$1" --out "research/results/find_v322_hold_$2" --device cuda
  grep -E '^\| FindNet_geom \|' "research/results/find_v322_hold_$2/REPORT.md"
done
