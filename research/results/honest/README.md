# Honest evaluation — TriDotPose

Produced by `research/tridot_temporal/eval_v2.py`; metrics defined in
`research/tridot_temporal/metrics.py`. Every number is attributable to a
checkpoint hash + split hash recorded in the per-run `summary.json`.

Definitions that differ from the historical `eval_find.py` reports:

- **det@10 (all frames)**: denominator is *every* frame; an occluded frame counts
  as a miss because the product must output three points. The historical number
  only counted frames where all three GT dots were visible (``legacy`` column).
- **pose coverage**: fraction of frames where PnP returned a pose.
- **t_err (clean)**: translation error on frames with three GT-visible dots and a
  successful PnP; median / p90 reported. Mean is dominated by PnP flip outliers.
- **jitter**: median |Δt|×fps between consecutive frames that produced a pose.
- **point recall@10**: per-dot recall over GT-visible dots (main three).
- **b1**: Python port of the Android app detector (HSV grid + probability
  matrix) with the *shipped* iterative depth solver, i.e. the app as released.
- **app_fixed**: same app detector, but driving the fixed Newton/LM depth solver —
  what the app scores once only the PnP math is corrected.

## test

| method | det@10 all | det@10 legacy | recall@10 (pt) | pose cov | t_err med | t_err p90 | jitter mm/s | latency ms | est FPS |
|---|---|---|---|---|---|---|---|---|---|
| b0 | 0.634 | 0.918 | 0.790 | 0.800 | 3.6 | 114.4 | 722 | 8 | 117.8 |
| b1 | 0.421 | 0.609 | 0.564 | 0.616 | 450.6 | 873.0 | 443 | 83 | 12.1 |
| app_fixed | 0.421 | 0.609 | 0.564 | 0.616 | 7.6 | 726.0 | 858 | 79 | 12.6 |
| find_geom(w0) | 0.606 | 0.877 | 0.916 | 0.999 | 26.0 | 108.3 | 1168 | 66 | 15.1 |
| find_geom(w3) | 0.607 | 0.879 | 0.920 | 0.999 | 25.9 | 107.6 | 1168 | 66 | 15.1 |

## hard_test

| method | det@10 all | det@10 legacy | recall@10 (pt) | pose cov | t_err med | t_err p90 | jitter mm/s | latency ms | est FPS |
|---|---|---|---|---|---|---|---|---|---|
| b0 | 0.172 | 0.245 | 0.346 | 0.878 | 419.8 | 1089.8 | 3593 | 45 | 22.2 |
| b1 | 0.149 | 0.212 | 0.380 | 0.714 | 513.3 | 1023.7 | 2416 | 77 | 13.0 |
| app_fixed | 0.149 | 0.212 | 0.380 | 0.714 | 80.4 | 1025.2 | 2465 | 79 | 12.7 |
| find_geom(w0) | 0.588 | 0.837 | 0.889 | 0.991 | 25.4 | 144.0 | 1255 | 66 | 15.1 |
| find_geom(w3) | 0.590 | 0.841 | 0.894 | 0.991 | 24.9 | 140.9 | 1246 | 62 | 16.0 |

| factor | b0 | b1 | app_fixed | find_geom(w0) | find_geom(w3) |
|---|---|---|---|---|---|
| clutter | 0.075 | 0.197 | 0.197 | 0.808 | 0.819 |
| crosscolor | 0.214 | 0.311 | 0.311 | 0.892 | 0.894 |
| desk | 0.214 | 0.147 | 0.147 | 0.914 | 0.922 |
| far | 0.297 | 0.011 | 0.011 | 0.897 | 0.897 |
| handheld | 0.183 | 0.369 | 0.369 | 0.572 | 0.572 |
| hardmix | 0.000 | 0.000 | 0.000 | 0.075 | 0.076 |
| occlude_hand | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| room | 0.731 | 0.453 | 0.453 | 0.917 | 0.917 |
| screen | 0.003 | 0.000 | 0.000 | 0.725 | 0.725 |

## Provenance

- `app_fixed_hard_test`: subset=`hard_test` method=`app_fixed` find_ckpt=None (-) geom_ckpt=None splits=bd40d8b6129e seed=0 torch=2.14.1+cpu
- `b0_hard_test`: subset=`hard_test` method=`b0` find_ckpt=None (-) geom_ckpt=None splits=bd40d8b6129e seed=0 torch=2.14.1+cpu
- `b1_hard_test`: subset=`hard_test` method=`b1` find_ckpt=None (-) geom_ckpt=None splits=bd40d8b6129e seed=0 torch=2.14.1+cpu
- `find_v312_hard_w0`: subset=`hard_test` method=`find_geom(w0)` find_ckpt=/home/miaox99/TriDotPnP/research/results/find_v3_12_data/best.pt (be24f182bfc0) geom_ckpt=/home/miaox99/TriDotPnP/research/results/find_geom_v34/best.pt splits=bd40d8b6129e seed=0 torch=2.14.1+cpu
- `find_v312_hard_w3`: subset=`hard_test` method=`find_geom(w3)` find_ckpt=/home/miaox99/TriDotPnP/research/results/find_v3_12_data/best.pt (be24f182bfc0) geom_ckpt=/home/miaox99/TriDotPnP/research/results/find_geom_v34/best.pt splits=bd40d8b6129e seed=0 torch=2.14.1+cpu
- `app_fixed_test`: subset=`test` method=`app_fixed` find_ckpt=None (-) geom_ckpt=None splits=bd40d8b6129e seed=0 torch=2.14.1+cpu
- `b0_test`: subset=`test` method=`b0` find_ckpt=None (-) geom_ckpt=None splits=bd40d8b6129e seed=0 torch=2.14.1+cpu
- `b1_test`: subset=`test` method=`b1` find_ckpt=None (-) geom_ckpt=None splits=bd40d8b6129e seed=0 torch=2.14.1+cpu
- `find_v312_test_w0`: subset=`test` method=`find_geom(w0)` find_ckpt=/home/miaox99/TriDotPnP/research/results/find_v3_12_data/best.pt (be24f182bfc0) geom_ckpt=/home/miaox99/TriDotPnP/research/results/find_geom_v34/best.pt splits=bd40d8b6129e seed=0 torch=2.14.1+cpu
- `find_v312_test_w3`: subset=`test` method=`find_geom(w3)` find_ckpt=/home/miaox99/TriDotPnP/research/results/find_v3_12_data/best.pt (be24f182bfc0) geom_ckpt=/home/miaox99/TriDotPnP/research/results/find_geom_v34/best.pt splits=bd40d8b6129e seed=0 torch=2.14.1+cpu
