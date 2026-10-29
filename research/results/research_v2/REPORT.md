# 研究主线评测：纯学习管线 vs B0（无 Hybrid）

- 时间：2026-10-11T02:30:03Z
- 方法：Propose(B0 blobs) → **TripleRanker** → **CropRefine**（推理不回退 B0 决策）

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.918 | 0.918 | 0.920 | 35.00 | 57.4 | 81.9 |
| Research_TripleRank | 0.974 | 0.974 | 0.977 | 18.77 | 28.1 | 70.2 |
| Research_TripleRefine | 0.974 | 0.974 | 0.977 | 18.41 | 23.7 | 41.7 |

## 分因子 det@10

| factor | B0 | Research_TripleRank | Research_TripleRefine |
|---|---:|---:|---:|
| blur | 1.000 | 1.000 | 1.000 |
| clean | 1.000 | 1.000 | 1.000 |
| distractor | 0.679 | 0.902 | 0.902 |
| light | 1.000 | 1.000 | 1.000 |
| mixed | 0.926 | 0.917 | 0.917 |
| noise | 0.920 | 1.000 | 1.000 |
| occ | 0.025 | 0.025 | 0.025 |

## 分因子 pix

| factor | B0 | Research_TripleRank | Research_TripleRefine |
|---|---:|---:|---:|
| blur | 0.8 | 0.8 | 0.4 |
| clean | 0.9 | 0.9 | 0.4 |
| distractor | 57.3 | 15.1 | 14.7 |
| light | 0.9 | 0.9 | 0.6 |
| mixed | 115.3 | 73.7 | 73.5 |
| noise | 0.9 | 0.9 | 0.5 |
| occ | 1.2 | 1.2 | 0.5 |
