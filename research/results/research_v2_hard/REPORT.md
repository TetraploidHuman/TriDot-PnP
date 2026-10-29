# 研究主线评测：纯学习管线 vs B0（无 Hybrid）

- 时间：2026-10-11T14:28:11Z
- 方法：Propose(B0 blobs) → **TripleRanker** → **CropRefine**（推理不回退 B0 决策）

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.245 | 0.247 | 0.256 | 236.29 | 485.6 | 20.8 |
| Research_TripleRank | 0.326 | 0.333 | 0.346 | 183.47 | 358.1 | 6.2 |
| Research_TripleRefine | 0.326 | 0.333 | 0.346 | 183.45 | 361.0 | 6.0 |

## 分因子 det@10

| factor | B0 | Research_TripleRank | Research_TripleRefine |
|---|---:|---:|---:|
| clutter | 0.075 | 0.242 | 0.242 |
| crosscolor | 0.223 | 0.368 | 0.362 |
| desk | 0.214 | 0.350 | 0.350 |
| far | 0.297 | 0.361 | 0.358 |
| handheld | 0.245 | 0.230 | 0.230 |
| hardmix | 0.000 | 0.049 | 0.049 |
| occlude_hand | 0.222 | 0.225 | 0.225 |
| room | 0.780 | 0.774 | 0.774 |
| screen | 0.003 | 0.072 | 0.078 |

## 分因子 pix

| factor | B0 | Research_TripleRank | Research_TripleRefine |
|---|---:|---:|---:|
| clutter | 297.2 | 193.2 | 193.1 |
| crosscolor | 229.7 | 135.0 | 135.0 |
| desk | 193.6 | 123.4 | 123.4 |
| far | 193.8 | 156.5 | 156.4 |
| handheld | 186.2 | 170.4 | 170.5 |
| hardmix | 337.0 | 305.4 | 305.2 |
| occlude_hand | 162.9 | 154.7 | 155.0 |
| room | 16.3 | 19.6 | 19.7 |
| screen | 344.1 | 248.8 | 248.5 |
