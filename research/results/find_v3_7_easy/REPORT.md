# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-11T13:26:33Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.897 | 0.958 | 0.965 | 17.01 | 61.5 | 142.9 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| blur | 0.955 |
| clean | 0.970 |
| distractor | 0.816 |
| light | 0.820 |
| mixed | 0.880 |
| noise | 0.924 |
| occ | 0.938 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| blur | 4.1 |
| clean | 4.3 |
| distractor | 18.8 |
| light | 23.1 |
| mixed | 44.6 |
| noise | 4.9 |
| occ | 5.5 |
