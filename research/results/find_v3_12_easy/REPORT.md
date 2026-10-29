# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-13T13:51:50Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.925 | 0.966 | 0.969 | 14.48 | 49.3 | 134.6 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| blur | 0.937 |
| clean | 0.987 |
| distractor | 0.846 |
| light | 0.893 |
| mixed | 0.963 |
| noise | 0.941 |
| occ | 0.958 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| blur | 4.3 |
| clean | 3.4 |
| distractor | 12.8 |
| light | 28.0 |
| mixed | 33.2 |
| noise | 4.2 |
| occ | 6.2 |
