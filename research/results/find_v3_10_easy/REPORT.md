# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-12T16:31:05Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.863 | 0.969 | 0.989 | 9.83 | 35.7 | 120.3 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| blur | 0.751 |
| clean | 0.983 |
| distractor | 0.872 |
| light | 0.917 |
| mixed | 0.537 |
| noise | 0.937 |
| occ | 0.917 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| blur | 6.3 |
| clean | 3.7 |
| distractor | 10.0 |
| light | 19.2 |
| mixed | 15.9 |
| noise | 4.3 |
| occ | 6.5 |
