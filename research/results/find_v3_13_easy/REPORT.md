# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-13T14:37:06Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.790 | 0.947 | 0.993 | 8.90 | 43.0 | 113.2 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| blur | 0.597 |
| clean | 0.975 |
| distractor | 0.825 |
| light | 0.893 |
| mixed | 0.389 |
| noise | 0.845 |
| occ | 0.904 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| blur | 8.7 |
| clean | 4.3 |
| distractor | 10.2 |
| light | 6.8 |
| mixed | 17.6 |
| noise | 5.3 |
| occ | 5.1 |
