# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-13T16:26:21Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.768 | 0.958 | 0.980 | 12.45 | 49.4 | 116.3 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| blur | 0.661 |
| clean | 0.903 |
| distractor | 0.833 |
| light | 0.806 |
| mixed | 0.435 |
| noise | 0.786 |
| occ | 0.929 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| blur | 7.2 |
| clean | 4.2 |
| distractor | 9.9 |
| light | 26.0 |
| mixed | 23.9 |
| noise | 5.8 |
| occ | 4.4 |
