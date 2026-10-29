# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-12T12:28:38Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.921 | 0.967 | 0.971 | 14.12 | 63.5 | 128.1 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| blur | 0.982 |
| clean | 0.966 |
| distractor | 0.846 |
| light | 0.908 |
| mixed | 0.963 |
| noise | 0.887 |
| occ | 0.967 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| blur | 3.7 |
| clean | 4.1 |
| distractor | 13.2 |
| light | 28.0 |
| mixed | 30.9 |
| noise | 4.3 |
| occ | 6.3 |
