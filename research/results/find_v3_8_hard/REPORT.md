# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-12T04:23:20Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.245 | 0.247 | 0.256 | 236.29 | 485.6 | 20.7 |
| FindNet_geom | 0.787 | 0.918 | 0.951 | 32.23 | 71.9 | 143.8 |

## 分因子 det@10

| factor | B0 | FindNet_geom |
|---|---:|---:|
| clutter | 0.075 | 0.725 |
| crosscolor | 0.223 | 0.881 |
| desk | 0.214 | 0.917 |
| far | 0.297 | 0.861 |
| handheld | 0.245 | 0.725 |
| hardmix | 0.000 | 0.395 |
| occlude_hand | 0.222 | 0.653 |
| room | 0.780 | 0.985 |
| screen | 0.003 | 0.575 |

## 分因子 pix

| factor | B0 | FindNet_geom |
|---|---:|---:|
| clutter | 297.2 | 13.6 |
| crosscolor | 229.7 | 8.1 |
| desk | 193.6 | 4.5 |
| far | 193.8 | 4.8 |
| handheld | 186.2 | 72.9 |
| hardmix | 337.0 | 67.7 |
| occlude_hand | 162.9 | 43.2 |
| room | 16.3 | 3.4 |
| screen | 344.1 | 42.1 |
