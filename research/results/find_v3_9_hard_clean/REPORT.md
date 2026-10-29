# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-12T12:27:37Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.797 | 0.915 | 0.948 | 29.25 | 72.4 | 131.8 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| clutter | 0.753 |
| crosscolor | 0.893 |
| desk | 0.942 |
| far | 0.872 |
| handheld | 0.706 |
| hardmix | 0.414 |
| occlude_hand | 0.675 |
| room | 0.988 |
| screen | 0.575 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| clutter | 14.9 |
| crosscolor | 5.6 |
| desk | 4.0 |
| far | 4.4 |
| handheld | 62.7 |
| hardmix | 65.5 |
| occlude_hand | 43.3 |
| room | 3.0 |
| screen | 29.5 |
