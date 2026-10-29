# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-18T12:45:11Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@10_all | det@10_cut | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.623 | 0.468 | 0.103 | 0.659 | 0.668 | 237.40 | 267.3 | 99.7 |

det@10 只统计三点都可见的帧。det@10_all 统计全部帧，遮挡点（vis>=1）也要在 10px 内。det@10_cut 只统计发生中断的帧。

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| clutter | 0.336 |
| crosscolor | 0.739 |
| desk | 0.972 |
| far | 0.806 |
| handheld | 0.416 |
| hardmix | 0.272 |
| occlude_hand | 0.300 |
| room | 0.760 |
| screen | 0.440 |

## 分因子 det@10_all（含中断）

| factor | FindNet_geom |
|---|---:|
| clutter | 0.336 |
| crosscolor | 0.750 |
| desk | 0.972 |
| far | 0.806 |
| handheld | 0.328 |
| hardmix | 0.099 |
| occlude_hand | 0.111 |
| room | 0.769 |
| screen | 0.411 |

## 分因子 det@10_cut（仅中断帧）

| factor | FindNet_geom |
|---|---:|
| clutter | nan |
| crosscolor | 1.000 |
| desk | nan |
| far | nan |
| handheld | 0.066 |
| hardmix | 0.048 |
| occlude_hand | 0.111 |
| room | 0.913 |
| screen | 0.038 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| clutter | 338.4 |
| crosscolor | 66.4 |
| desk | 3.4 |
| far | 35.9 |
| handheld | 302.6 |
| hardmix | 401.0 |
| occlude_hand | 436.2 |
| room | 125.8 |
| screen | 298.1 |
