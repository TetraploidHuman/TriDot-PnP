# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-28T03:27:04Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@10_all | det@10_cut | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.245 | 0.175 | 0.010 | 0.247 | 0.256 | 236.29 | 485.6 | 11.4 |

det@10 只统计三点都可见的帧。det@10_all 统计全部帧，遮挡点（vis>=1）也要在 10px 内。det@10_cut 只统计发生中断的帧。

## 分因子 det@10

| factor | B0 |
|---|---:|
| clutter | 0.075 |
| crosscolor | 0.223 |
| desk | 0.214 |
| far | 0.297 |
| handheld | 0.245 |
| hardmix | 0.000 |
| occlude_hand | 0.222 |
| room | 0.780 |
| screen | 0.003 |

## 分因子 det@10_all（含中断）

| factor | B0 |
|---|---:|
| clutter | 0.075 |
| crosscolor | 0.214 |
| desk | 0.214 |
| far | 0.297 |
| handheld | 0.211 |
| hardmix | 0.000 |
| occlude_hand | 0.003 |
| room | 0.731 |
| screen | 0.003 |

## 分因子 det@10_cut（仅中断帧）

| factor | B0 |
|---|---:|
| clutter | nan |
| crosscolor | 0.000 |
| desk | nan |
| far | nan |
| handheld | 0.110 |
| hardmix | 0.000 |
| occlude_hand | 0.003 |
| room | 0.000 |
| screen | 0.000 |

## 分因子 pix

| factor | B0 |
|---|---:|
| clutter | 297.2 |
| crosscolor | 229.7 |
| desk | 193.6 |
| far | 193.8 |
| handheld | 186.2 |
| hardmix | 337.0 |
| occlude_hand | 162.9 |
| room | 16.3 |
| screen | 344.1 |
