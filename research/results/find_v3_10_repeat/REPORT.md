# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-12T16:30:04Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.772 | 0.897 | 0.930 | 38.12 | 85.0 | 130.0 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| clutter | 0.644 |
| crosscolor | 0.881 |
| desk | 0.908 |
| far | 0.872 |
| handheld | 0.669 |
| hardmix | 0.432 |
| occlude_hand | 0.661 |
| room | 0.988 |
| screen | 0.575 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| clutter | 28.6 |
| crosscolor | 14.0 |
| desk | 4.6 |
| far | 4.3 |
| handheld | 75.9 |
| hardmix | 82.4 |
| occlude_hand | 43.4 |
| room | 3.2 |
| screen | 48.7 |
