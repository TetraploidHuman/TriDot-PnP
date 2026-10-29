# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-13T16:03:20Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.881 | 0.947 | 0.960 | 25.66 | 68.9 | 140.2 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| clutter | 0.828 |
| crosscolor | 0.945 |
| desk | 0.956 |
| far | 0.933 |
| handheld | 0.833 |
| hardmix | 0.580 |
| occlude_hand | 0.708 |
| room | 0.991 |
| screen | 0.808 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| clutter | 15.4 |
| crosscolor | 7.6 |
| desk | 3.8 |
| far | 3.6 |
| handheld | 52.6 |
| hardmix | 53.9 |
| occlude_hand | 38.3 |
| room | 3.0 |
| screen | 29.1 |
