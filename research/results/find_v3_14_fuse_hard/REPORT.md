# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-13T16:25:19Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.816 | 0.960 | 0.982 | 20.33 | 48.4 | 117.8 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| clutter | 0.653 |
| crosscolor | 0.930 |
| desk | 0.975 |
| far | 0.942 |
| handheld | 0.636 |
| hardmix | 0.636 |
| occlude_hand | 0.619 |
| room | 0.840 |
| screen | 0.778 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| clutter | 11.7 |
| crosscolor | 6.9 |
| desk | 3.5 |
| far | 4.0 |
| handheld | 51.8 |
| hardmix | 33.0 |
| occlude_hand | 41.8 |
| room | 5.1 |
| screen | 16.2 |
