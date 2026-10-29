# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-13T16:07:06Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.883 | 0.950 | 0.964 | 23.30 | 68.1 | 143.5 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| clutter | 0.831 |
| crosscolor | 0.945 |
| desk | 0.956 |
| far | 0.933 |
| handheld | 0.833 |
| hardmix | 0.605 |
| occlude_hand | 0.711 |
| room | 0.991 |
| screen | 0.811 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| clutter | 14.5 |
| crosscolor | 5.1 |
| desk | 3.8 |
| far | 3.6 |
| handheld | 50.4 |
| hardmix | 47.5 |
| occlude_hand | 38.6 |
| room | 3.0 |
| screen | 23.5 |
