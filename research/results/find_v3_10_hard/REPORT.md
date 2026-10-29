# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-12T16:28:10Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.786 | 0.932 | 0.972 | 21.40 | 53.9 | 126.9 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| clutter | 0.633 |
| crosscolor | 0.951 |
| desk | 0.972 |
| far | 0.869 |
| handheld | 0.673 |
| hardmix | 0.549 |
| occlude_hand | 0.642 |
| room | 0.923 |
| screen | 0.554 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| clutter | 10.6 |
| crosscolor | 5.1 |
| desk | 3.7 |
| far | 3.9 |
| handheld | 61.4 |
| hardmix | 33.8 |
| occlude_hand | 37.3 |
| room | 3.8 |
| screen | 24.0 |
