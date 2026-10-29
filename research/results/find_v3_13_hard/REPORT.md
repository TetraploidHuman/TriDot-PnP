# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-13T14:34:06Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.770 | 0.945 | 0.979 | 16.85 | 47.3 | 118.8 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| clutter | 0.564 |
| crosscolor | 0.899 |
| desk | 0.961 |
| far | 0.908 |
| handheld | 0.628 |
| hardmix | 0.593 |
| occlude_hand | 0.647 |
| room | 0.807 |
| screen | 0.671 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| clutter | 11.1 |
| crosscolor | 4.8 |
| desk | 4.0 |
| far | 5.0 |
| handheld | 53.1 |
| hardmix | 22.9 |
| occlude_hand | 27.8 |
| room | 5.5 |
| screen | 13.8 |
