# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-13T10:42:45Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.828 | 0.930 | 0.954 | 23.15 | 60.2 | 130.3 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| clutter | 0.767 |
| crosscolor | 0.913 |
| desk | 0.950 |
| far | 0.917 |
| handheld | 0.714 |
| hardmix | 0.463 |
| occlude_hand | 0.750 |
| room | 0.982 |
| screen | 0.692 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| clutter | 12.8 |
| crosscolor | 6.4 |
| desk | 3.9 |
| far | 3.7 |
| handheld | 56.6 |
| hardmix | 50.9 |
| occlude_hand | 24.5 |
| room | 3.0 |
| screen | 22.7 |
