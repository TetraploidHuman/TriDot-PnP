# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-13T14:36:04Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.794 | 0.915 | 0.947 | 28.00 | 71.6 | 122.9 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| clutter | 0.611 |
| crosscolor | 0.890 |
| desk | 0.908 |
| far | 0.894 |
| handheld | 0.695 |
| hardmix | 0.451 |
| occlude_hand | 0.706 |
| room | 0.938 |
| screen | 0.760 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| clutter | 15.8 |
| crosscolor | 9.8 |
| desk | 5.5 |
| far | 5.0 |
| handheld | 55.9 |
| hardmix | 65.0 |
| occlude_hand | 35.1 |
| room | 4.3 |
| screen | 23.8 |
