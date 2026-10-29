# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-12T04:24:19Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.909 | 0.964 | 0.969 | 15.93 | 58.7 | 136.5 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| blur | 0.968 |
| clean | 0.962 |
| distractor | 0.812 |
| light | 0.898 |
| mixed | 0.963 |
| noise | 0.882 |
| occ | 0.938 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| blur | 3.9 |
| clean | 3.6 |
| distractor | 17.3 |
| light | 30.3 |
| mixed | 36.0 |
| noise | 4.5 |
| occ | 6.0 |
