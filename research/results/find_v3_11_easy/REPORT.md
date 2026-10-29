# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-13T10:43:44Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.932 | 0.963 | 0.969 | 14.76 | 50.3 | 140.1 |

## 分因子 det@10

| factor | FindNet_geom |
|---|---:|
| blur | 0.955 |
| clean | 0.983 |
| distractor | 0.821 |
| light | 0.942 |
| mixed | 0.954 |
| noise | 0.950 |
| occ | 0.958 |

## 分因子 pix

| factor | FindNet_geom |
|---|---:|
| blur | 4.1 |
| clean | 3.5 |
| distractor | 15.2 |
| light | 18.4 |
| mixed | 39.5 |
| noise | 4.1 |
| occ | 6.2 |
