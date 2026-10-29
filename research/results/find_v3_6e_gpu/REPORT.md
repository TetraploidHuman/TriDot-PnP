# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-11T10:43:52Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.950 | 0.977 | 0.980 | 15.90 | 47.0 | 144.8 |
| FindNet_geom_fast | 0.950 | 0.977 | 0.980 | 15.90 | 47.0 | 167.8 |

## 分因子 det@10

| factor | FindNet_geom | FindNet_geom_fast |
|---|---:|---:|
| blur | 0.986 | 0.986 |
| clean | 1.000 | 1.000 |
| distractor | 0.850 | 0.850 |
| light | 0.942 | 0.942 |
| mixed | 0.926 | 0.926 |
| noise | 0.983 | 0.983 |
| occ | 0.912 | 0.912 |

## 分因子 pix

| factor | FindNet_geom | FindNet_geom_fast |
|---|---:|---:|
| blur | 4.0 | 4.0 |
| clean | 3.2 | 3.2 |
| distractor | 14.8 | 14.8 |
| light | 13.8 | 13.8 |
| mixed | 46.9 | 46.9 |
| noise | 3.8 | 3.8 |
| occ | 9.3 | 9.3 |
