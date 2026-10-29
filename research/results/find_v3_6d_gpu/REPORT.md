# FindNet：纯学习找点 vs B0（无经典 proposal）

- 时间：2026-10-11T10:22:47Z
- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道
- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FindNet_geom | 0.936 | 0.976 | 0.982 | 16.13 | 52.4 | 146.2 |
| FindNet_geom_fast | 0.936 | 0.976 | 0.982 | 16.13 | 52.4 | 172.0 |

## 分因子 det@10

| factor | FindNet_geom | FindNet_geom_fast |
|---|---:|---:|
| blur | 0.982 | 0.982 |
| clean | 0.992 | 0.992 |
| distractor | 0.846 | 0.846 |
| light | 0.913 | 0.913 |
| mixed | 0.926 | 0.926 |
| noise | 0.950 | 0.950 |
| occ | 0.883 | 0.883 |

## 分因子 pix

| factor | FindNet_geom | FindNet_geom_fast |
|---|---:|---:|
| blur | 4.1 | 4.1 |
| clean | 3.4 | 3.4 |
| distractor | 15.1 | 15.1 |
| light | 12.9 | 12.9 |
| mixed | 46.5 | 46.5 |
| noise | 4.2 | 4.2 |
| occ | 11.6 | 11.6 |
