# B0-Classic vs B1-TriDot 对比

- 时间：2026-10-10T14:23:49Z
- 数据：`/home/miaox99/TriDotPnP/datasets/TriDotPose` split=`test`（7 序列）
- B1 深度 PnP 内参：App FOV≈60° 估计
- 方法：`B0`=经典；`B1`=App 检测+深度迭代 PnP；`B1det_B0pnp`=App 检测+OpenCV PnP（隔离检测）

## 总体

| method | det@10 | detect_ok | pix_rmse | t_err_mm | r_err_deg | FPS | p50_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.978 | 0.798 | 11.89 | 18.2 | 17.4 | 86.6 | 8.06 |
| B1 | 0.980 | 0.795 | 7.69 | nan | nan | 1.1 | 367.80 |
| B1det_B0pnp | 0.980 | 0.795 | 7.69 | 34.1 | 57.6 | 1.1 | 367.80 |

## 分因子 det@10

| factor | B0 | B1 | B1det_B0pnp |
|---|---:|---:|---:|
| blur | 1.000 | 1.000 | 1.000 |
| clean | 1.000 | 1.000 | 1.000 |
| distractor | 0.900 | 0.900 | 0.900 |
| light | 0.989 | 1.000 | 1.000 |
| mixed | 0.411 | 0.367 | 0.367 |
| noise | 1.000 | 1.000 | 1.000 |
| occ | 0.000 | 0.000 | 0.000 |

## 分因子像素 RMSE

| factor | B0 | B1 | B1det_B0pnp |
|---|---:|---:|---:|
| blur | 0.9 | 3.1 | 3.1 |
| clean | 0.8 | 2.3 | 2.3 |
| distractor | 5.9 | 11.6 | 11.6 |
| light | 0.8 | 2.1 | 2.1 |
| mixed | 87.3 | 34.8 | 34.8 |
| noise | 0.8 | 2.3 | 2.3 |
| occ | nan | nan | nan |

## 结论要点

- **检测**：`B0` vs `B1` 的 det@10 / pix_rmse 直接可比。
- **位姿（公平）**：看 `B1det_B0pnp`（同一 OpenCV PnP）；完整 App 管线 `B1` 的深度迭代在本数据上常不收敛（edge 误差大，已过滤）。
- B1 为 Kotlin App 的 Python 对齐实现；Python 网格实现偏慢，不代表手机 App 帧率。

## 图

![](figures/b0_vs_b1_overview.png)

![](figures/det_by_factor.png)
