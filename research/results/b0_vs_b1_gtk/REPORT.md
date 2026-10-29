# B0-Classic vs B1-TriDot 对比

- 时间：2026-10-10T14:11:26Z
- 数据：`/home/miaox99/TriDotPnP/datasets/TriDotPose` split=`test`（7 序列）
- B1 PnP 内参：数据集 GT K

## 总体

| method | det@10 | detect_ok | pix_rmse | t_err_mm | r_err_deg | FPS | p50_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.978 | 0.798 | 11.89 | 18.2 | 17.4 | 83.4 | 8.53 |
| B1_gtK | 0.980 | 0.795 | 7.69 | 505.1 | 96.9 | 1.1 | 385.63 |

## 分因子 det@10

| factor | B0 | B1_gtK |
|---|---:|---:|
| blur | 1.000 | 1.000 |
| clean | 1.000 | 1.000 |
| distractor | 0.900 | 0.900 |
| light | 0.989 | 1.000 |
| mixed | 0.411 | 0.367 |
| noise | 1.000 | 1.000 |
| occ | 0.000 | 0.000 |

## 分因子像素 RMSE

| factor | B0 | B1_gtK |
|---|---:|---:|
| blur | 0.9 | 3.1 |
| clean | 0.8 | 2.3 |
| distractor | 5.9 | 11.6 |
| light | 0.8 | 2.1 |
| mixed | 87.3 | 34.8 |
| noise | 0.8 | 2.3 |
| occ | nan | nan |

## 结论要点

- 检测（det@10 / 像素）：直接可比；看上表 **B0** vs **B1_gtK**。
- 位姿：B0 用 OpenCV PnP + 数据集内参；B1 默认同 App 用 FOV 估焦距（可用 `--b1-gt-k` 消融）。
- B1 为 Kotlin App 的 Python 对齐实现（软 HSV + 网格 Top-N + 深度迭代），非逐行比特级复刻。

## 图

![](figures/b0_vs_b1_overview.png)

![](figures/det_by_factor.png)
