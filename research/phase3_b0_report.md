# 阶段 3：B0-Classic 基线评测报告

- 时间：2026-10-10T09:50:45Z
- 数据集：`/home/miaox99/TriDotPnP/datasets/TriDotPose`
- 基线：B0-Classic（HSV 连通域 + 几何验证 + OpenCV solvePnP）
- 序列数：7

## 1. 总体结果

| 指标 | 数值 |
|------|------|
| 平均检测成功率（按序列） | 0.762 |
| 平均 FPS | 98.5 |
| 平均延迟 p50 (ms) | 11.91 |
| 全局像素 RMSE 均值 (px) | 13.192 |
| 自动消歧平移误差均值 (mm) | 34.11 |
| 自动消歧旋转误差均值 (deg) | 32.30 |
| Oracle 消歧平移误差均值 (mm) | 18.21 |
| Oracle 消歧旋转误差均值 (deg) | 7.75 |

> 说明：共面三点 PnP 存在双解。**自动消歧**用时序连续性选解；**Oracle**在候选解中选最接近 GT 的解，用于把检测误差与双解翻转分开。

## 2. 分序列表

| seq_id | factor | det_succ | pix_rmse | t_auto | r_auto | t_oracle | r_oracle | FPS | p50_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| syn_blur_0003 | blur | 1.000 | 0.835 | 33.27 | 37.87 | 1.85 | 4.32 | 117.4 | 7.49 |
| syn_clean_0001 | clean | 1.000 | 0.953 | 10.53 | 9.44 | 2.75 | 0.41 | 125.9 | 6.78 |
| syn_distractor_0005 | distractor | 0.893 | 26.931 | 85.72 | 24.72 | 81.98 | 19.82 | 99.7 | 7.71 |
| syn_light_0002 | light | 1.000 | 0.940 | 36.55 | 46.95 | 2.13 | 4.20 | 133.2 | 6.58 |
| syn_mixed_0007 | mixed | 0.400 | 81.986 | nan | nan | nan | nan | 35.2 | 27.39 |
| syn_noise_0006 | noise | 1.000 | 0.827 | 4.15 | 42.23 | 2.15 | 9.89 | 133.4 | 6.26 |
| syn_occ_0004 | occ | 0.040 | 1.599 | nan | nan | nan | nan | 44.3 | 21.13 |

## 3. 分因子汇总

| factor | seqs | det_succ | pix_rmse | t_auto | t_oracle | r_auto | r_oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| blur | 1 | 1.000 | 0.835 | 33.27 | 1.85 | 37.87 | 4.32 |
| clean | 1 | 1.000 | 0.953 | 10.53 | 2.75 | 9.44 | 0.41 |
| distractor | 1 | 0.893 | 26.931 | 85.72 | 81.98 | 24.72 | 19.82 |
| light | 1 | 1.000 | 0.940 | 36.55 | 2.13 | 46.95 | 4.20 |
| mixed | 1 | 0.400 | 81.986 | nan | nan | nan | nan |
| noise | 1 | 1.000 | 0.827 | 4.15 | 2.15 | 42.23 | 9.89 |
| occ | 1 | 0.040 | 1.599 | nan | nan | nan | nan |

## 4. 曲线图

![](figures/metrics_by_factor.png)

![](figures/latency_hist.png)

![](figures/clean_error_curves.png)

## 5. 失败案例分析

- GT 全可见但检测失败帧数：8
- 疑似颜色错配帧数：0

代表性失败帧（白叉=GT，彩圈=预测）：

- `phase3_b0/failures/syn_distractor_0005_f0000_miss.jpg`
- `phase3_b0/failures/syn_distractor_0005_f0021_miss.jpg`
- `phase3_b0/failures/syn_distractor_0005_f0073_miss.jpg`
- `phase3_b0/failures/syn_mixed_0007_f0000_miss.jpg`
- `phase3_b0/failures/syn_mixed_0007_f0035_miss.jpg`
- `phase3_b0/failures/syn_mixed_0007_f0051_miss.jpg`
- `phase3_b0/failures/syn_mixed_0007_f0066_miss.jpg`

![](failures/syn_distractor_0005_f0000_miss.jpg)

![](failures/syn_distractor_0005_f0021_miss.jpg)

![](failures/syn_distractor_0005_f0073_miss.jpg)

![](failures/syn_mixed_0007_f0000_miss.jpg)

### 定性结论（基于本轮示例集）

- **检测主战场**：`distractor` / `occlusion` / `mixed` 下漏检与错选显著；`clean`/`light`/`noise`/`blur` 像素误差约 1 px。
- **位姿双解**：自动消歧在部分序列会锁到错误分支；Oracle 误差通常小得多——说明检测够好时几何可解，瓶颈在消歧与动态检测。
- **时序改进空间**：短暂遮挡与同色干扰最适合用 Transformer/轨迹先验增强检测，而非改 PnP。
- **速度**：CPU 上约 100+ FPS（720p），满足手机端实时余量（ONNX 对照留给后续阶段）。

## 6. 代码结构

| 模块 | 路径 |
|------|------|
| 检测器 | `research/baseline_b0/detector.py` |
| PnP | `research/baseline_b0/pnp.py` |
| 评测 | `research/baseline_b0/evaluate.py` |
| 本报告产物 | `research/results/phase3_b0/` |

## 7. 复现命令

```bash
source research/.venv/bin/activate
python -m research.baseline_b0.evaluate --dataset datasets/TriDotPose --split examples
```
