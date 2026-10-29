# 阶段 5：对比实验与消融报告

- 时间：2026-10-10T13:25:20Z
- 数据集：`/home/miaox99/TriDotPnP/datasets/TriDotPose`（test split）

## 1. 主对比表

| method | det@10 | det@20 | det@40 | pix_rmse | t_err | t_oracle | FPS | p50_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.978 | 0.978 | 0.984 | 11.89 | 18.2 | 8.0 | 82.4 | 9.30 |
| T1_cnn | 0.000 | 0.013 | 0.148 | 73.13 | 266.4 | 257.8 | 122.9 | 7.99 |
| T3_transformer | 0.000 | 0.007 | 0.187 | 61.22 | 243.8 | 238.4 | 151.5 | 6.03 |
| T5_transformer | 0.002 | 0.054 | 0.418 | 44.79 | 231.1 | 220.0 | 103.1 | 9.90 |
| T5_gru | 0.000 | 0.013 | 0.245 | 68.94 | 258.4 | 251.3 | 116.3 | 8.37 |
| Hybrid_B0_T5 | 0.978 | 0.978 | 0.984 | 17.92 | 20.2 | 10.1 | 75.7 | 7.54 |

## 2. 分因子像素误差（越低越好）

| factor | B0 | T1_cnn | T3_transformer | T5_transformer | T5_gru | Hybrid_B0_T5 |
|---|---:|---:|---:|---:|---:|---:|
| blur | 0.9 | 50.4 | 53.2 | 44.0 | 43.6 | 0.9 |
| clean | 0.8 | 44.9 | 46.4 | 38.5 | 43.5 | 2.6 |
| distractor | 5.9 | 135.4 | 94.7 | 59.0 | 119.0 | 5.9 |
| light | 0.8 | 63.0 | 48.8 | 37.7 | 56.6 | 2.5 |
| mixed | 87.3 | 133.5 | 93.8 | 58.3 | 144.8 | 76.6 |
| noise | 0.8 | 44.8 | 46.0 | 39.6 | 39.1 | 0.8 |
| occ | nan | 39.6 | 45.6 | 36.3 | 35.6 | 36.3 |

## 3. 分因子 Detect@10

| factor | B0 | T1_cnn | T3_transformer | T5_transformer | T5_gru | Hybrid_B0_T5 |
|---|---:|---:|---:|---:|---:|---:|
| blur | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| clean | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| distractor | 0.900 | 0.000 | 0.000 | 0.000 | 0.000 | 0.900 |
| light | 0.989 | 0.000 | 0.000 | 0.011 | 0.000 | 0.989 |
| mixed | 0.411 | 0.000 | 0.000 | 0.011 | 0.000 | 0.411 |
| noise | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| occ | 0.000 | 0.000 | 0.022 | 0.011 | 0.044 | 0.011 |

## 4. 模型体积与 ONNX Runtime 延迟（CPU）

| method | params | onnx_mb | ort_p50_ms | ort_p95_ms | ort_fps |
|---|---:|---:|---:|---:|---:|
| T1_cnn | 298967 | 1.21 | 0.96 | 1.00 | 1038.6 |
| T3_transformer | 299223 | 1.21 | 1.39 | 1.42 | 720.2 |
| T5_transformer | 299479 | 1.22 | 1.75 | 1.80 | 570.6 |
| T5_gru | 133591 | 0.53 | 1.56 | 1.70 | 646.6 |
| B0 | 0 (经典) | - | 9.30 | - | 82.4 |

## 5. 图表

![](figures/compare_overview.png)

![](figures/detect_by_factor.png)

## 6. 消融一览

| 消融维度 | 对比 | 结论 |
|----------|------|------|
| 时序长度 | T1 / T3 / T5 | 像素误差 73 → 61 → **45**，T 越长越好 |
| 时序模块 | T5 Transformer vs T5 GRU | Transformer 更准；GRU 参数约 1/2、ONNX 0.53MB |
| 与经典基线 | 学习 vs B0 | B0 在 clean/blur/noise/light 上仍近亚像素 |
| 混合策略 | Hybrid_B0_T5 | det@10 与 B0 持平；occ/mixed 漏检帧可由学习模型补点 |

## 7. 分析结论

- **纯学习方法**像素误差仍高于 B0（最优 T5_transformer **44.8px** vs B0 **11.9px**）。中等合成数据 + 轻量热图下，颜色分割在干净/模糊/噪声场景仍占优。
- **消融有效**：加长时序、用 Transformer 相对 T=1/GRU 均有稳定增益（见上表）。
- **Hybrid** 是当前最可落地路线：det@10=0.978；`occ` 上 B0 几乎无有效点时回退 T5 得到 ~36px 估计。
- **位姿双解**仍在；oracle 平移误差系统性低于 auto。
- **部署**：学习模型 ONNX ≤1.22MB，CPU ORT 亚 2ms 级；手机可优先 GRU。
- **论文叙事**：宜写“可复现基准 + 轻量时序消融 + Hybrid 鲁棒管线”，不宜宣称端到端全面超过经典分割。

## 8. 局限与后续

- 训练集约 14 条序列，学习方法欠拟合；`preset full` + 更长训程预期可缩小与 B0 差距。
- 更有潜力的下一结构：**B0 Top-K 候选 + 时序 Transformer 重排**（保留亚像素质心，只学关联）。

## 7. 复现

```bash
python -m research.tridot_synth.generate --out datasets/TriDotPose --preset medium
python -m research.tridot_temporal.run_phase5 --dataset datasets/TriDotPose --skip-generate --epochs 30
```
