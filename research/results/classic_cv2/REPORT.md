# 经典 CV 检测器 v2：诊断、实现与诚实对比

> 数据来源：`research/results/honest_cv2/try4`（eval_v2，固定解码、无在线门控；test/hard_test 为历史调参集，
> holdout_hard 为 2026-10-28 新生成的、从未参与调参的留存集）。

## 0. 一句话结论

B0 的瓶颈主要在**前端**（hard_test 53% 的全可见帧有一色根本没有候选）与**三点选择**
（22% 有候选但拼错），而不是 PnP。把前端换成"颜色差 + top-hat 背景抑制 + 自适应阈值"，
把选择换成"光度/几何/尺寸/时序打分下的穷举三元组"，并保留 B0 的 HSV 候选做并集：

### test

| method | det@10 all | det@10 legacy | recall@10 | pose cov | t_err med (mm) | t_err p90 (mm) | jitter (mm/s) | est FPS |
|---|---|---|---|---|---|---|---|---|
| b0 | 0.634 | 0.918 | 0.790 | 0.800 | 3.6 | 114.4 | 722 | 119 |
| classic2 | 0.647 | 0.936 | 0.867 | 0.943 | 4.7 | 118.0 | 801 | 38 |

参考：学习型检测器（同一评测器、同一子集，来自 `research/results/honest/`）

| method | det@10 all | det@10 legacy | recall@10 | pose cov | t_err med (mm) | t_err p90 (mm) | est FPS |
|---|---|---|---|---|---|---|---|
| find_geom | 0.662 | 0.958 | 0.964 | 0.993 | 6.8 | 122.5 | 8 |
| find_offset_geom | 0.663 | 0.960 | 0.965 | 0.992 | 6.2 | 120.3 | 4 |


<details><summary>per-factor</summary>

| factor | B0 det@10 all | classic2 det@10 all | delta |
|---|---|---|---|
| blur | 0.921 | 0.921 | +0.000 |
| clean | 0.988 | 0.988 | +0.000 |
| distractor | 0.662 | 0.767 | +0.104 |
| light | 0.858 | 0.804 | -0.054 |
| mixed | 0.278 | 0.286 | +0.008 |
| noise | 0.912 | 0.946 | +0.033 |
| occ | 0.000 | 0.000 | +0.000 |

</details>

### hard_test

| method | det@10 all | det@10 legacy | recall@10 | pose cov | t_err med (mm) | t_err p90 (mm) | jitter (mm/s) | est FPS |
|---|---|---|---|---|---|---|---|---|
| b0 | 0.172 | 0.245 | 0.346 | 0.878 | 419.8 | 1089.8 | 3593 | 26 |
| classic2 | 0.501 | 0.714 | 0.711 | 0.970 | 24.0 | 867.0 | 1284 | 29 |

参考：学习型检测器（同一评测器、同一子集，来自 `research/results/honest/`）

| method | det@10 all | det@10 legacy | recall@10 | pose cov | t_err med (mm) | t_err p90 (mm) | est FPS |
|---|---|---|---|---|---|---|---|
| find_geom | 0.628 | 0.894 | 0.913 | 0.995 | 16.6 | 104.6 | 4 |
| find_offset_geom | 0.622 | 0.886 | 0.909 | 0.997 | 15.3 | 101.4 | 5 |


<details><summary>per-factor</summary>

| factor | B0 det@10 all | classic2 det@10 all | delta |
|---|---|---|---|
| clutter | 0.075 | 0.792 | +0.717 |
| crosscolor | 0.214 | 0.789 | +0.575 |
| desk | 0.214 | 0.864 | +0.650 |
| far | 0.297 | 0.744 | +0.447 |
| handheld | 0.183 | 0.608 | +0.425 |
| hardmix | 0.000 | 0.026 | +0.026 |
| occlude_hand | 0.000 | 0.000 | +0.000 |
| room | 0.731 | 0.919 | +0.189 |
| screen | 0.003 | 0.244 | +0.242 |

</details>

### holdout_hard

| method | det@10 all | det@10 legacy | recall@10 | pose cov | t_err med (mm) | t_err p90 (mm) | jitter (mm/s) | est FPS |
|---|---|---|---|---|---|---|---|---|
| b0 | 0.169 | 0.231 | 0.336 | 0.881 | 432.1 | 1027.3 | 3528 | 23 |
| classic2 | 0.492 | 0.674 | 0.692 | 0.965 | 27.7 | 812.8 | 1356 | 22 |

参考：学习型检测器（同一评测器、同一子集，来自 `research/results/honest/`）

| method | det@10 all | det@10 legacy | recall@10 | pose cov | t_err med (mm) | t_err p90 (mm) | est FPS |
|---|---|---|---|---|---|---|---|
| find_geom | 0.666 | 0.911 | 0.921 | 0.995 | 12.8 | 80.3 | 4 |
| find_offset_geom | 0.662 | 0.906 | 0.919 | 0.997 | 13.2 | 74.3 | 5 |


<details><summary>per-factor</summary>

| factor | B0 det@10 all | classic2 det@10 all | delta |
|---|---|---|---|
| clutter | 0.028 | 0.567 | +0.539 |
| crosscolor | 0.133 | 0.825 | +0.692 |
| desk | 0.142 | 0.708 | +0.567 |
| far | 0.356 | 0.967 | +0.611 |
| handheld | 0.275 | 0.703 | +0.428 |
| hardmix | 0.001 | 0.026 | +0.025 |
| occlude_hand | 0.000 | 0.000 | +0.000 |
| room | 0.744 | 0.900 | +0.156 |
| screen | 0.008 | 0.203 | +0.194 |

</details>

## 1. B0 失败诊断（`research/results/classic_diag/`）

- **test** (1800 帧 / 1244 全可见): B0 det@10 0.918, oracle 关联 0.999, 前端漏检帧占比 0.001, 关联拼错帧占比 0.081
- **hard_test** (3600 帧 / 2527 全可见): B0 det@10 0.245, oracle 关联 0.469, 前端漏检帧占比 0.531, 关联拼错帧占比 0.224
- **holdout_hard** (3600 帧 / 2631 全可见): B0 det@10 0.231, oracle 关联 0.416, 前端漏检帧占比 0.584, 关联拼错帧占比 0.184

## 2. 实现

`research/tridot_temporal/classic_v2.py`

1. **前端**：`red = clip(R - max(G,B), 0)` 等颜色差图 → `cv2.MORPH_TOPHAT`
   （31×31）去掉缓慢变化的背景（墙面/桌面/屏幕泛光）→ 阈值
   `max(8, 0.22 * p99.5)` → 连通域 → 面积/长宽比过滤 → 窗口加权质心（亚像素）。
2. **候选并集**：B0 的固定 HSV 前端在"室内泛光、低饱和"场景仍有独立价值，因此
   把 B0 候选（半分辨率 seed + 全分辨率质心精修）并入候选池，由打分器仲裁。
3. **三点选择**：穷举颜色三元组，代价 = 形状（等边性）+ 尺寸一致性 + 光度 +
   深度-边长一致性 + **软时序先验**（上一帧接受位姿的常速预测，只加罚分不硬拒绝）。
4. **时序状态**：只在 PnP 成功且位姿跳变在 `max(150mm, 0.25·depth)` 内时更新；
   跳变超限丢弃状态，避免错误关联污染速度。
5. `eval_v2.py --methods b0,classic2` 复用同一评测器（`det10_all`、逐点召回、
   姿态覆盖、t_err、jitter、延迟）。

## 3. 迁移风险：哪些收益依赖当前合成集

把 `classic2` 拆成两组开关看收益来源（同一评测器、同一解码）：

| 配置 | test det@10 all | hard_test det@10 all | holdout_hard det@10 all |
|---|---|---|---|
| try2 无 top-hat | 0.659 | 0.367 | 0.366 |
| try3 恒开 top-hat | 0.647 | 0.503 | 0.489 |
| try4 自适应 top-hat | 0.647 | 0.501 | 0.492 |

- **场景无关的部分**（颜色差前端 + B0 候选并集 + 几何/光度/尺寸/时序三元组打分 + 亚像素加权质心 + pose-jump 门控）：hard_test 0.172 → 0.367，holdout 0.169 → 0.366；这部分不依赖"有泛光"的假设。
- **场景相关的部分**（31×31 top-hat + `p99.5>15 / p50>2` 触发规则）：再 +0.13，但只在"亮而平滑的墙面/桌面/屏幕泛光"这类合成 preset 上验证过；真实场景若没有这种背景，这部分收益无法预期，甚至可能在 easy 域小亏（test 0.659 → 0.647）。
- 因此应把 top-hat 视为"假设集里的一路"而不是默认前置：真实数据到位后应改为"raw / top-hat / 尺度自适应带通"三路并集、由打分器仲裁；核大小按预期光斑半径（`r_px ≈ fx·r_mm/depth`，depth 取上一帧位姿）设为 3–5×`r_px`。


## 4. 复现

```bash
research/.venv/bin/python -m research.scripts.b0_diag --subset test,hard_test,holdout_hard
research/.venv/bin/python -m research.tridot_temporal.eval_v2 \
    --subset hard_test --methods b0,classic2 --out research/results/honest_cv2/try4/hard_test
research/.venv/bin/python -m research.scripts.classic_cv2_report
```

## 5. 局限

- `occlude_hand` 因子按定义 0 帧三点全可见，`det@10 all` 上该因子恒为 0；
- `hardmix` 大量帧为部分遮挡，补全需要"2 点 + 已知几何"才能改善；
- 延迟仍高于 B0（多为 top-hat / 双前端），端侧部署前需要下采样或 NEON 级优化；
- 无真实数据：所有结论只覆盖合成域，真实集到位后必须重跑
  `eval_v2 --subset real_test`。
