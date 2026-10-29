# 遮挡帧的时序补全（Temporal Completion）— 实验与端侧实现

日期：2026-10-28 · 作者：本轮自动实验记录
相关代码：`research/tridot_temporal/completion.py`、`research/scripts/occlusion_experiment.py`、
`research/scripts/completion_ab.py`、`composeApp/.../PnPTemporalTracker.kt`、
`PnPDistanceCalculator.kt`、`DetectionCoordinator.kt`

## 0. 一句话结论

- **补全在「只检测到 2 个点、第 3 个点完全没有像素」时是有用的**：受控实验的 unmasked 窗口
  t_err 中位 12.37 → 9.09 mm（-27%），全部帧 14.33 → 10.50 mm。
- **补全在「连续 8 帧完全遮住同一个点」时会变差**：masked 窗口 t_err 61.42 → 87.58 mm。
  原因是匀速外推误差随遮挡帧数累积；这说明外推帧数（`maxCoastFrames`/`max_gap`）应保持很小。
- **补全不应该覆盖「检测到了但颜色证据弱」的部分遮挡点**：自然遮挡实验里这类帧
  t_err 121.91 → 249.94 mm（直接测量比几何外推更准）。
- 端侧只在「某个颜色的点完全没检测到」（`spots.size == 2`）时启用，且只输出姿态外推、
  不把观测写回滤波器；连续外推超过 `maxCoastFrames` 后自动停止。

## 1. 动机

hard/realistic 子集约 30% 的帧不是三点全可见（`occlude_hand` 0 帧三点全可见），
而 3 点 PnP 需要三个像素。目标是刚体 + 已知等边三角形（边长可配置，默认 100mm），
所以可以用「上一帧位姿 + 速度」把缺失的角点像素投影出来，让姿态在短暂丢点时不断档。

实现：`research/tridot_temporal/completion.py`

- `TemporalCompleter(vis_thr=0.35, max_gap=8, max_missing=1, edge_margin=4.0, min_depth_mm=80.0)`
- 状态只由「三点都观测到」的帧更新（`fixed_solve_depths` 收敛且边误差 ≤2mm 才接受）；
- `complete()`：预测 `t_pred = t + (t - t_prev)/span * gap`，投影 `R @ obj_i + t_pred`；
  返回 `observed / no-state / stale / too-many-missing / completed / offscreen`；
- 只补缺失的角点，绝不修改已观测到的角点。

## 2. 受控遮挡实验（masked test）

做法（`research/scripts/occlusion_experiment.py`）：在 test 的 15 条序列上，把 GT blue 点
在第 40–48 帧（8 帧）用环带中值涂掉，模型看到的就是「真的没有这个点」。
触发条件 = 预测点 10px 窗口内平均颜色证据 < 12（可见蓝点的证据中位 19–74，涂掉后 0–5）。
两种配置共用同一批检测/解码/snap/PnP。

| config | window | det@10 | pose cov | t_err med (mm) | blue err med (px) | completed pts |
|---|---|---|---|---|---|---|
| no_completion | masked | 0.200 | 0.992 | 61.42 | 64.4 | 0 |
| completion | masked | 0.092 | 0.992 | **87.58** | 54.9 | 112 |
| no_completion | unmasked | 0.703 | 0.992 | 12.37 | 1.1 | 0 |
| completion | unmasked | 0.696 | 0.992 | **9.09** | 1.1 | 476 |
| no_completion | all | 0.669 | 0.992 | 14.33 | 1.2 | 0 |
| completion | all | 0.656 | 0.992 | **10.50** | 1.2 | 588 |

原始数据：`research/results/completion_exp/test_masked8/summary.json`。

解读：

- unmasked 窗口里触发的是「检测到假点 / 该颜色本来就不在画面里」的帧（476 点），
  用几何外推替换这些点后 t_err 明显下降——这正是端侧 `spots.size == 2` 的情形。
- masked 窗口是「连续 8 帧完全没有观测」，匀速外推的误差随帧数增长，姿态反而更差；
  det@10 也受影响（补出来的点进不了 10px）。
  → 外推只应维持 1–3 帧；本项目的 `maxCoastFrames` 默认 5，属于偏乐观的取值。

## 3. 自然遮挡实验（evidence 触发，negative result）

在同一批 FindNet 解码上，用「像素颜色证据 < 12」而不是 vis 头来触发补全，
跑 hard_test 的 `occlude_hand` + `hardmix` 前 3 条序列（270 帧，全部至少缺 1 个点）。
命令见 `research/scripts/completion_ab.py`。

| config | frames | pose cov | t_err med (mm) | t_err p90 | <=50mm | completed pts |
|---|---|---|---|---|---|---|
| no-comp / 1 missing | 184 | 1.000 | 87.00 | 514.76 | 0.33 | 0 |
| ev+comp / 1 missing | 184 | 1.000 | **110.28** | 652.10 | 0.30 | 50 |
| no-comp / fired frames | 76 | 1.000 | 121.91 | 582.80 | 0.25 | 0 |
| ev+comp / fired frames | 76 | 1.000 | **249.94** | 673.66 | 0.18 | 76 |
| no-comp / has vis==0 | 48 | 0.958 | 347.63 | 5027.94 | 0.07 | 0 |
| ev+comp / has vis==0 | 48 | 0.958 | 300.53 | 5032.18 | 0.09 | 10 |

结论：**不要用补全去覆盖还能定位到的部分遮挡点**（GT vis=1 的那类帧，直接测量的
像素误差中位只有 2.4px，几何外推反而更差）；对 vis=0 的帧里补全略微有帮助，
但 5 秒级的 p90 说明这些帧本身是病态的。另一个事实：`vis` 头在当前 offset-only ckpt 上
输出 0.978–1.0（饱和失效），所以触发器不能用 vis 头，只能用像素颜色证据。

## 4. 端侧实现（Android / Kotlin）

- `PnPTemporalTracker.projectCorner(colorIndex, edgeMm, fx, fy, cx, cy, timeSec)`：
  用滤波后的「质心 + 速度」预测当前中心，用最后一帧旋转把目标系角点投到像素；
- `PnPTemporalTracker.predictSnapshot(timeSec)`：只预测、**不更新滤波器状态**，
  连续超过 `maxCoastFrames` 帧返回 null（避免外推漂移写回状态）；
- `PnPDistanceCalculator.predictMissingPixel(...)`：按 FOV 估焦，越界返回 null；
- `PnPDistanceCalculator.calculate3PointPnPCompleted(...)`：用 `predictSnapshot` 产出
  `PnPResult`（DISTANCE / AZIMUTH / ELEVATION / 6DOF），复用 `buildTrackedResult`；
- `DetectionCoordinator.calculateRgbPnpResult`：
  - `spots.size == 3` → 原 tracked 路径（吸收观测）；
  - `spots.size == 2` 且开启时序滤波 → 外推姿态一帧（不吸收观测）；
  - 其它 → 返回 null（与原行为一致）。
- 单测：`composeApp/src/androidUnitTest/.../PnPTemporalTrackerTest.kt` 共 11 个测试全部通过
  （含「投影角点与真值 <1px」「匀速目标外推」「连续降级到上限后返回 null」「外推不污染状态」）。
- 产物：`composeApp/build/outputs/apk/debug/composeApp-debug.apk`（68,786,987 B，19:53），
  dex 中确认包含 `projectCorner`、`predictSnapshot`、`calculate3PointPnPCompleted`。

## 5. 局限与后续

1. **点出画**（GT vis=0 且外推投影出界）不可补：需要「2 点 + 已知几何」的 PnP，
   属于下一步工作；当前会回退为「本帧无结果」。
2. 端侧 `maxCoastFrames` 默认 5 偏乐观：第 7 节的逐序列/max_gap 扫描显示「补全是否改善精度」不稳定
   （15 序列上仅 5/15 变好），所以它只应作为「缺色时保持一帧位姿」的连续性兜底，不承诺精度；
   建议真实数据到位后再扫描 `maxCoastFrames ∈ {2,3,5}` 决定默认值（构造参数可调，App 未暴露）。
3. 外推只用平移速度、假设旋转不变；快速旋转时投影会偏。
4. 这个模块不改变 `det@10 (all frames)` 的统计定义：GT vis<2 的帧仍然记为失败，
   所以收益只能在「零样本丢点」场景和 Pose coverage / t_err 上体现。

## 6. 复现

```bash
# 受控遮挡实验（test 15 序列，蓝点 40..48 帧）
research/.venv/bin/python -m research.scripts.occlusion_experiment \
  --subset test --limit-seq 15 --mask-start 40 --mask-end 48 \
  --out research/results/completion_exp/test_masked8

# 自然遮挡实验：两份 eval_v2 + 对比
research/.venv/bin/python -m research.tridot_temporal.eval_v2 --subset hard_test \
  --factors occlude_hand,hardmix --limit-seq 3 --methods find_offset_geom \
  --find-ckpt research/results/find_v3_25_offset/best.pt \
  --geom-ckpt research/results/find_geom_v34/best.pt \
  --snap-radius 14 --top-k 5 --upsample 1 --out /tmp/nat_none
research/.venv/bin/python -m research.tridot_temporal.eval_v2 --subset hard_test \
  --factors occlude_hand,hardmix --limit-seq 3 --methods find_offset_geom \
  --find-ckpt research/results/find_v3_25_offset/best.pt \
  --geom-ckpt research/results/find_geom_v34/best.pt \
  --snap-radius 14 --top-k 5 --upsample 1 --complete --complete-mode evidence \
  --out /tmp/nat_evidence
research/.venv/bin/python -m research.scripts.completion_ab \
  --a /tmp/nat_none --b /tmp/nat_evidence --method find_offset_geom

# max_gap 扫描（test 前 6 序列，mask 8 帧）
for gap in 2 12; do
  research/.venv/bin/python -m research.scripts.occlusion_experiment \
    --subset test --limit-seq 6 --mask-start 40 --mask-end 48 --max-gap $gap \
    --out research/results/completion_exp/gap$gap
done
```

测试：`research/.venv/bin/python -m unittest discover -s research/tests` → 68 tests OK；
`:composeApp:testDebugUnitTest` → 11 tests OK。

## 7. 补充：逐序列稳定性与 max_gap 扫描（2026-10-28 晚）

### 7.1 15 序列 masked 窗口逐序列中位（mask 8 帧，max_gap=12）

**5/15 条序列变好、10/15 变差**（中位 61.4 → 87.6 mm）——收益不稳定，取决于该段运动是否接近匀速、
以及遮挡期间检测器是否产生了位置错误的「假点」。

| 序列 | 无补全 t_err（mm） | 有补全 t_err（mm） | 变化 |
|---|---|---|---|
| `syn_blur_0026` | 69.4 | 20.2 | -49.2 |
| `syn_blur_0027` | 5.5 | 13.5 | +8.0 |
| `syn_clean_0005` | 65.7 | 12.9 | -52.8 |
| `syn_clean_0009` | 26.9 | 28.9 | +2.0 |
| `syn_distractor_0041` | 347.9 | 517.4 | +169.5 |
| `syn_distractor_0050` | 31.9 | 151.4 | +119.5 |
| `syn_light_0012` | 89.5 | 84.8 | -4.7 |
| `syn_light_0017` | 17.4 | 33.0 | +15.6 |
| `syn_mixed_0066` | 15.5 | 236.2 | +220.7 |
| `syn_mixed_0069` | 596.6 | 887.8 | +291.2 |
| `syn_mixed_0072` | 66.1 | 89.3 | +23.2 |
| `syn_noise_0054` | 364.7 | 247.4 | -117.3 |
| `syn_noise_0058` | 45.5 | 112.8 | +67.3 |
| `syn_occ_0031` | 56.3 | 122.8 | +66.5 |
| `syn_occ_0037` | 68.7 | 36.6 | -32.1 |

### 7.2 max_gap 扫描（test 前 6 条序列，mask 8 帧）

| max_gap | 窗口 | 无补全 t_err（mm） | 有补全 t_err（mm） | det@10 无补全 | det@10 有补全 |
|---|---|---|---|---|---|
| 2 | masked | 55.7 | 60.4 | 0.333 | 0.229 |
| 2 | unmasked | 6.3 | 6.0 | 0.921 | 0.921 |
| 2 | all | 6.7 | 6.4 | 0.882 | 0.875 |
| 12 | masked | 55.7 | 30.4 | 0.333 | 0.167 |
| 12 | unmasked | 6.3 | 5.0 | 0.921 | 0.921 |
| 12 | all | 6.7 | 5.2 | 0.882 | 0.871 |

注意：这 6 条序列与上面 15 条的结果方向相反（gap12 在 6 条子集上 masked 反而更好），
说明「补全是否改善精度」没有稳定结论；小样本子集不能作为调参依据。

### 7.3 结论修正

- 端侧默认值保持 `maxCoastFrames = 5`，但**口径要诚实**：补全不是精度提升，而是
  「缺一个颜色时仍给出一帧位姿」的连续性降级；连续遮挡超过约 3 帧后误差增长明显。
- 若上游检测器能给出「该颜色完全缺失」的可靠信号（`spots.size == 2`），补全作为兜底是安全的；
  若只是「位置偏差/证据弱」，补全反而会把可用的部分测量覆盖成外推结果（见第 3 节），必须避免。
