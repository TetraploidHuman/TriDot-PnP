# 真机远距验证协议（GPU/真实数据到位前先写好，到位即执行）

目标：用**同一份 APK、同一套评测口径**回答「多远开始失效」「退化模型是否低估了真实难度」「E 的采集开关是否真的有用」。

## 0. 硬件与准备

- 手机：目标机型（当前开发机 Sony Xperia，含两段式快门键）；固定 60° FOV 假设（`estimatedFovDegrees` 默认 60，需先在 App 里用已知靶标标定一次并记录实际值）。
- 靶标：边长 100mm 的 RGB 三点板（与合成集一致），贴在刚性平板上，板面做哑光处理避免反光。
- 距离：1 / 2 / 3 / 5 / 8 / 10 m 六档（用激光测距仪记录实际距离，精度 ±5mm），每档 3 种姿态（正对 / 绕 y 轴 ±30° / 绕 x 轴 ±20°）。
- 光照：三种环境各录一次——室内白光（300–500 lux）、室内暗光（<50 lux）、屏幕/窗户逆光。
- 相机设置每组固定：默认自动；再各录一遍 `captureHighResAnalysis=on`、`captureNoiseReductionOff=on`、`captureLockWhiteBalance=on` 三项全开。
- 每个条件录 10–15 s（300–450 帧 @30fps），并同时用卷尺/激光记录真值距离与角度（或用标定板做位姿真值）。

## 1. 采集清单（每组一行）

| 距离 | 姿态 | 光照 | 采集开关 | 文件名 |
| --- | --- | --- | --- | --- |
| 1–10 m × 3 × 3 × 2 | | | 默认 / 三项全开 | `real_<dist>m_<pose>_<light>_<flags>.mp4` |

## 2. 现场要记录的数

1. 首帧画面里三点的像素坐标（从 App 的 DEBUG 导出，或用帧截图人工点）。
2. App 显示的 DISTANCE/AZIMUTH/ELEVATION 与 6DOF（每 10 帧记一次值）。
3. 失败帧（`refinedSpots.size != 3`）的比例；`farDotAssist` 是否触发。
4. FPS 与每帧耗时（App 面板已有 FPS）。

## 3. 分析脚本（真实视频到位后直接跑）

- 无标注：`research/scripts/real_video_probe.py`（参考帧选最可信的一帧，量点漂移、视边长误差、退化下的保持率）。
- 有标注：用 `research/label_tool`（已重建，`python -m research.label_tool --port 8899`）标 3 点 → export 写入 `datasets/TriDotPose/real/annotations/` 与 `splits.json` 的 `real_test`/`real_train` → `eval_v2 --subset real_test --methods b0,classic2,classic3`（以及 `--degrade` 做「真实 vs 退化合成」的对照）。
- 真机对照表要回答：det@10 all、逐点 recall@10、pose coverage、t_err 中位/p90、距离相对误差、延迟。

## 4. 判定标准（提前写死，避免事后挑数）

- **可用**：1–5m 内 det@10 all ≥0.8、距离相对误差中位 ≤5%、t_err 中位 ≤30mm。
- **临界**：5–8m 内 det@10 all ≥0.5，且不得出现「检出三点但距离错 >20%」的假阳性（当前合成退化下 classic2 在 phone_far 会锁到干扰块上，正是一类假阳性）。
- **失效**：≥8m 或 screen/逆光场景，明确记录「不可用」并给出需要的外设（更高分辨率/长焦/主动光源）。

## 5. E 的三项采集开关的 A/B

同一场景同一时刻，开关单独切换各录一遍（其他不变），比较：
- `captureHighResAnalysis`：分析分辨率 1920×1080 vs 1280×720 → 5–10m 的 det@10 all；
- `captureNoiseReductionOff`：ISP 降噪开/关 → 1–2px 点是否被抹掉（用逐点 recall@10 判断）；
- `captureLockWhiteBalance`：AWB 锁定 → 色差图峰值稳定性（帧间 `mean color diff` 抖动）。

## 采集后验收（一键）

```bash
# 有标注之后
PYTHONPATH=. research/.venv/bin/python -m research.scripts.real_eval_acceptance
# 还没有标注时先预检画面/检测稳定性
PYTHONPATH=. research/.venv/bin/python -m research.scripts.real_eval_acceptance \
    --video datasets/TriDotPose/real/videos/<clip>.mp4
```

阈值硬编码在脚本里（1–5 m det≥0.80 / 距离误差≤5% / t_err≤30 mm；5–8 m det≥0.50 且无「检出但距离错 >20%」帧；≥8 m 仅记录），并有用例锁定，避免事后改判据。

## 验收脚本自检（合成替代 dry-run）

真机数据未到时，可先用合成子集验证「验收脚本本身」工作正常（阈值不变，但距离段落在 1–5 m 才有判定）：

```bash
# 预期 PASS（v3+分类器，clean）
PYTHONPATH=. research/.venv/bin/python -m research.scripts.real_eval_acceptance \
  --subset holdout_hard --limit-seq 6 \
  --dot-cls research/results/dot_cls/dot_cls_scaled.pt --dot-cls-scale \
  --out research/results/acceptance_dryrun/pass.json

# 预期 FAIL（classic2 + phone_far）
PYTHONPATH=. research/.venv/bin/python -m research.scripts.real_eval_acceptance \
  --subset holdout_hard --limit-seq 6 --frontend v2 --degrade phone_far \
  --out research/results/acceptance_dryrun/fail.json
```

`<1 m` 的帧会被标为 `IGNORED`，不计入 PASS/FAIL（真机协议约定的距离从 1 m 起）。
