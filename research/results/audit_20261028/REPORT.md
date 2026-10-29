# TriDotPnP 学习方案诊断报告（2026-10-28）

审计对象：`research/tridot_temporal`（FindNet）、`research/results/*` 全部实验报告（46 份）、`datasets/TriDotPose`。
目的：回答"为什么学习方案只比经典 CV（B0）强一点点"、"是不是训练集/评测集不合适"，并给出可执行路线。

---

## 0. 结论

1. **在能公平比较的集合上（easy = `splits.json:test`），B0 = 0.918 det@10，最好的学习模型 = 0.960（+4.2 pt）。** 10 次同集对比里 4 次学习 ≤ B0。B0 在 blur/clean/light 三个 factor 上已是 1.000，学习增益几乎全部来自 `distractor`（0.679 → 0.876）。
2. **hard_test 上 0.245 → 0.884 的"大胜"不是能力差距，而是分布 + 选型造成的**：hard_test 与 hard_train 同 preset、同 generator（v1.1.0），B0 从未针对 realistic 分布适配；且 hard_test 先于训练块生成，之后每次调参都在它上面选型 → 0.884 是调参集最优值，不是泛化估计。
3. **transformer 时序路径贡献为 0**：消融 `coords = 0.000`、`coords0 = 0.152`、`geom = 0.833`。真正带来提升的是 5,249 参数的 GeomTripleNet 对 125/343 个**手工候选三元组**重打分，加上 bicubic upsample + 9×9 softmax 质心 + 峰值 NMS。所谓"学习系统"约 90% 仍是手工 CV。
4. **训练集/评测集确实不合适，且可量化**：train 有 90% 的帧接近全黑（mean V 21/255、1 点/色），hard_test 是带纹理/光照、3–5 个同色干扰块的场景（mean V 71.5、Laplacian 方差 29×、饱和像素 59×）；9 个 hard factor 在 train 中 **0 帧**，7 个旧 factor 在 hard_train 中 **0 帧**；两侧连 generator 版本都不同（v1.0.0 vs v1.1.0）。
5. **没有任何真实相机数据**：`real_test` 空、`real/annotations` 空、唯一的"真机"视频 `smoke_real_desk.mp4` 实为程序化渲染，`real/videos` 另外 3 条是合成序列的重新编码。因此"经典 CV 不够用"和"学习更强"目前都没有真实数据支撑。

---

## 1. "只强一点点"到底差多少（同一张表）

### easy（`splits.json:test`，15 序列 / 1800 帧 / 1244 帧三点全可见）

| 方法 | det@10 | det@10_all | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|
| B0（本次从 pyc 恢复后复跑） | **0.918** | 0.655 | 35.00 | 57.4 mm | 74.9 |
| FindNet_geom 最好（`find_v3_6_gpu`） | **0.960** | — | 13.20 | 41.3 mm | 143.3 |
| Research_TripleRank（classic proposal） | 0.974 | — | 18.77 | 28.1 mm | 81.9 |

B0 分 factor（本次复跑）：

| factor | blur | clean | light | noise | mixed | distractor | occ |
|---|---:|---:|---:|---:|---:|---:|---:|
| det@10 | 1.000 | 1.000 | 1.000 | 0.920 | 0.926 | **0.679** | 0.025 |
| det@10_all | 0.921 | 0.988 | 0.858 | 0.912 | 0.367 | 0.683 | 0.000 |

→ easy 上经典 CV 已贴着合成数据天花板；学习唯一的战术价值是彩色干扰块。

### hard（`splits.json:hard_test`，40 序列 / 3600 帧 / 2527 帧三点全可见）

| 方法 | det@10 | det@10_all | det@10_cut | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|---:|
| B0（本次复跑，与历史报告逐位吻合） | **0.245** | 0.175 | 0.010 | 236.29 | 485.6 mm | 11.4 |
| FindNet_geom 零样本（v3.6） | 0.041 | — | — | 285.32 | 2213.5 mm | 147.6 |
| + hard_train 0300s（v3.7） | 0.663 | — | — | 55.67 | 176.3 mm | 151.8 |
| + 0400s（v3.8） | 0.787 | — | — | 32.23 | 71.9 mm | 143.8 |
| **v3.12（历史最佳）** | **0.884** | 0.641 | 0.188 | 22.36 | 58.7 mm | 137.4 |
| + 0500s tail（v3.13） | 0.770 | — | — | 16.85 | 47.3 mm | 118.8 |
| 10-18 重测 plain geom（`ablate_v312_vel`） | 0.833 | 0.641 | 0.188 | 31.94 | 107.1 mm | 143.5 |

B0 分 factor（本次复跑）：room 0.780、far 0.297、handheld 0.245、crosscolor 0.223、occlude_hand 0.222、desk 0.214、clutter 0.075、screen 0.003、hardmix **0.000**。

关键观察：
- 0.041 → 0.884 的爬升全部对应**同分布训练数据的加入**（0300s / 0400s / 0500s 三批），没有一次是架构改动。
- 加入 0500s tail 后 det@10 反而 0.884 → 0.770，而 pix 22.4 → 16.9：两个阶段是在 precision 与 recall 之间移动，很可能落在 4 场景/factor 的噪声里。
- 0.884（v3.12）与 0.833（10-18 同名义集合重测）**无法解释**：10-18 目录只有 REPORT/STATUS，没有 per_frame.json、没有 ckpt、没有复现命令。

---

## 2. 为什么 transformer 完全没起作用（代码级）

`research/tridot_temporal/find_model.py`：
- 编码器 5 层 depthwise CNN，stride 4，输出 3×64×112 热图（720p 下 **11.4 px/格**），**没有 offset/回归头**；亚像素精度靠后处理 `upsample_heat`（bicubic ×2…×7）+ 9×9 softmax 质心。
- `heat = heat_head(feat) + aff_to_heat(avg_pool2d(x[:,3:6],4))`，`aff_to_heat` 以 identity×3.0 初始化 —— 热图本身一半是手工 HSV 亲和通道的直通。
- 时序头只吃 5 帧 × 每帧 (6 坐标 + 3 vis + GAP) token，输出残差 `coords = clamp(coords0 + 0.06*tanh(dcoord), 0, 1)`：最大修正量约 77 px（1280 宽），实际 tanh 输出方差小 → 近似 identity。
- 损失 `lambda_xy0 = 20` 远大于 `lambda_xy = 8`，而评测冠军用 peak/geom 解码，几乎不用 coords → 时序头拿不到"有用"的梯度信号。
- 消融：`coords 0.000 < coords0 0.152 < vel 0.345 < local 0.626 < geom 0.833`；`--repeat-frame`（去掉时序）只掉 1–2 个点。

结论：5 帧内点位移用 3 个坐标 token 表达信息量太少，transformer 没有可学的东西；有效的时序增益应来自轨迹关联/滤波，而非坐标残差回归。

---

## 3. 数据集/评测集的问题（量化）

### 3.1 训练与评测分布几乎正交

| 指标 | train（full） | hard_train | hard_test |
|---|---:|---:|---:|
| mean V (p50) | **21.4** /255 | 75.6 | **71.5** |
| 灰度 σ（纹理） | **4.14** | 17.98 | **12.99** |
| Laplacian 方差 | **31** | 1,062 | **906（29×）** |
| 饱和像素占比 | **0.003** | 0.152 | **0.178（59×）** |
| 全黑帧（σ<8）占比 | **90.0%** | 5.9% | **5.0%** |
| 每帧红/绿/蓝斑块数 | 1/1/1 | 4/4/4 | **3/3/3** |
| 点半径中位数 | 8.06 px | 6.15 px | 6.48 px |
| 点 SNR 中位数 / <3 占比 | 26.1 / 2.8% | 12.6 / 25.3% | **15.3 / 19.4%** |

- **9 个 hard factor 在 train 中 0 帧；7 个旧 factor 在 hard_train 中 0 帧**；`full` 用 v1.0.0 生成器，`realistic*` 用 v1.1.0。
- 默认 `--train-subset train`（`train_find.py:89-92`）→ 所有 hard 数字都是 zero-shot；v3.6 的 0.041 正因此产生。
- 最差因子：`screen`（Laplacian 方差 63,510，**86% 的点 SNR<3**，信息论上不可检测）、`far`（点半径 p50 3.99 px、p10 2.46 px、z≈1.6 m）、`hardmix/clutter/desk/crosscolor`（在点旁 3–7 px 放 40–80 px 的**同色实心矩形**）。

### 3.2 评测口径

- `det@10` 分母恒为"三点全可见帧"（`eval_find.py:449` `denom = [r for r in rows if r.gt_all_visible] or rows`），hard_test 3600 帧只有 2527 帧入账。
- 遮挡因子有**兜底分母**（`eval_find.py:457` `denom = [...] or rs`）：`occlude_hand` 没有一帧三点全可见（标注规则 vis=1 保留完整 GT），于是它实际在"只剩 2 个点"的帧上评分 → 表里 0.717，`det@10_all` 只有 0.111；`hardmix` 0.605 vs 0.099。
- `pix_rmse` 只在检测成功帧上统计；干净帧上 B0 0.9 px、学习 3.1 px → 学习侧"精度更高"部分是幸存者偏差。
- 同一 ckpt（v3.12）：det@10 0.833 / det@10_all 0.641 / det@10_cut 0.188。旧报告没有 all/cut 列，读者会误以为 det@10 = 全帧。
- **分辨率不匹配**：5 px @1280×720 = **1.75 px @448×256 输入**，该输入下点半径中位数仅 2.3 px（far 1.4、screen 1.3）；热图 stride 4 时一格 = 11.4 px 原始像素。用这个表示承诺 <10 px（更别说 5 px）本身不成立，所以 det@10 呈双峰（0.04 或 0.88）。

### 3.3 hard_test 是调参集

- 02xx 评测块先于训练块生成（`created_utc`：02xx 10-11T11，训练块 T12/T14/T16）；之后 v3.7→v3.24 每次选型都在这同样 40 条序列上。
- 每个 factor 只有 4 个场景（hardmix 8），360 帧/因子，方差极大；"clean"/"repeat"/"zero-train" 三个集合名在任何报告里都没有定义（"clean" 与 hard 帧号、可见性完全一致，只是不同模型的重复评测）。

### 3.4 实验卫生

- `train_find.py` 无 `manual_seed`/`worker_init_fn`/`generator`；4–8 epoch 微调 + `val_l2` 容差 4% 的 ckpt 选择 → v3.10–v3.24 的排序很可能在噪声内。
- `find_dataset.py:56-66` 对 hardmix 过采样 24×、occlude_hand 18×、screen 14×：~2.6k 唯一 clip 膨胀到 ~35k draws/epoch（13× 重复，只复制不生成）。
- 报告无 `--subset`/ckpt/flags 记录；`eval_find.py:581-584` 会用 geom ckpt 覆盖命令行 `top_k/upsample`；`eval_find.py:741` 每次覆盖 `research/phase5_find_v3_report.md`；多份 REPORT 互为副本（`results/REPORT.md` = `find_v3_12_hard`，`REPORT_A/B/D` = `6a/6b/6d`）。
- 代码领先于报告：v3.19–v3.24 + peak FT 只存在于 `/home/lenovo` 的脚本与 `find_v3_peak/train.log`，本地无 ckpt、无结果。
- 关键源码（`baseline_b0`、`baseline_b1`、`tridot_synth`、`label_tool`）只剩 `__pycache__`；`train_find_geom` 模块不存在。

---

## 4. 本次独立验证

用 `importlib.machinery.SourcelessFileLoader` 从 `research/baseline_b0/__pycache__/*.cpython-311.pyc` 恢复 `B0Detector` / `PoseSolver`（解释器 3.11.15，magic a70d0d0a 匹配），注入 `sys.modules` 后直接调用 `eval_find.main`，对 B0 做完整重跑：

| 集合 | det@10 | det@10_all | pix_rmse | t_err | FPS |
|---|---:|---:|---:|---:|---:|
| easy test | 0.9180 | 0.655 | 35.00 | 57.44 mm | 74.9 |
| hard_test | 0.2450 | 0.175 | 236.29 | 485.65 mm | 11.4 |

与历史报告（0.918 / 35.00 / 57.4；0.245 / 236.29 / 485.6）逐位吻合 → **B0 数据与评测流程可复现，历史对比基线可信**。同时确认 B0 在 hard 上并非"坏掉"，而是固定 HSV 阈值/几何打分在 realistic 渲染上失效（hardmix 0.000、screen 0.003），但并非不可改进（room 0.780、far 0.297）。

产物：`b0_easy/`、`b0_hard_test/`（含 per_frame.json）、`b0_pyc_boot.py`、`run_b0_eval.py`。

---

## 5. 更好的方案（按优先级）

### 5.0 先换量尺

当前所有数字都在优化 `det@10`（合成数据上的像素命中率），产品要的是位姿/距离精度和真机鲁棒性。建议主指标：

| 指标 | 定义 | 现状（easy） |
|---|---|---|
| t_err (mm) | PnP 平移误差中位数 | B0 57.4 / 学习 41.3 |
| dist_rel | 距离相对误差 | — |
| ang_err | 方位/俯仰误差（°） | — |
| det@all | 全帧检出率（遮挡算失败） | B0 0.655 / 学习 0.641(hard) |
| FP/frame | 假阳点数 | — |
| jitter | 静止时距离/角度的帧间 σ | 未测 |
| recovery | 遮挡后恢复帧数 | 未测 |

### 5.1 建真实评测集（最高优先，1–2 天）

- 50–100 段手机视频（桌面、屏幕、手持、远近、反光、逆光、卷帘、遮挡），用改造后的 B0/现有模型预标注 + 人工修正（`label_tool` 已丢失，需重写，约 200 行）。
- 冻结 `real_test`，从此只报告、不再调参；每因子 ≥100 帧，报均值 ± std。
- `datasets/TriDotPose/real/videos` 里 3 条是合成序列的重新编码，不能当真实集。

### 5.2 修数据管线（1 周）

- 相机模型换成真机标定结果（含径向/切向畸变）；现在合成器是理想 pinhole、零畸变（fx=fy=1108.5）。
- 合成器参数已足够丰富（`motion_blur_ksize`、`noise_sigma`、`jpeg_quality`、`lighting_scale`、`color_cast_bgr`、`flicker_amp`、`crosscolor_patches`、`moving_occluder`、`bg_style`），问题在**采样分布**：train 是近黑背景 + 1 点/色，应改为真实场景同分布（桌面/屏幕/房间、多点、反光），并按真实相机的噪声/模糊/曝光统计重新采样。
- 去掉 factor_w 的 24× 复制式过采样，改为每 epoch 重新随机化 corruption（才有新样本）。
- 修标注不一致：遮挡帧统一标 vis=0（或 vis=1 但评测时排除），给 <3 点的帧加 unscorable mask。

### 5.3 网络结构改对（1–2 周）

1. **分辨率**：不要在 448×256 上做终检。两级：640×360 粗检 → ROI 128×128 在原分辨率精修；或直接单阶段 stride-2 640×360。
2. **检测头**：CenterNet 式 heatmap(3 通道) + sub-pixel offset(2) + vis/score，focal loss；去掉 coords0 全局 soft-argmax 作为精度来源。
3. **时序**：删掉坐标 token transformer；改成 per-frame 检测 + 3 条独立轨迹 + 数据关联（匈牙利/最近邻）+ Kalman 或轻量 GRU 外推。若要用 attention，token 用 heatmap patch 或 ROI 特征，而非 3 个坐标。
4. **端到端目标**：把可微 PnP（或直接回归 6DoF + 重投影 loss）接在检测之后，让 loss 直接优化 t_err/角误差；现在的 `lambda_xy0=20` 与评测解码完全脱节。
5. **保留并产品化经典部分**：三角形边长/角度一致性、颜色顺序、重投影残差、双解消歧——这是当前唯一真正有效的模块，应做成显式的可微验证分支，而不是散落在解码器里的十余个手工 gate（jump-rescue/repeak/heat-hold/occ-coast/inject-expect 等实际上是在拟合 40 条序列）。

### 5.4 部署（与算法并行）

App 目前没有任何模型推理。建议 TFLite（INT8，NNAPI/XNNPACK）或 NCNN，640×360 输入，目标 <15 ms/帧；先做真机 A/B：经典 vs 学习在 t_err/抖动/漏检上的差异。

### 5.5 最短路径

1. 本周：恢复 B0 源码（已可从 pyc 还原）+ 真机采集 + 新指标脚本 + 重写标注工具 → 拿到"现在到底差在哪"的真实数字。
2. 下周：B0 + 高分辨率 CenterNet proposal 的混合模型（不做 transformer），在真实集上对齐 t_err。
3. 之后：可微 PnP + 轨迹滤波，把 t_err 目标定到 <20 mm @1 m。
4. 最后：TFLite 部署 + 真机闭环 A/B。

---

## 6. 不建议做的事

- 不要继续在 hard_test 上调参、加手工 gate（46 份报告里的增益多半是这类拟合）。
- 不要先重训更大的 transformer：数据与指标不解决，架构改动无法验证。
- 不要把报告里的 FPS/排名当结论（无种子、无 provenance，同模型 FPS 差 2×）。
- 不要把 `real/videos` 里的合成拷贝当真实验证集。
