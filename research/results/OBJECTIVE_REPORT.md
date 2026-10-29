# 远距/小光斑检测：objective 总报告（A–F）

> 目标：解决远距离（≤11 px 视边长、1–2 px 光斑）检测失效。先建「真实退化 + 距离」评测（F），
> 再实现局部对比度检测（A）、尺度自适应（B）、检测-颜色解耦（D）、ROI 上采样与亚像素（C）、
> 相机端采集优化（E），在退化合成集与真机/APK 上给出前后对比证据。
>
> 状态（冻结于 2026-10-29，round 28）：**合成域全部完成并有证据（A–F + 端侧链路，131 Python / 44 Kotlin 测试全过）；
> 真机域因缺硬件尚未验证**——这是唯一未完成项（协议与一键验收脚本已就绪，数据到位即可出 PASS/FAIL）。

## 1. 主结果：hard_test 12 序列 × 4 档退化（1080 帧/档，det@10 all）

| 退化档 | b0 | classic2 | v3 union | **v3 union + dot-cls** |
| --- | --- | --- | --- | --- |
| clean | 0.1676 | 0.8148 | 0.8241 | **0.8806** |
| phone_easy | 0.2935 | 0.3611 | 0.8139 | **0.8454** |
| phone_far | 0.4259 | 0.3398 | 0.7769 | **0.8380** |
| phone_harsh | 0.4250 | 0.3361 | 0.5463 | **0.6889** |

小点桶（40–80 px 视边长，即 1.4–2.8 m）：

| 退化档 | classic2 | v3 union | v3 + dot-cls |
| --- | --- | --- | --- |
| clean | 0.369 | 0.662 | **0.754** |
| phone_easy | 0.062 | 0.523 | **0.600** |
| phone_far | 0.108 | 0.292 | **0.369** |
| phone_harsh | 0.015 | 0.200 | **0.231** |

读法：v3 前端把「退化域」从「几乎不可用」拉到可用；dot-cls 在四档上一致再 +2.7 ~ +14 pt；
clean 档也不掉（+5.7 pt），说明收益不是靠适配退化。

## 2. 未见序列泛化（holdout_hard 12 序列，训练从未见过）

| 退化档 | classic2 | v3 union | v3 + dot-cls |
| --- | --- | --- | --- |
| clean | 0.7250 | 0.7583 | **0.8019** |
| phone_far | 0.1083 | 0.6444 | **0.7861** |

## 3. 采集侧（E）与该做的取舍

hard_test 6 序列、phone_far、classic2 / v3 两行：

| 配置 | classic2 | v3 union |
| --- | --- | --- |
| baseline | 0.367 | 0.656 |
| 4:4:4 色度（chroma444） | **0.659** | **0.746** |
| 提高分析分辨率（只去降采样） | 0.324 | 0.667 |
| 关 ISP 降噪 | 0.389 | — |
| JPEG=100 | 0.417 | — |
| 全开 | 0.524 | 0.452 |
| 4:4:4 + 高分辨率 + 分类器 | — | **0.8185** |

结论：色度 4:2:0 下采样是远距失效头号外因；分辨率只有与 4:4:4 同时打开才有效；「全开」反而更差
（关降噪/关 JPEG 会把色度优势抵消）。App 侧的开关实现见 §4.8 / `CaptureRequest.NOISE_REDUCTION_MODE`。

## 4. 负结果清单（都记录在报告里，避免重复踩坑）

1. 评测口径：`det@10 legacy` 在遮挡因子下虚高 3 倍（0.1667 vs 0.0561 observable）；
2. 阈值自适应（tile / tile_rel / snr）：与百分位阈值逐帧相同（有效阈值都停在常数底 12），假设证伪；
3. 候选级时序门控（prune）：四档全掉（clean −0.030 … all_on −0.041），软时序约束不能变硬；
4. 形状无关匹配滤波（union_mf）：半遮挡召回 +0.017，退化域显著掉点；
5. 三元组打分的圆度奖励：clean/far 都变差（2–4 px 上圆度=噪声）；
6. 2 点 + 已知几何（运动先验 / 平面法向先验 / completion）：遮挡帧是**观测上限**，
   平面先验高覆盖但 t_err 中位 551 mm，比不输出更糟；
7. 更早的 transformer offset 头/分辨率路线同样以负结果告终（记录在 transformer_ab 报告）。

## 5. 交付物索引

- 报告：`research/results/classic_cv3/REPORT.md`（§2 距离换算、§3 A–D 实现、§4.1–4.12 证据、§5 真实视频、§6 局限）
- 代码：`research/tridot_temporal/{degrade,classic_v2,classic_v3,dot_cls,pnp2,temporal,completion,metrics,eval_v2}.py`
- 端侧：`FarSpotDetector.kt`、`DotPatchClassifier.kt`、`PnPTemporalTracker.kt`、`CameraPreview.kt`（采集开关）、`CameraFrameProcessing.kt`（主线三点自洽性校验 `TripleConsistency`）
- 资产：`composeApp/src/androidMain/assets/dot_cls/{dot_cls.bin,dot_cls.json}`
- 数据：`research/results/far_bench/*.json`、`research/results/dot_cls/*.json`、`research/results/observable/*`
- 协议：`research/docs/real_device_protocol.md`；一键验收：`research/scripts/real_eval_acceptance.py`
- APK：`composeApp/build/outputs/apk/debug/composeApp-debug.apk`（68.8 MB，含远距辅助 + 分类器 + 时序滤波 + PnP 修复）

## 6. 复现命令

```bash
# 退化域矩阵（一行一档）
PYTHONPATH=. research/.venv/bin/python -m research.scripts.far_bench \
  --subset hard_test --limit-seq 12 --degrade phone_far \
  --methods b0,classic2,classic3_union --v3-configs union:12:union \
  --dot-cls research/results/dot_cls/dot_cls_scaled.pt --dot-cls-scale \
  --out research/results/dot_cls/far_phone_far_dotcls12.json

# 全量测试
PYTHONPATH=. research/.venv/bin/python -m unittest discover -s research/tests

# 真机验收（数据到位后）
PYTHONPATH=. research/.venv/bin/python -m research.scripts.real_eval_acceptance
```

## 7. 诚实局限

- 全部数字来自**合成退化**，退化的方向可信但幅度不可承诺（真机还有 AE/AWB 漂移、卷帘、镜头污渍等未建模）；
- 唯一真实视频是近景桌面（60 帧），只能做内容级预检，不能代表远距；
- 遮挡帧（`occlude_hand` 0 帧三点全可见）对该传感器/前端是观测上限，不是算法缺陷；
- 端侧已把 A–D 接入主线（`farDotPrimary=true` 默认：对比度前端可信时直接用它，不可信才回退原 BrightSpotDetector）；
  性能代价约 +75 ms/帧，可在「采集与远距」里关掉恢复旧语义；
- 真机验收阈值（1–5 m det≥0.8 / 距离误差≤5% / t_err≤30 mm；5–8 m det≥0.5）是**采集前写死**的，不应事后调整。

## 8. 最终状态与回归证据（round 25）

### 8.1 回归（2026-10-29）

| 项 | 结果 |
| --- | --- |
| Python 测试 | **131 tests OK**（`research/tests/`，8.7 s） |
| Kotlin 单测 | **39 tests，0 失败**（8 个测试类：tracker / far / subpixel / chooser / consistency / dot-cls / fixtures / pnp-free-scale） |
| 构建 | `:composeApp:testDebugUnitTest :composeApp:assembleDebug` BUILD SUCCESSFUL（离线 gradle 8.14.3） |
| APK | **68,853,225 B**，sha256 前缀 `03c1b0454206f777` |
| dex 符号 | `PnPTemporalTracker`28 / `FarSpotDetector`40 / `DotPatchClassifier`10 / `DotPatchExtractor`2 / `TripleConsistency`2 / `TripleChooser`2 / `refineSubpixel`2 / `scanTriples`2 / `buildFarTripleValidator`2 / `posesForTriple`1 / `farDotPrimary`2 / `captureHighResAnalysis`2 / `temporalFilterEnabled`2 |

### 8.2 A–F 逐项最终状态

| 项 | 交付 | 关键证据 | 状态 |
| --- | --- | --- | --- |
| **F 退化+距离评测** | `degrade.py`（clean/easy/far/harsh）、`far_bench.py`（<20/20–40/40–80/80+ 分桶）、`test_degrade.py` 9 测试 | 四档矩阵 + 分桶表 | ✅ |
| **A 局部对比度** | `classic_v3.py` 对比度前端（自适应核 + MAD/相对阈值） | clean 0.8241 / phone_far 0.7769（12 序列，vs classic2 0.81/0.34） | ✅ |
| **B 尺度自适应** | 由上一帧位姿推 `r_px`/核尺寸；无位姿时多尺度回退 | 合成 1–2 px 点检出 <3 px；`test_classic_v3.py` | ✅ |
| **C ROI 上采样 + 亚像素 | 4× 上采样质心（Python）+ 抛物线亚像素（Kotlin，≤3.5 px 触发） | 触发率 ~0%（当前光斑 8–27 px）→ 保留但如实标注「窄触发」 | ⚠️ 实现完成、收益可忽略 |
| **D 检测-颜色解耦 | 灰度候选 + 颜色合理性否决；色度丢失时仍可定位 | 合成测试 <2 px；远距/退化帧回收 | ✅ |
| **E 采集侧** | App 三开关（高分辨率 / 关 ISP 降噪 / 锁 AWB）+ 采集 A/B 归因 | 4:4:4：harsh 0.600→0.804；端到端 0.52→0.653 | ✅ |
| dot-cls（额外） | 15×15 候选分类器（23.6 k 参数，尺度归一化 patch） | 四档 +2.7~+14 pt，clean 也不掉；holdout +4.3~+14 pt | ✅ |
| 端侧接入 | `farDotPrimary` + `TripleChooser` + `TripleConsistency` | 链路模拟 clean 0.41→0.77、t_err 61→19 mm | ✅ |
| 成本/调度 | `FarAssistScheduler`（帧间隔 1–4，默认 1）+ `farFrontendMaxDim`（448–640） | 跳帧 N=2 掉 19 pt（禁用）；512 分析长边掉 2–3 pt、省 ~36% 像素 | ⚠️ 部分降本 |

### 8.3 结论

- 合成域目标（1–2 px 光斑、2.8–5.5 m 等效）已达到：v3+分类器在 clean/phone_far/phone_harsh 上 0.88/0.84/0.69
  （det@10 all），未见序列 holdout 0.80/0.79，端到端链路 0.65–0.77；
- 真机域**未验证**是整个 objective 唯一未完成项，验收脚本与前置阈值已就绪，数据到位即可出 PASS/FAIL；
- 性能与精度已显式取舍（+75 ms/帧 换 det +0.36 / t_err −70%），可通过开关回退。

## 9. 冻结交付（round 28）

### 9.1 版本与回归

| 项 | 值 |
| --- | --- |
| 冻结时间 | 2026-10-29（Asia/Shanghai） |
| APK | `composeApp/build/outputs/apk/debug/composeApp-debug.apk`，**68,858,341 B**，sha256 `a8000490629313f1…` |
| Python 测试 | **131 passed**（`research/.venv/bin/python -m unittest discover -s research/tests`，~9 s） |
| Kotlin 测试 | **44 passed**，0 failed（8 个测试类，含远距前端 3 档分辨率、自洽门、选择器、调度器、分类器一致性） |
| 构建 | 离线 gradle 8.14.3，`:composeApp:testDebugUnitTest :composeApp:assembleDebug` BUILD SUCCESSFUL |
| 关键数字抽检 | clean dot-cls 0.8806 / harsh base 0.600 / harsh 4:4:4+高分辨率 0.8037 / 链路 clean 0.7695 / 跳帧 k=2 0.5771 全部在 JSON 中一致 |

### 9.2 交付索引

| 你要的东西 | 在哪 | 怎么验证 |
| --- | --- | --- |
| 结论与全部证据 | `research/results/classic_cv3/REPORT.md`（§0–§4.18） | 读 §1 矩阵、§4.16 harsh、§4.13–4.17 链路与成本 |
| 一页总账 | `research/results/OBJECTIVE_REPORT.md`（本文件） | §1 四档矩阵、§2 泛化、§4 负结果、§8 状态 |
| 退化/距离评测 | `research/tridot_temporal/degrade.py` + `research/scripts/far_bench.py` | `--degrade phone_far --subset hard_test` |
| 对比度前端 | `research/tridot_temporal/classic_v3.py`（`frontend="union"`） | `test_classic_v3.py` |
| 候选分类器 | `dot_cls.py` + `research/results/dot_cls/dot_cls_scaled.pt` + 导出 `assets/dot_cls/*` | `test_dot_cls.py`、Kotlin `DotPatchClassifierTest` |
| 端侧实现 | `FarSpotDetector.kt` / `DotPatchClassifier.kt` / `CameraFrameProcessing.kt`（自洽门 + 选择器 + 调度） | Kotlin 44 测试 + APK dex |
| 真机协议 | `research/docs/real_device_protocol.md` | 采集矩阵与判定阈值（前置写死） |
| 一键验收 | `research/scripts/real_eval_acceptance.py` | `--video` 预检；scored 模式读 `real_test` |
| 标注工具 | `research/label_tool`（服务端 pyc 直载 + 重建前端） | `python -m research.label_tool --port 8899` |

### 9.3 真机验收流程（数据到位后照做）

1. 按协议拍视频（1/2/3/5/8/10 m × 3 姿态 × 3 光照，手机原生相机）；
2. `label_tool` 抽帧标注 3–5 个距离段（每段 ≥10 帧）→ 导出到 `datasets/TriDotPose/real/annotations`；
3. `PYTHONPATH=. research/.venv/bin/python -m research.scripts.real_eval_acceptance` → 分段 det / 距离误差 / t_err 与 PASS/FAIL；
4. 若 FAIL，按 §7 局限与 §4.2/§4.16 的采集归因逐项排查（先看色度/分辨率，再看法前端召回）。

### 9.4 阻塞声明

- 合成域的四项交付（矩阵、泛化、采集归因、链路与成本）全部完成并有可复跑数字；
- **真机域未验证**，原因是本机没有可用的「三色三角靶标 + 距离已知」的真实采集素材（唯一真实视频经检查无稳定靶标，不能用于精度验收）；
- 该阻塞与算法无关，只等硬件/数据；在此之前不建议把任何真机精度数字写入对外材料。

## 10. 验收脚本自检（round 29）

真机数据未到，先证明「验收脚本本身」在 PASS/FAIL 两个方向都会正确判定——用合成 `holdout_hard` 6 序列做 dry-run
（阈值与前文完全一致；`<1 m` 帧标 `IGNORED`，因为真机协议从 1 m 起算）：

| dry-run 配置 | 1–5 m 帧数 | det@10 | 距离相对误差 | t_err 中位 | 判定 |
| --- | --- | --- | --- | --- | --- |
| v3+分类器，clean（预期 PASS） | 83 | **1.000** | **0.4%** | **5.8 mm** | **PASS** |
| classic2 + phone_far（预期 FAIL） | 83 | 0.000 | 82.8% | 1108 mm | **FAIL** |

两份结果分别在 `research/results/acceptance_dryrun/{pass,fail}.json`。加上
`test_real_acceptance.py` 新增的 3 个测试（band 边界、`<1 m` 无判定、合成子集路径），验收链路在数据到位前已完整自检。

