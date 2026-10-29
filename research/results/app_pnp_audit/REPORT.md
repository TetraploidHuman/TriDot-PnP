# App 自带 PnP 深度解算器审计与修复

日期：2026-10-28
审计对象：`composeApp/src/androidMain/kotlin/org/example/tridotpnp/PnPDistanceCalculator.kt`
数据：`datasets/TriDotPose`（synthetic，test 1244 帧、hard_test 2527 帧，均只取三点全可见帧）

## 一句话结论

App 的 `solve3PointDepths` 不是"精度略差"，而是**数学上发散**：即使喂进完美的 GT 像素和与 GT 完全一致的焦距（60° FOV → fx=1108.51），它给出的距离中位误差是 **54%（test）/ 62%（hard_test）**，平移误差中位 **457 mm / 576 mm**，只有 0.8% / 4.3% 的帧收敛。修复成 3 未知数深度 + 解析 Jacobian + LM 阻尼后，同样输入下距离误差 1.9% / 1.2%、平移误差 **0.09 mm / 0.11 mm**、99.9% 收敛，与 OpenCV `solvePnP` 同级。

也就是说：**App 的测距/测姿误差主要不是检测器的问题，是解算器的问题。** 研究侧 B1（app 算法的 Python 移植）在诚实评测里 t_err 455 mm，换成修复解算器后同样检测结果是 4 mm——100× 的差距全部来自这一步。

## 1. 旧算法是什么

`calculate3PointPnP`（`PnPDistanceCalculator.kt:213`）把像素归一化为 `(x-cx)/fx` 后调 `solve3PointDepths`，未知数是三个点的深度 z_i，点模型 `P_i=(u_i z_i, v_i z_i, z_i)`，约束是三条边长度等于已知边长 100 mm。

旧更新规则（原 306–403 行）：

```
z_i *= 1 + (d_target / d_current - 1) * learning_rate     # lr = 0.5
```

每轮对三个点各做两次这样的乘法缩放，然后 `z_i = max(z_i, solverMinDepthMm=100)`。初值 `z_i = sqrt((d_edge / a_ij) * (d_edge / a_ik))`（`a` 为归一化图像边长）。

问题：这是一个**无阻尼的不动点迭代**，且只按每点边长的比值做乘法，没有任何耦合求解。当三条边约束互相矛盾时（热噪声、量化解、透视），迭代被放大而不是收敛，最终三个深度各自撞到 100 mm 的地板。

### 逐行复刻的证据

用 Python 按 Kotlin 规则逐行复刻（`app_solve_depths`），在 `syn_room_0200` 第 0 帧（GT 深度 575.2 / 546.4 / 539.2 mm）：

| 迭代 | max 边长误差 (mm) | depths (mm) |
|---|---|---|
| 初值 | 4.7 | 522.8 / 537.2 / 524.5 |
| 1 | 113.9 | 523.8 / 539.8 / 525.7 |
| 2 | 158.5 | 211.8 / 340.0 / 320.5 |
| 50（终） | 409.7 | **100.0 / 304.0 / 297.8** |

误差随迭代单调增大，不是"噪声抖"，是发散。端侧 B1 端口给出的结果与这份复刻逐位一致，所以不是移植 bug，是 App 本身。

## 2. 定量审计（GT 像素输入，排除检测因素）

脚本：`research/scripts/app_pnp_audit.py` → `research/results/app_pnp_audit/summary.json`
三条路径吃**同一份 GT 像素**：

- `app`：旧 Kotlin 规则复刻（FOV 60° 估焦，fx=1108.51，与 GT 一致）
- `fixed`：本次修复的解算器（同样 FOV 估焦）
- `b0`：OpenCV `solvePnP`（研究侧基线，GT 内参）

| 指标 | app | fixed | b0 (OpenCV) |
|---|---|---|---|
| **test 1244 帧** | | | |
| 收敛率 | 0.8 % | 99.92 % | — |
| 迭代中位数 | 49 | 4 | — |
| 距离相对误差 median | **0.539** | **0.019** | 0.0015 |
| 距离相对误差 p90 | 0.665 | 0.075 | 0.101 |
| 距离误差 ≤5% 帧占比 | 2.17 % | 77.49 % | 84.08 % |
| 平移误差 median (mm) | **456.7** | **0.09** | 1.54 |
| 平移误差 p90 (mm) | 882.7 | 96.5 | 81.1 |
| pose 误差 <50 mm 占比 | 2.41 % | 86.5 % | 86.74 % |
| **hard_test 2527 帧** | | | |
| 收敛率 | 4.27 % | 99.96 % | — |
| 迭代中位数 | 49 | 4 | — |
| 距离相对误差 median | **0.620** | **0.012** | 0.0002 |
| 距离相对误差 p90 | 0.684 | 0.035 | 0.043 |
| 距离误差 ≤5% 帧占比 | 4.35 % | 95.01 % | 92.56 % |
| 平移误差 median (mm) | **576.0** | **0.115** | 0.217 |
| 平移误差 p90 (mm) | 1078.2 | 45.5 | 44.8 |
| pose 误差 <50 mm 占比 | 4.39 % | 91.45 % | 91.45 % |

结论：**完美检测输入下，App 解算器仍然是全链路最大的误差源**。修好后与 OpenCV 同级，剩下的 ~9% 失败帧是三点 PnP 固有的多解/病态（见 §4）。

## 3. 修复

### 3.1 新算法

未知数仍是三个深度 `z = (z1, z2, z3)`，点模型 `P_i = (u_i z_i, v_i z_i, z_i)`（精确针孔模型，无小角近似）。残差取三条边与已知边长的差：

```
r_ij = ||P_i - P_j|| - edge          # (1,2), (2,3), (3,1)
```

解析 Jacobian：对每条边对每个深度求偏导

```
d r_ij / d z_i = ((P_i - P_j) · b_i) / ||P_i - P_j||,   b_i = (u_i, v_i, 1)
```

（`b_i·b_i` 展开即 `u_i²+v_i²+1`，`P_i·b_i = z_i (u_i²+v_i²+1)`，`P_j·b_i = z_j (u_j u_i + v_j v_i + 1)`）

用 LM：`(JᵀJ + λI) δ = -Jᵀr`，λ 初始 1e-6 每步减半；`z ← clamp(z + δ, z*0.5, z*1.5)`，且 `z ≥ solverMinDepthMm`；收敛判据 `max|r| < solverConvergenceErrorMm`（默认 0.1 mm），上限 50 次。GT 点上 2–4 次迭代收敛，边长误差 ~0.0 mm。

### 3.2 上帧深度播种（解 P3P 二义性）

`private var lastDepthsMm: FloatArray?`：用上一帧收敛的深度作为本帧初值。连续视频里同一几何解是连续的，这样解算器会稳定停在 GT 所在的根，而不会在两种解之间跳。收敛才更新，未收敛置空。

### 3.3 改动位置

| 位置 | 内容 |
|---|---|
| `PnPDistanceCalculator.kt:19` | `private var lastDepthsMm: FloatArray? = null` |
| `PnPDistanceCalculator.kt:322` | 新 `solve3PointDepths(...)`（Newton+LM，返回 Triple 深度，签名与调用点不变） |
| `PnPDistanceCalculator.kt:426` | 新 `solve3x3(...)`（Cramer 法则 3×3 线性解） |
| `PnPDistanceCalculator.kt:245` | 调用点不变 |

`calculate3PointPnP` 的距离口径（三点模长均值）、方位/仰角、`calculate6DOFPose` 均未改。

### 3.4 验证

- Python 复刻 `fixed_solve_depths`：GT 输入距离误差 1.9%/1.2%，平移 0.09/0.11 mm（§2 表）。
- 单元测试 `research/tests/test_app_pnp.py`（9 个，全过）：包含"旧解算器不收敛/距离偏小"的回归护栏、修复后收敛与 GT 恢复、退化输入、双解说明、时序种子。
- Kotlin 编译：`JAVA_HOME=<jdk17> ANDROID_HOME=~/Android/Sdk ~/.gradle/.../gradle-8.14.3/bin/gradle --offline --console=plain :composeApp:compileDebugKotlinAndroid` → BUILD SUCCESSFUL；`javap -p` 确认编译产物含 `lastDepthsMm`、`solve3x3`、`solve3PointDepths`。
  - 注：仓库里的 `./gradlew` 指向 gradle 9.3.1，本机只有未下完的 `.part`，离线会超时；用已缓存的 8.14.3 二进制即可。
- `assembleDebug`（离线）：BUILD SUCCESSFUL，`composeApp/build/outputs/apk/debug/composeApp-debug.apk` 于 12:24 重建（68 MB）；用 `zipfile` 检查 `classes*.dex` 含 `lastDepthsMm`、`solve3x3`，且已不含已废弃的 `solverLearningRate` 字段。
  - 顺带清理：`PnPDistanceCalculator` 的 `solverLearningRate` 字段、`AppTuningSettings.pnpSolverLearningRate`、设置面板的「PnP 学习率」滑块全部删除（新解算器不用这个参数；留着一个"改了没反应"的滑块比删掉更糟）。

## 4. 残余问题（不是解算器能解决的）

1. **三点 PnP 双解**：三条射线 + 三条已知边可能有两个深度解，二者重投影残差都为零。`syn_room_0200` 90 帧里有 9 帧失败，且**修复解算器与 OpenCV 的失败帧完全相同**（9/9），说明这是问题本身的性质。上帧深度播种能在连续跟踪时选对根；单帧、无先验时无法从几何上区分。若要彻底解决，需要 4 个以上点、或 IMU 先验、或时序滤波（多帧联合）。
2. **FOV 假设**：`estimatedFovDegrees` 默认 60°，真实设备若不标定，距离会成比例偏差（FOV 差 10%，距离差 ~10%）。研究侧数据是理想的 60°，所以这里看不出来。建议用 CameraX 的真实内参或做一次标定。
3. **2 点路径**（`calculate2PointPnP:57`）不受本次修复影响：它用 `realDistance / imageDistance` 的小基线近似，在点对同深度时误差 ~1%，可以接受；但 imageDistance≈0 时 `realDistance * fx` 这个 fallback 量纲是错的（会给出一个 10 万 mm 级的数），建议改成返回失败。
4. **端侧未实测**：本机没有 Android 设备，只能证明数学与编译正确；真机上的帧率、实际检测噪声、跟踪连续性需要用户装上 APK 验证。

## 5. App 级 A/B：同一个检测器，只换解算器

`research/results/honest/`（诚实评测：分母=全部帧、无在线门控、同 splits sha `bd40d8b6129e`、seed 0）：

| 方法 | 子集 | det@10(all) | det@10(legacy) | pose cov | t_err 中位 | t_err p90 | 抖动 mm/s | 延迟 ms |
|---|---|---|---|---|---|---|---|---|
| b1（app 现状：B1 检测 + 旧解算器） | test | 0.421 | 0.609 | 0.616 | **450.6 mm** | 873.0 | 443 | 83 |
| app_fixed（B1 检测 + 修复解算器） | test | 0.421 | 0.609 | 0.616 | **7.6 mm** | 726.0 | 858 | 79 |
| b0（OpenCV 基线） | test | 0.634 | 0.918 | 0.800 | 3.6 mm | 114.4 | 722 | 8 |
| b1 | hard_test | 0.149 | 0.212 | 0.714 | 513.3 mm | 1023.7 | 2416 | 77 |
| app_fixed | hard_test | 0.149 | 0.212 | 0.714 | **80.4 mm** | 1025.2 | 2465 | 79 |
| b0 | hard_test | 0.172 | 0.245 | 0.878 | 419.8 mm | 1089.8 | 3593 | 45 |

要点：**检测完全相同时，解算器把 t_err 从 450.6 mm 压到 7.6 mm（test，59×）**；hard_test 上从 513.3 mm 压到 80.4 mm（6.4×），甚至比 B0 的 419.8 mm 定位误差还小（B0 在这一集检测更稳但定位精度差）。所以：

1. App 现在"测距不准"的最大单一原因已定位并修复；
2. 修复后，App 的短板回到**检测**（det@10 legacy 0.61 → B0 0.92 / 学习版 0.88）与**抖动/p90 长尾**（三点 PnP 双解翻转）；
3. 真实瓶颈清单变成：检测召回/精度 → 时序滤波抑制翻转 → 相机内参标定（FOV）。

## 6. 建议的后续工作（按收益排序）

1. **相机内参与 FOV 标定**：现在默认 60°，真实设备不标定会有 ~10% 比例的系统距离偏差；用已知尺寸靶标做一次标定即可，也可用 CameraX 提供的内参。
2. **时序滤波**：对深度三元组（或平移）做轻量 Kalman/EMA + 双解消歧（比较与上帧预测的一致性），可消掉 p90 的长尾与 858 mm/s 抖动。这比换检测器便宜。
3. **检测器升级**：B1 在 hard 集 det@10 只有 0.21，学习侧（find_geom）有 0.84 但慢且重；可考虑把热图 + 几何解码蒸馏成小模型，或用颜色质心 snap 在端侧后处理（研究侧 snap r=14 把 26.7 mm 降到 9.1 mm）。
4. 训练数据：`hard_test` 与 `hard_train` 同分布、真实集为空，这些报告数字不能当作泛化证据；真实评测集到位后需重跑同一脚本。

## 7. 复现命令

```
# GT 输入审计（app / fixed / b0）
research/.venv/bin/python research/scripts/app_pnp_audit.py

# 单元测试
research/.venv/bin/python -m unittest research.tests.test_app_pnp -v

# 端到端诚实评测（B1 检测 + 修复解算器）
research/.venv/bin/python -m research.tridot_temporal.eval_v2 \
  --subset test --methods app_fixed --b1-grid 32 --out research/results/honest/app_fixed_test
```



## 8. 端侧时序滤波与双解消歧（2026-10-28 追加）

修好解算器之后还剩两个端侧观感问题：**三点 PnP 的镜像解会让 6DOF 姿态在帧间翻面**（距离几乎不变，但 roll/pitch/yaw 会瞬间错 40–150°），以及**检测噪声让 DISTANCE 数字抖动**。这两件事都不需要改检测器，用跨帧信息就能压住。

### 实现

- 端侧新文件 `composeApp/src/androidMain/kotlin/org/example/tridotpnp/PnPTemporalTracker.kt`：
  - `solveRoots(u,v,edgeMm)`：5 个初值多起点阻尼 Newton（复用修好的解算器内核），收敛根按 2% 相对差去重、按平均深度升序返回——平面靶标通常得到 1 或 2 个几何解；
  - `track(roots,u,v,edgeMm,timeSec)`：无历史时用「最接近单位旋转」先验选根（人手拿板大致正对相机）；有历史时按 `(旋转差, 平移差)` 字典序选根，并用旋转不一致门限拒绝镜像解跳变；平移超门限则滑行（constant-velocity 预测，最多 5 帧）后重新锁定；接受时做 alpha-beta 更新（默认 α=0.85, β=0.15, gate=60mm/8%）；
  - 上一帧深度作为下一次 `solveRoots` 的种子之一，连续运动时保持同一几何解。
- 接入点：`PnPDistanceCalculator.calculate3PointPnPTracked(...)`（`PnPDistanceCalculator.kt:319`）→ `DetectionCoordinator.calculateRgbPnpResult` 在 `tuning.temporalFilterEnabled` 时优先走它，失败回退到无状态 `calculate3PointPnP`；时间戳取 `System.nanoTime()`。
- App 设置「校准与 PnP」新增开关「时序滤波与双解消歧」（默认开）与「滤波平滑强度」滑块（0.30–0.95，默认 0.85；值越小越平滑）。

### 证据（B1 检测器固定，只换解算器；详见 `research/results/temporal_ab/REPORT.md`）

| 数据集 | 指标 | fixed（无时序） | tracked（默认配置） |
|---|---|---|---|
| test（15 序列/1800 帧） | 旋转跳变 >60° | 300 | **41** |
| | rot_err 中位 | 40.2° | **1.8°** |
| | t_err 中位 / p90 | 7.6 / 726 mm | **5.5** / 705 mm |
| | 帧间位移中位 | 858 mm/s | **540 mm/s** |
| hard_test（40 序列/3600 帧） | 旋转跳变 >60° | 1042 | **175** |
| | rot_err 中位 | 54.4° | **44.7°** |
| | t_err 中位 / p90 | 80.4 / 1025 mm | 84.2 / **926 mm** |
| | 帧间位移中位 | 2465 mm/s | **41 mm/s** |

纯解算耗时 0.9→4.9 ms/帧；检测侧 B1 在 grid 32 下约 188 ms/帧，因此整机帧率不变。

### 验证

- Kotlin 单测 `composeApp/src/androidUnitTest/kotlin/org/example/tridotpnp/PnPTemporalTrackerTest.kt`（JUnit4，5 个）：正视目标解算 <1 mm、枚举根含 GT 深度（<2%）、40 帧移动目标无 >30° 帧间跳变、+600 mm 单帧离群被拒并滑行、reset 后重新锁定。
- `research/tests/test_temporal.py`（11 个，Python 参考实现）：含选根、平滑、滑行、重锁、`smooth=False` 输出测量值等。
- `:composeApp:testDebugUnitTest` 通过；`:composeApp:assembleDebug` 出包并核对 dex 含 `PnPTemporalTracker`/`calculate3PointPnPTracked`。

### 局限

首帧无历史时用 identity 先验，hard_test 双根帧上与 GT 分支一致率 81.6%；一旦选错，旋转连续性会让它锁死在镜像解上（hard 的 far/hardmix 提升有限即此因）。彻底解决需要「双假设并行 + 残差竞争」或首次让用户点一下就绪；真实数据到位后可在评测集上量化。
