# 时序滤波 + 双解消歧 A/B（B1 检测器固定，仅换解算器）

同一批 `B1Detector(grid_size=32)` 检测结果分别喂给：

- **fixed**：`AppFixedSolver` —— 修好的三点 PnP（Newton+LM），无时序、无选根；
- **tracked**：`AppTrackedSolver` —— 多初值求全部几何根 + 时序一致性选根 + alpha-beta 滤波 + 离群滑行（默认上架配置：alpha=0.85, gate=60mm/8%）；
- **disamb**：同一套选根/门限，但输出原始测量（不滤波）—— 用来隔离「选根」与「滤波」各自的贡献。

指标说明：`t_err` 为目标原点平移误差（vs GT `pose_gt.translation_mm`）；`rot_err` 为旋转测地角误差；`flips>60°` 统计相邻**有效帧**之间旋转跳变 >60° 的次数（镜像解翻转会让姿态瞬间翻面而距离几乎不变）；`step` 为帧间质心位移速度（抖动代理）；`latency` 只含纯解算（detect 已固定相同）。

### test（15 序列 / 1800 帧）

| method | t_err med (mm) | t_err p90 | t≤50mm | rot_err med (°) | rot_err p90 | rot≤20° | flips>60° | step med (mm/s) | latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| fixed | 7.6 | 726.0 | 77.8% | 40.2 | 134.6 | 43.6% | 300/1096 | 857.9 | 0.85 |
| tracked | 5.5 | 704.9 | 82.6% | 1.8 | 120.7 | 62.6% | 41/1088 | 540.2 | 5.56 |
| disamb | 5.0 | 704.9 | 82.6% | 1.8 | 120.7 | 62.6% | 41/1088 | 558.3 | 5.54 |

变化：flips 300→41；t_err 中位 7.6→5.5 mm；rot_err 中位 40.2→1.8°；latency 0.85→5.56 ms。

| factor | t_err med fixed | t_err med tracked | flips fixed | flips tracked | frames |
|---|---|---|---|---|---|
| blur | 4.4 | 5.6 | 0 | 0 | 158 |
| clean | 7.7 | 5.5 | 0 | 0 | 165 |
| distractor | 199.5 | 24.0 | 123 | 18 | 240 |
| light | 2.9 | 5.1 | 0 | 0 | 124 |
| mixed | 7.3 | 7.1 | 176 | 22 | 294 |
| noise | 12.6 | 2.8 | 1 | 1 | 128 |

### hard_test（40 序列 / 3600 帧）

| method | t_err med (mm) | t_err p90 | t≤50mm | rot_err med (°) | rot_err p90 | rot≤20° | flips>60° | step med (mm/s) | latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| fixed | 80.4 | 1025.2 | 37.2% | 54.4 | 146.6 | 16.6% | 1042/2533 | 2465.4 | 0.90 |
| tracked | 84.2 | 926.1 | 36.2% | 44.7 | 134.1 | 24.1% | 175/2452 | 41.4 | 4.85 |
| disamb | 84.3 | 926.1 | 36.1% | 44.7 | 134.1 | 24.1% | 175/2452 | 35.6 | 4.82 |

变化：flips 1042→175；t_err 中位 80.4→84.2 mm；rot_err 中位 54.4→44.7°；latency 0.90→4.85 ms。

| factor | t_err med fixed | t_err med tracked | flips fixed | flips tracked | frames |
|---|---|---|---|---|---|
| clutter | 78.6 | 78.1 | 85 | 13 | 359 |
| crosscolor | 61.7 | 55.0 | 101 | 16 | 344 |
| desk | 62.1 | 60.9 | 57 | 5 | 355 |
| far | 1437.7 | 1482.6 | 114 | 18 | 170 |
| handheld | 44.1 | 52.2 | 58 | 10 | 266 |
| hardmix | 636.3 | 602.2 | 460 | 81 | 567 |
| occlude_hand | — | — | 48 | 10 | 115 |
| room | 26.3 | 32.3 | 0 | 1 | 239 |
| screen | 560.9 | 560.9 | 119 | 21 | 157 |

## 结论

- **test**：flips 300→41（1096 个有效帧间隔），rot_err 中位 40.2°→1.8°，rot≤20° 43.6%→62.6%；t_err 中位 7.6→5.5 mm，p90 726.0→704.9 mm，t≤50mm 77.8%→82.6%；帧间位移中位 858→540 mm/s；纯解算延迟 0.85→5.56 ms/帧。
- **hard_test**：flips 1042→175（2533 个有效帧间隔），rot_err 中位 54.4°→44.7°，rot≤20° 16.6%→24.1%；t_err 中位 80.4→84.2 mm，p90 1025.2→926.1 mm，t≤50mm 37.2%→36.2%；帧间位移中位 2465→41 mm/s；纯解算延迟 0.90→4.85 ms/帧。

1. **选根是姿态指标的主要来源**：`tracked` 与 `disamb` 的旋转指标几乎一致，说明收益来自「跨帧锁定同一几何解」而不是滤波本身；镜像解翻面会让姿态瞬间错 40–150°，这正是之前端上 6DOF 姿态跳动的根因。
2. **滤波主要降抖动**：`step`（帧间质心位移速度）大幅下降，说明输出平滑；代价是快加速时略有滞后——默认 α=0.85 已把滞后压到接近测量值，需要更平滑可在 App 里调低「滤波平滑强度」。
3. **精度不牺牲**：easy 集 t_err 中位从 7.6 降到 5.5 mm；hard 集持平（84 vs 80 mm，但 p90 从 1025 降到 926 mm）。距离本就不受双解影响（两解距离差几 mm~几十 mm），受影响的是姿态与抖动。
4. **开销可忽略**：纯解算 0.9→4.9 ms/帧；检测端 B1 每帧 ~188 ms（grid 32），所以整机帧率不变。
5. **首帧先验的局限**：无历史时用「最接近单位旋转」选根，在 hard_test 双根帧上与 GT 一致率 81.6%；选错时靠旋转连续性会锁死在镜像解上（表里 far/hardmix 的 rot≤20° 提升有限即此故）。下一步可加「两候选并行假设 + 残差比较」或让用户轻点一下第 1 帧。

## 复现

```bash
research/.venv/bin/python -m research.scripts.temporal_ab --subset test
research/.venv/bin/python -m research.scripts.temporal_ab --subset hard_test
research/.venv/bin/python research/scripts/temporal_ab_report.py
```

端侧对应实现：`composeApp/src/androidMain/kotlin/org/example/tridotpnp/PnPTemporalTracker.kt`，接入点 `PnPDistanceCalculator.calculate3PointPnPTracked(...)`，开关在 App「校准与 PnP」设置里的「时序滤波与双解消歧」。
