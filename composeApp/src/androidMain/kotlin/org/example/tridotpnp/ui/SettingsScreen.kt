package org.example.tridotpnp.ui


import org.example.tridotpnp.config.AppTuningSettings
import org.example.tridotpnp.config.updateCamera
import org.example.tridotpnp.config.updateDetect
import org.example.tridotpnp.config.updateTrackLock
import org.example.tridotpnp.config.updatePnp
import org.example.tridotpnp.config.updateFusion
import org.example.tridotpnp.config.updateBle
import org.example.tridotpnp.config.ProbabilityMatrix32x24
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DividerDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.unit.dp

@Composable
fun SettingsScreen(
    onDismiss: () -> Unit,
    themeMode: ThemeMode,
    onThemeModeChange: (ThemeMode) -> Unit,
    tuning: AppTuningSettings,
    onTuningChange: (AppTuningSettings) -> Unit,
    probabilityMatrix: ProbabilityMatrix32x24,
    onProbabilityMatrixEditRequest: () -> Unit,
    modifier: Modifier = Modifier
) {
    val haptic = LocalHapticFeedback.current
    Surface(
        modifier = modifier.fillMaxSize(),
        color = FlatUiColors.SidebarBackground
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "设置",
                    style = MaterialTheme.typography.headlineLarge,
                    color = FlatUiColors.TextPrimary
                )
                FlatIconActionButton(
                    onClick = {
                        haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                        onDismiss()
                    },
                    icon = {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "返回",
                            tint = FlatUiColors.TextPrimary
                        )
                    }
                )
            }

            SettingsOverviewContent(
                themeMode = themeMode,
                tuning = tuning,
                probabilityMatrix = probabilityMatrix,
                onThemeModeChange = onThemeModeChange,
                onTuningChange = onTuningChange,
                onOpenProbabilityEditor = {
                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                    onProbabilityMatrixEditRequest()
                }
            )
        }
    }
}

@Composable
internal fun SettingsOverviewContent(
    themeMode: ThemeMode,
    tuning: AppTuningSettings,
    probabilityMatrix: ProbabilityMatrix32x24,
    onThemeModeChange: (ThemeMode) -> Unit,
    onTuningChange: (AppTuningSettings) -> Unit,
    onOpenProbabilityEditor: () -> Unit
) {
    val haptic = LocalHapticFeedback.current
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        FlatPanel {
            Text(
                text = "设置选项",
                style = MaterialTheme.typography.titleLarge,
                color = FlatUiColors.TextPrimary,
                modifier = Modifier.padding(bottom = 8.dp)
            )

            Text(
                text = "界面已切换为扁平化风格。后续设置项建议继续沿用统一的浅底色板、低对比边框和无阴影按钮。",
                style = MaterialTheme.typography.bodyMedium,
                color = FlatUiColors.TextSecondary
            )

            HorizontalDivider(
                thickness = DividerDefaults.Thickness,
                color = FlatUiColors.Border
            )

            Text(
                text = "主题模式",
                style = MaterialTheme.typography.titleMedium,
                color = FlatUiColors.TextPrimary
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                ThemeModeButton(
                    label = "明亮",
                    selected = themeMode == ThemeMode.Light,
                    onClick = {
                        haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                        onThemeModeChange(ThemeMode.Light)
                    },
                    modifier = Modifier.weight(1f)
                )
                ThemeModeButton(
                    label = "暗黑",
                    selected = themeMode == ThemeMode.Dark,
                    onClick = {
                        haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                        onThemeModeChange(ThemeMode.Dark)
                    },
                    modifier = Modifier.weight(1f)
                )
            }
        }

        FlatPanel {
            Text(
                text = "概率矩阵",
                style = MaterialTheme.typography.titleMedium,
                color = FlatUiColors.TextPrimary
            )
            Text(
                text = "进入编辑模式后，会直接在摄像头画面上显示 64×48 概率矩阵覆盖层，并可直接触摸编辑。",
                style = MaterialTheme.typography.bodyMedium,
                color = FlatUiColors.TextSecondary
            )
            Text(
                text = "当前中心值 ${"%.2f".format(probabilityMatrix.weightAt(ProbabilityMatrix32x24.ROWS / 2, ProbabilityMatrix32x24.COLS / 2))}",
                style = MaterialTheme.typography.bodySmall,
                color = FlatUiColors.TextMuted
            )
            Button(
                onClick = onOpenProbabilityEditor,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = FlatUiColors.Accent,
                    contentColor = FlatUiColors.TextOnAccent
                ),
                shape = FlatUiShapes.Control,
                elevation = ButtonDefaults.buttonElevation(defaultElevation = 0.dp)
            ) {
                Text("进入概率矩阵编辑模式")
            }
        }

        SettingsSection(
            title = "识别流程",
            description = "控制采样密度、工作分辨率和基础阈值。值越高通常更稳，但也更耗时。",
            initiallyExpanded = true
        ) {
            BooleanSettingRow(
                title = "ROI 优化",
                description = "优先在上一帧目标附近搜索，可明显减少计算量；快速运动时可能需要关闭。",
                checked = tuning.enableRoiOptimization,
                onCheckedChange = { onTuningChange(tuning.updateCamera { copy(enableRoiOptimization = it) }) }
            )
            BooleanSettingRow(
                title = "GMM 软分割",
                description = "在 HSV 相似度上叠加双高斯颜色后验，提升光照变化下的稳定性。",
                checked = tuning.enableGmmSoftSegment,
                onCheckedChange = { onTuningChange(tuning.updateDetect { copy(enableGmmSoftSegment = it) }) }
            )
            FloatSliderSetting(
                title = "GMM 融合权重",
                description = "越大越依赖 GMM 后验，越小越接近原 HSV 相似度。",
                value = tuning.gmmBlend,
                valueText = "%.2f".format(tuning.gmmBlend),
                range = 0f..1f,
                steps = 19,
                onValueChange = { onTuningChange(tuning.updateDetect { copy(gmmBlend = it) }) }
            )
            FloatSliderSetting(
                title = "检测阈值",
                description = "亮点候选最低得分。升高可减少误检，降低可提升弱光目标通过率。",
                value = tuning.detectionThreshold,
                valueText = "%.0f".format(tuning.detectionThreshold),
                range = 10f..200f,
                steps = 18,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(detectionThreshold = it) }) }
            )
            IntSliderSetting(
                title = "检测网格",
                description = "每帧采样网格边长。越大越精细，但计算量近似按平方增加。",
                value = tuning.gridSize,
                valueText = "${tuning.gridSize} x ${tuning.gridSize}",
                range = 32..512,
                step = 8,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(gridSize = it) }) }
            )
            IntSliderSetting(
                title = "工作分辨率上限",
                description = "图像会先缩放到此最大边长再做检测。更高更精细，更低更省算力。",
                value = tuning.workingMaxDimension,
                valueText = "${tuning.workingMaxDimension}px",
                range = 320..1920,
                step = 32,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(workingMaxDimension = it) }) }
            )
            IntSliderSetting(
                title = "曝光补偿",
                description = "直接影响画面亮度和高光饱和，间接影响识别稳定性。",
                value = tuning.exposureCompensation,
                valueText = tuning.exposureCompensation.toString(),
                range = -6..6,
                step = 1,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(exposureCompensation = it) }) }
            )
        }

        SettingsSection(
            title = "视觉融合与多节点",
            description = "置信度门控、灯语身份解码、多节点跟踪、IMU 稀疏融合与姿态软约束。"
        ) {
            BooleanSettingRow(
                title = "视觉置信度门控",
                description = "综合几何/重投影/纯度评分，低质量观测不进入位姿输出与融合。",
                checked = tuning.enableVisionConfidenceGate,
                onCheckedChange = { onTuningChange(tuning.updateFusion { copy(enableVisionConfidenceGate = it) }) }
            )
            FloatSliderSetting(
                title = "最低置信度",
                description = "低于该阈值的视觉观测将被拒绝。",
                value = tuning.visionMinConfidence,
                valueText = "%.2f".format(tuning.visionMinConfidence),
                range = 0.1f..0.95f,
                steps = 16,
                onValueChange = { onTuningChange(tuning.updateFusion { copy(visionMinConfidence = it) }) }
            )
            FloatSliderSetting(
                title = "边长残差上限",
                description = "PnP 恢复的三边与已知边长平均偏差超过该值则拒收（毫米）。",
                value = tuning.visionMaxReprojectionErrorMm,
                valueText = "%.0f mm".format(tuning.visionMaxReprojectionErrorMm),
                range = 5f..60f,
                steps = 10,
                onValueChange = { onTuningChange(tuning.updateFusion { copy(visionMaxReprojectionErrorMm = it) }) }
            )
            BooleanSettingRow(
                title = "灯语身份解码",
                description = "对编码灯亮度序列做同步头相关 + CRC，确认节点 ID。",
                checked = tuning.enableIdentityDecode,
                onCheckedChange = { onTuningChange(tuning.updateFusion { copy(enableIdentityDecode = it) }) }
            )
            IntSliderSetting(
                title = "身份确认次数",
                description = "连续成功解码同一 ID 达到该次数后才锁定身份。",
                value = tuning.identityConfirmNeeded,
                valueText = tuning.identityConfirmNeeded.toString(),
                range = 1..8,
                step = 1,
                onValueChange = { onTuningChange(tuning.updateFusion { copy(identityConfirmNeeded = it) }) }
            )
            BooleanSettingRow(
                title = "多节点跟踪",
                description = "从候选亮点中抽取多个三角形，并为每个节点维护独立跟踪器。",
                checked = tuning.enableMultiNodeTracking,
                onCheckedChange = { onTuningChange(tuning.updateFusion { copy(enableMultiNodeTracking = it) }) }
            )
            IntSliderSetting(
                title = "最大节点数",
                description = "同时跟踪的节点上限（建议先 3～6）。",
                value = tuning.maxTrackedNodes,
                valueText = tuning.maxTrackedNodes.toString(),
                range = 1..12,
                step = 1,
                onValueChange = { onTuningChange(tuning.updateFusion { copy(maxTrackedNodes = it) }) }
            )
            BooleanSettingRow(
                title = "人体拓扑先验",
                description = "解码 ID 与空间角色冲突时暂缓切换，抑制 ID 抖动。",
                checked = tuning.enableTopologyPrior,
                onCheckedChange = { onTuningChange(tuning.updateFusion { copy(enableTopologyPrior = it) }) }
            )
            BooleanSettingRow(
                title = "ESKF 惯性融合",
                description = "用 IMU 做短时预测，高置信视觉再校正；不确定度过大时提示需要视觉校准。",
                checked = tuning.enableEskfFusion,
                onCheckedChange = { onTuningChange(tuning.updateFusion { copy(enableEskfFusion = it) }) }
            )
            BooleanSettingRow(
                title = "IK 姿态软约束",
                description = "多节点位姿后再按骨骼长度做软约束，压掉明显不合理的姿态。",
                checked = tuning.enableIkPostProcess,
                onCheckedChange = { onTuningChange(tuning.updateFusion { copy(enableIkPostProcess = it) }) }
            )
            BooleanSettingRow(
                title = "BLE IMU 接入",
                description = "扫描名为 TriDot* 的节点并接收 IMU 数据包（需固件配合）。",
                checked = tuning.enableBleImu,
                onCheckedChange = { onTuningChange(tuning.updateBle { copy(enableBleImu = it) }) }
            )
        }

        SettingsSection(
            title = "ROI 搜索",
            description = "控制跟踪目标后的局部搜索范围、扩张速度和失败回退策略。"
        ) {
            BooleanSettingRow(
                title = "ROI 预算网格",
                description = "ROI 内按目标步长加密采样；圈变大时用格子预算自动变稀，且不会粗过全图网格。",
                checked = tuning.enableRoiBudgetGrid,
                onCheckedChange = { onTuningChange(tuning.updateCamera { copy(enableRoiBudgetGrid = it) }) }
            )
            IntSliderSetting(
                title = "ROI 目标步长",
                description = "工作图像素间距。越小越密。仅在 ROI 预算网格开启时生效。",
                value = tuning.roiIdealStepPx,
                valueText = "${tuning.roiIdealStepPx}px",
                range = 1..16,
                step = 1,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(roiIdealStepPx = it) }) }
            )
            IntSliderSetting(
                title = "ROI 格子预算",
                description = "单次 ROI 扫描最多约多少格。大圈时靠它自动放宽步长，避免算力爆炸。",
                value = tuning.roiMaxCells,
                valueText = tuning.roiMaxCells.toString(),
                range = 512..16384,
                step = 512,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(roiMaxCells = it) }) }
            )
            FloatSliderSetting(
                title = "初始半径倍率",
                description = "以上一帧三点外接圆半径为基础扩大多少倍作为首个 ROI。",
                value = tuning.roiInitialRadiusMultiplier,
                valueText = "%.2f".format(tuning.roiInitialRadiusMultiplier),
                range = 0.5f..4f,
                steps = 13,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(roiInitialRadiusMultiplier = it) }) }
            )
            FloatSliderSetting(
                title = "扩张倍率",
                description = "ROI 每次未命中后的放大比例。更大能更快重新找回目标，但更耗时。",
                value = tuning.roiExpandFactor,
                valueText = "%.2f".format(tuning.roiExpandFactor),
                range = 1.05f..2f,
                steps = 18,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(roiExpandFactor = it) }) }
            )
            FloatSliderSetting(
                title = "最大搜索半径比例",
                description = "ROI 最大可扩到画面长边的比例，过小可能丢目标，过大则接近全图搜索。",
                value = tuning.roiMaxRadiusRatio,
                valueText = "%.2f".format(tuning.roiMaxRadiusRatio),
                range = 0.1f..1f,
                steps = 17,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(roiMaxRadiusRatio = it) }) }
            )
            IntSliderSetting(
                title = "最大扩张次数",
                description = "单帧局部搜索最多扩几轮，限制最坏情况下的计算时间。",
                value = tuning.roiMaxExpansions,
                valueText = tuning.roiMaxExpansions.toString(),
                range = 1..20,
                step = 1,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(roiMaxExpansions = it) }) }
            )
            FloatSliderSetting(
                title = "稳定性容差",
                description = "与上一轮三点半径变化的允许比例。越低越严格，越高越容易接受突变。",
                value = tuning.roiStabilityTolerance,
                valueText = "%.2f".format(tuning.roiStabilityTolerance),
                range = 0f..1f,
                steps = 19,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(roiStabilityTolerance = it) }) }
            )
            IntSliderSetting(
                title = "最近尝试窗口",
                description = "ROI 连续失败统计窗口长度，用来判断何时退回全图搜索。",
                value = tuning.roiRecentAttempts,
                valueText = tuning.roiRecentAttempts.toString(),
                range = 1..15,
                step = 1,
                onValueChange = {
                    val attempts = it
                    onTuningChange(
                        tuning.updateCamera {
                            copy(
                                roiRecentAttempts = attempts,
                                roiFailureThreshold = tuning.roiFailureThreshold.coerceAtMost(attempts)
                            )
                        }
                    )
                }
            )
            IntSliderSetting(
                title = "失败回退阈值",
                description = "最近窗口中失败达到多少次时直接切回全图搜索。",
                value = tuning.roiFailureThreshold,
                valueText = tuning.roiFailureThreshold.toString(),
                range = 1..tuning.roiRecentAttempts,
                step = 1,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(roiFailureThreshold = it) }) }
            )
            FloatSliderSetting(
                title = "跟踪粘滞裕度",
                description = "基础粘滞：新三角形得分需高出该比例才允许切换。实际裕度还会随锁定置信度放大。0 关闭。",
                value = tuning.trackSwitchMargin,
                valueText = "%.0f%%".format(tuning.trackSwitchMargin * 100f),
                range = 0f..1f,
                steps = 19,
                onValueChange = { onTuningChange(tuning.updateTrackLock { copy(trackSwitchMargin = it) }) }
            )
            FloatSliderSetting(
                title = "锁定匹配距离",
                description = "判定仍是同一组锁定点时，每个颜色允许偏离上一帧位置的最大像素距离（工作图）。",
                value = tuning.trackLockMaxDistPx,
                valueText = "%.0f px".format(tuning.trackLockMaxDistPx),
                range = 16f..200f,
                steps = 22,
                onValueChange = { onTuningChange(tuning.updateTrackLock { copy(trackLockMaxDistPx = it) }) }
            )
            FloatSliderSetting(
                title = "锁定置信上升",
                description = "每连续稳定跟踪一帧，锁定置信度增加多少。跟踪越久，越难被旁路亮点抢走。",
                value = tuning.trackLockConfidenceRise,
                valueText = "%.2f".format(tuning.trackLockConfidenceRise),
                range = 0.01f..0.3f,
                steps = 28,
                onValueChange = { onTuningChange(tuning.updateTrackLock { copy(trackLockConfidenceRise = it) }) }
            )
            FloatSliderSetting(
                title = "锁定置信衰减",
                description = "丢检时每帧锁定置信度下降多少。衰减太快会重新变“飘”。",
                value = tuning.trackLockConfidenceDecay,
                valueText = "%.2f".format(tuning.trackLockConfidenceDecay),
                range = 0.01f..0.3f,
                steps = 28,
                onValueChange = { onTuningChange(tuning.updateTrackLock { copy(trackLockConfidenceDecay = it) }) }
            )
            FloatSliderSetting(
                title = "置信放大粘滞",
                description = "有效粘滞 = 基础 × (1 + 本值 × 置信度)。例如 2.0 且置信度满时，粘滞为基础的 3 倍。",
                value = tuning.trackLockConfidenceBoost,
                valueText = "%.1f×".format(tuning.trackLockConfidenceBoost),
                range = 0f..4f,
                steps = 19,
                onValueChange = { onTuningChange(tuning.updateTrackLock { copy(trackLockConfidenceBoost = it) }) }
            )
        }

        SettingsSection(
            title = "中心细化",
            description = "在原图上对三点中心做二次定位，主要影响位置精度和耗时。"
        ) {
            IntSliderSetting(
                title = "细化迭代次数",
                description = "重复质心细化的次数。更多迭代可能更准，但会增加单帧计算。",
                value = tuning.refineIterations,
                valueText = tuning.refineIterations.toString(),
                range = 1..5,
                step = 1,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(refineIterations = it) }) }
            )
            FloatSliderSetting(
                title = "细化半径倍率",
                description = "原图细化窗口大小，越大越不容易偏移丢点，但更易受背景干扰。",
                value = tuning.refineRadiusMultiplier,
                valueText = "%.2f".format(tuning.refineRadiusMultiplier),
                range = 0.5f..4f,
                steps = 13,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(refineRadiusMultiplier = it) }) }
            )
            IntSliderSetting(
                title = "最小细化半径",
                description = "细化窗口的像素下限，防止小网格时窗口过小。",
                value = tuning.refineMinRadiusPx,
                valueText = "${tuning.refineMinRadiusPx}px",
                range = 1..20,
                step = 1,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(refineMinRadiusPx = it) }) }
            )
            FloatSliderSetting(
                title = "背景抑制权重",
                description = "越偏向黑背景的像素会额外加分，适合亮点在黑底上的场景。",
                value = tuning.refineBackgroundDarkness,
                valueText = "%.2f".format(tuning.refineBackgroundDarkness),
                range = 0f..2f,
                steps = 19,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(refineBackgroundDarkness = it) }) }
            )
            FloatSliderSetting(
                title = "高斯聚焦系数",
                description = "越小越强调中心区域，越大越均匀地参考整个细化窗口。",
                value = tuning.refineGaussianSigmaFactor,
                valueText = "%.2f".format(tuning.refineGaussianSigmaFactor),
                range = 0.1f..1.5f,
                steps = 13,
                onValueChange = { onTuningChange(tuning.updateCamera { copy(refineGaussianSigmaFactor = it) }) }
            )
        }

        SettingsSection(
            title = "颜色筛选",
            description = "控制候选区域亮度、面积和 HSV 相似度的门槛。"
        ) {
            FloatSliderSetting("像素最小亮度", "校准取色时忽略过暗像素。", tuning.minPixelBrightness, "%.0f".format(tuning.minPixelBrightness), 0f..100f, 19) {
                onTuningChange(tuning.updateDetect { copy(minPixelBrightness = it) })
            }
            FloatSliderSetting("区域最小亮度", "候选区域平均亮度下限，过低会被直接丢弃。", tuning.minTotalBrightness, "%.0f".format(tuning.minTotalBrightness), 0f..255f, 24) {
                onTuningChange(tuning.updateDetect { copy(minTotalBrightness = it) })
            }
            FloatSliderSetting("动态阈值下限", "局部最亮像素不足时的保底阈值。", tuning.dynamicThresholdMin, "%.0f".format(tuning.dynamicThresholdMin), 0f..255f, 24) {
                onTuningChange(tuning.updateDetect { copy(dynamicThresholdMin = it) })
            }
            FloatSliderSetting("动态阈值比例", "取区域峰值亮度的多少作为亮像素阈值。", tuning.dynamicThresholdRatio, "%.2f".format(tuning.dynamicThresholdRatio), 0.05f..1f, 18) {
                onTuningChange(tuning.updateDetect { copy(dynamicThresholdRatio = it) })
            }
            IntSliderSetting("最少亮像素数", "候选区域至少包含多少个亮像素。", tuning.minPixelCount, tuning.minPixelCount.toString(), 0..20, 1) {
                onTuningChange(tuning.updateDetect { copy(minPixelCount = it) })
            }
            IntSliderSetting("检测器细化半径下限", "候选点在检测器内部二次细化时使用的最小半径。", tuning.detectorMinRefineRadius, "${tuning.detectorMinRefineRadius}px", 1..20, 1) {
                onTuningChange(tuning.updateDetect { copy(detectorMinRefineRadius = it) })
            }
            FloatSliderSetting("亮像素占比上限", "候选区域里亮像素占比过高时视为大块光斑并过滤。", tuning.maxBrightPixelRatio, "%.2f".format(tuning.maxBrightPixelRatio), 0.01f..0.5f, 24) {
                onTuningChange(tuning.updateDetect { copy(maxBrightPixelRatio = it) })
            }
            IntSliderSetting("亮像素数量兜底上限", "小窗口场景下的亮像素绝对数量上限。", tuning.maxBrightPixelCountMin, tuning.maxBrightPixelCountMin.toString(), 1..50, 1) {
                onTuningChange(tuning.updateDetect { copy(maxBrightPixelCountMin = it) })
            }
            FloatSliderSetting("最小饱和度", "过滤偏灰偏白高光，降低镜面反射误检。", tuning.hsvMinSaturation, "%.2f".format(tuning.hsvMinSaturation), 0f..1f, 19) {
                onTuningChange(tuning.updateDetect { copy(hsvMinSaturation = it) })
            }
            FloatSliderSetting("最小明度", "过滤过暗噪声。", tuning.hsvMinValue, "%.2f".format(tuning.hsvMinValue), 0f..1f, 19) {
                onTuningChange(tuning.updateDetect { copy(hsvMinValue = it) })
            }
            FloatSliderSetting("色相权重", "HSV 相似度中色相差异的影响强度。", tuning.hsvHueWeight, "%.2f".format(tuning.hsvHueWeight), 0f..3f, 29) {
                onTuningChange(tuning.updateDetect { copy(hsvHueWeight = it) })
            }
            FloatSliderSetting("饱和度权重", "HSV 相似度中饱和度差异的影响强度。", tuning.hsvSatWeight, "%.2f".format(tuning.hsvSatWeight), 0f..3f, 29) {
                onTuningChange(tuning.updateDetect { copy(hsvSatWeight = it) })
            }
            FloatSliderSetting("明度权重", "HSV 相似度中明度差异的影响强度。", tuning.hsvValWeight, "%.2f".format(tuning.hsvValWeight), 0f..3f, 29) {
                onTuningChange(tuning.updateDetect { copy(hsvValWeight = it) })
            }
            FloatSliderSetting("HSV 锐度", "越大越挑剔，相似度会衰减得更快。", tuning.hsvSharpness, "%.2f".format(tuning.hsvSharpness), 1f..20f, 18) {
                onTuningChange(tuning.updateDetect { copy(hsvSharpness = it) })
            }
            FloatSliderSetting("暗背景增益", "候选周围越暗，整体亮度加权越高。", tuning.darkBackgroundBoost, "%.2f".format(tuning.darkBackgroundBoost), 0f..2f, 19) {
                onTuningChange(tuning.updateDetect { copy(darkBackgroundBoost = it) })
            }
        }

        SettingsSection(
            title = "三点组合评分",
            description = "控制 RGB 三点组合时的几何约束、颜色纯度和概率矩阵影响。"
        ) {
            IntSliderSetting("每色候选数", "每种颜色保留多少个候选点参与三点组合。更高更稳，但组合数会暴涨。", tuning.topNCandidatesPerColor, tuning.topNCandidatesPerColor.toString(), 1..64, 1) {
                onTuningChange(tuning.updateDetect { copy(topNCandidatesPerColor = it) })
            }
            FloatSliderSetting("最小点距倍率", "约束三点之间最短允许距离，避免把噪点拼成一组。", tuning.minPairDistanceMultiplier, "%.2f".format(tuning.minPairDistanceMultiplier), 0.5f..10f, 18) {
                onTuningChange(tuning.updateDetect { copy(minPairDistanceMultiplier = it) })
            }
            FloatSliderSetting("最大点距倍率", "约束三点之间最大允许距离，避免把分散目标拼成一组。", tuning.maxPairDistanceMultiplier, "%.2f".format(tuning.maxPairDistanceMultiplier), 2f..40f, 18) {
                onTuningChange(tuning.updateDetect { copy(maxPairDistanceMultiplier = it) })
            }
            FloatSliderSetting("形状惩罚", "越高越偏向接近等边三角形的三点组合。", tuning.triShapePenaltyWeight, "%.2f".format(tuning.triShapePenaltyWeight), 0f..2f, 19) {
                onTuningChange(tuning.updateDetect { copy(triShapePenaltyWeight = it) })
            }
            FloatSliderSetting("距离惩罚", "越高越严格限制边长是否落在期望范围内。", tuning.triGeometryPenaltyWeight, "%.2f".format(tuning.triGeometryPenaltyWeight), 0f..3f, 29) {
                onTuningChange(tuning.updateDetect { copy(triGeometryPenaltyWeight = it) })
            }
            FloatSliderSetting("纯度加分", "越高越强调单色明显强于另外两色。", tuning.triPurityBoostWeight, "%.2f".format(tuning.triPurityBoostWeight), 0f..2f, 19) {
                onTuningChange(tuning.updateDetect { copy(triPurityBoostWeight = it) })
            }
            FloatSliderSetting("中心偏黑加分", "越高越偏向三点中间为暗区的结构。", tuning.triCenterBlackWeight, "%.2f".format(tuning.triCenterBlackWeight), 0f..3f, 29) {
                onTuningChange(tuning.updateDetect { copy(triCenterBlackWeight = it) })
            }
            FloatSliderSetting("颜色总分权重", "整体放大或缩小颜色分数在最终组合中的影响。", tuning.triColorScoreWeight, "%.2f".format(tuning.triColorScoreWeight), 0f..3f, 29) {
                onTuningChange(tuning.updateDetect { copy(triColorScoreWeight = it) })
            }
            FloatSliderSetting("候选概率指数", "候选阶段概率矩阵影响强度，越高越压低低概率区域。", tuning.probabilityCandidateExponent, "%.2f".format(tuning.probabilityCandidateExponent), 0f..3f, 29) {
                onTuningChange(tuning.updateDetect { copy(probabilityCandidateExponent = it) })
            }
            FloatSliderSetting("组合概率指数", "三点组合阶段概率矩阵影响强度。", tuning.probabilityGroupExponent, "%.2f".format(tuning.probabilityGroupExponent), 0f..3f, 29) {
                onTuningChange(tuning.updateDetect { copy(probabilityGroupExponent = it) })
            }
            FloatSliderSetting("概率权重下限", "避免某些区域权重过低而被完全排除。", tuning.probabilityWeightFloor, "%.3f".format(tuning.probabilityWeightFloor), 0.001f..0.2f, 19) {
                onTuningChange(tuning.updateDetect { copy(probabilityWeightFloor = it) })
            }
        }

        SettingsSection(
            title = "校准与 PnP",
            description = "控制颜色校准取样方式，以及姿态计算里的焦距与迭代求解参数。"
        ) {
            IntSliderSetting("校准搜索步长", "校准时在圆区域内隔多少像素取一个样点。步长越小越准，也更耗时。", tuning.calibrationSearchStep, tuning.calibrationSearchStep.toString(), 1..30, 1) {
                onTuningChange(tuning.updateDetect { copy(calibrationSearchStep = it) })
            }
            IntSliderSetting("校准搜索半径", "每个校准样点周围取色窗口的半径。", tuning.calibrationSearchRadius, "${tuning.calibrationSearchRadius}px", 1..40, 1) {
                onTuningChange(tuning.updateDetect { copy(calibrationSearchRadius = it) })
            }
            FloatSliderSetting("校准高亮像素比例", "取色时只平均最亮部分像素，避免背景污染。", tuning.sampleTopPercent, "%.2f".format(tuning.sampleTopPercent), 0.01f..0.8f, 19) {
                onTuningChange(tuning.updateDetect { copy(sampleTopPercent = it) })
            }
            FloatSliderSetting("三点边长", "已知 RGB 三点构成的实际边长（节点设计为 18mm），直接影响距离解算结果。", tuning.knownTriangleEdgeLengthMm, "%.0f mm".format(tuning.knownTriangleEdgeLengthMm), 5f..100f, 94) {
                onTuningChange(tuning.updatePnp { copy(knownTriangleEdgeLengthMm = it) })
            }
            FloatSliderSetting("估算视场角", "用于在无真实焦距时估算相机焦距，直接影响 PnP 距离尺度。", tuning.estimatedFovDegrees, "%.1f°".format(tuning.estimatedFovDegrees), 20f..120f, 19) {
                onTuningChange(tuning.updatePnp { copy(estimatedFovDegrees = it) })
            }
            FloatSliderSetting("深度初值", "PnP 求解初始化深度，主要影响收敛速度与稳定性。", tuning.pnpDepthInitFallbackMm, "%.0f mm".format(tuning.pnpDepthInitFallbackMm), 100f..5000f, 24) {
                onTuningChange(tuning.updatePnp { copy(pnpDepthInitFallbackMm = it) })
            }
            IntSliderSetting("PnP 最大迭代数", "求解器最多迭代次数。更高更容易收敛，但更耗时。", tuning.pnpSolverMaxIterations, tuning.pnpSolverMaxIterations.toString(), 1..200, 1) {
                onTuningChange(tuning.updatePnp { copy(pnpSolverMaxIterations = it) })
            }
            FloatSliderSetting("PnP 收敛误差", "误差低于该值就提前停止。越低越精确，但可能迭代更久。", tuning.pnpSolverConvergenceErrorMm, "%.3f".format(tuning.pnpSolverConvergenceErrorMm), 0.001f..5f, 19) {
                onTuningChange(tuning.updatePnp { copy(pnpSolverConvergenceErrorMm = it) })
            }
            FloatSliderSetting("PnP 学习率", "每轮调整深度的步长。过低收敛慢，过高可能震荡。", tuning.pnpSolverLearningRate, "%.2f".format(tuning.pnpSolverLearningRate), 0.01f..1f, 19) {
                onTuningChange(tuning.updatePnp { copy(pnpSolverLearningRate = it) })
            }
            FloatSliderSetting("PnP 最小深度", "限制求解结果深度不能低于该值，避免不合理解。", tuning.pnpSolverMinDepthMm, "%.0f mm".format(tuning.pnpSolverMinDepthMm), 1f..1000f, 19) {
                onTuningChange(tuning.updatePnp { copy(pnpSolverMinDepthMm = it) })
            }
        }
    }
}

@Composable
internal fun SettingsSection(
    title: String,
    description: String,
    initiallyExpanded: Boolean = false,
    content: @Composable () -> Unit
) {
    var expanded by rememberSaveable(title) { mutableStateOf(initiallyExpanded) }
    val haptic = LocalHapticFeedback.current

    FlatPanel {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        )
        {
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium,
                    color = FlatUiColors.TextPrimary
                )
                Text(
                    text = description,
                    style = MaterialTheme.typography.bodyMedium,
                    color = FlatUiColors.TextSecondary
                )
            }
            FlatIconActionButton(
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                    expanded = !expanded
                },
                icon = {
                    Icon(
                        imageVector = if (expanded) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore,
                        contentDescription = if (expanded) "收起" else "展开",
                        tint = FlatUiColors.TextPrimary
                    )
                }
            )
        }

        AnimatedVisibility(
            visible = expanded,
            enter = expandVertically(
                animationSpec = tween(durationMillis = 220),
                expandFrom = Alignment.Top
            ) + fadeIn(animationSpec = tween(durationMillis = 180)),
            exit = shrinkVertically(
                animationSpec = tween(durationMillis = 180),
                shrinkTowards = Alignment.Top
            ) + fadeOut(animationSpec = tween(durationMillis = 140))
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                HorizontalDivider(
                    thickness = DividerDefaults.Thickness,
                    color = FlatUiColors.Border
                )
                content()
            }
        }
    }
}

@Composable
internal fun FloatSliderSetting(
    title: String,
    description: String,
    value: Float,
    valueText: String,
    range: ClosedFloatingPointRange<Float>,
    steps: Int,
    onValueChange: (Float) -> Unit
) {
    val haptic = LocalHapticFeedback.current
    val bucketCount = (steps + 1).coerceAtLeast(1)
    fun bucketIndexOf(current: Float): Int {
        val start = range.start
        val end = range.endInclusive
        val normalized = if (end > start) {
            ((current - start) / (end - start)).coerceIn(0f, 1f)
        } else {
            0f
        }
        return kotlin.math.round(normalized * bucketCount).toInt().coerceIn(0, bucketCount)
    }
    var lastBucketIndex by remember(title, range.start, range.endInclusive, steps) {
        mutableIntStateOf(bucketIndexOf(value))
    }

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(title, color = FlatUiColors.TextPrimary, style = MaterialTheme.typography.bodyLarge)
            Text(valueText, color = FlatUiColors.Accent, style = MaterialTheme.typography.labelLarge)
        }
        Text(description, color = FlatUiColors.TextMuted, style = MaterialTheme.typography.bodySmall)
        Slider(
            value = value.coerceIn(range.start, range.endInclusive),
            onValueChange = { nextValue ->
                val nextBucketIndex = bucketIndexOf(nextValue)
                if (nextBucketIndex != lastBucketIndex) {
                    haptic.performHapticFeedback(HapticFeedbackType.SegmentTick)
                    lastBucketIndex = nextBucketIndex
                }
                onValueChange(nextValue)
            },
            onValueChangeFinished = {
                lastBucketIndex = bucketIndexOf(value)
                haptic.performHapticFeedback(HapticFeedbackType.GestureEnd)
            },
            valueRange = range,
            steps = steps
        )
    }
}

@Composable
internal fun IntSliderSetting(
    title: String,
    description: String,
    value: Int,
    valueText: String,
    range: IntRange,
    step: Int,
    onValueChange: (Int) -> Unit
) {
    val safeStep = step.coerceAtLeast(1)
    val start = range.first.toFloat()
    val end = range.last.toFloat()
    val sliderValue = value.coerceIn(range.first, range.last).toFloat()
    FloatSliderSetting(
        title = title,
        description = description,
        value = sliderValue,
        valueText = valueText,
        range = start..end,
        steps = ((range.last - range.first) / safeStep - 1).coerceAtLeast(0),
        onValueChange = { raw ->
            val snapped = (kotlin.math.round((raw - range.first) / safeStep) * safeStep + range.first)
                .toInt()
                .coerceIn(range.first, range.last)
            onValueChange(snapped)
        }
    )
}

@Composable
internal fun BooleanSettingRow(
    title: String,
    description: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    val haptic = LocalHapticFeedback.current
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(title, color = FlatUiColors.TextPrimary, style = MaterialTheme.typography.bodyLarge)
                Text(description, color = FlatUiColors.TextMuted, style = MaterialTheme.typography.bodySmall)
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ThemeModeButton(
                    label = "关",
                    selected = !checked,
                    onClick = {
                        haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                        onCheckedChange(false)
                    }
                )
                ThemeModeButton(
                    label = "开",
                    selected = checked,
                    onClick = {
                        haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                        onCheckedChange(true)
                    }
                )
            }
        }
    }
}
