package org.example.tridotpnp

import android.Manifest
import android.util.Log
import android.widget.Toast
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.compose.animation.core.tween
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsDraggedAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DividerDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.lightColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.accompanist.permissions.shouldShowRationale

@Composable
@Preview
fun App() {
    var themeMode by remember { mutableStateOf(ThemeMode.Light) }
    TriDotFlatTheme(themeMode = themeMode) {
        BrightSpotDetectionApp(
            themeMode = themeMode,
            onThemeModeChange = { themeMode = it }
        )
    }
}

@androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun BrightSpotDetectionApp(
    themeMode: ThemeMode,
    onThemeModeChange: (ThemeMode) -> Unit
) {
    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)
    var detectedSpotsCount by remember { mutableIntStateOf(0) }
    var detectedSpots by remember { mutableStateOf<List<BrightSpot>>(emptyList()) }
    val targetSpotCount = 3
    var tuning by remember { mutableStateOf(AppTuningSettings()) }
    var probabilityMatrix by remember { mutableStateOf(buildManualProbabilityMatrix32x24()) }
    var pnpResult by remember { mutableStateOf<PnPDistanceCalculator.PnPResult?>(null) }
    var imageSize by remember { mutableStateOf<Pair<Int, Int>?>(null) }
    var fps by remember { mutableFloatStateOf(0f) } // 识别帧率
    var showSettings by remember { mutableStateOf(false) } // 设置页面显示状态
    var isProbabilityEditMode by remember { mutableStateOf(false) }
    var probabilityEditMode by remember { mutableStateOf(ProbabilityEditMode.Decrease) }
    var probabilityEditStep by remember { mutableFloatStateOf(0.2f) }
    var selectedProbabilityCell by remember { mutableStateOf<Pair<Int, Int>?>(null) }
    var lastGridSizeHapticStep by remember { mutableIntStateOf(tuning.gridSize) }
    val gridSliderInteractionSource = remember { MutableInteractionSource() }
    val isGridSliderDragged by gridSliderInteractionSource.collectIsDraggedAsState()

    // 颜色校准相关
    var isCalibrating by remember { mutableStateOf(false) }
    var calibrationData by remember { mutableStateOf<ColorCalibration?>(null) }
    var calibrationBitmap by remember { mutableStateOf<android.graphics.Bitmap?>(null) }

    val context = androidx.compose.ui.platform.LocalContext.current
    val haptic = LocalHapticFeedback.current
    val pnpCalculator = remember { PnPDistanceCalculator() }
    val detector = remember { BrightSpotDetector() }

    LaunchedEffect(tuning) {
        detector.minPixelBrightness = tuning.minPixelBrightness
        detector.minTotalBrightness = tuning.minTotalBrightness
        detector.dynamicThresholdMin = tuning.dynamicThresholdMin
        detector.dynamicThresholdRatio = tuning.dynamicThresholdRatio
        detector.minPixelCount = tuning.minPixelCount
        detector.minRefineRadius = tuning.detectorMinRefineRadius
        detector.maxBrightPixelRatio = tuning.maxBrightPixelRatio
        detector.maxBrightPixelCountMin = tuning.maxBrightPixelCountMin
        detector.hsvMinSaturation = tuning.hsvMinSaturation
        detector.hsvMinValue = tuning.hsvMinValue
        detector.hsvHueWeight = tuning.hsvHueWeight
        detector.hsvSatWeight = tuning.hsvSatWeight
        detector.hsvValWeight = tuning.hsvValWeight
        detector.hsvSharpness = tuning.hsvSharpness
        detector.triShapePenaltyWeight = tuning.triShapePenaltyWeight
        detector.triGeometryPenaltyWeight = tuning.triGeometryPenaltyWeight
        detector.triPurityBoostWeight = tuning.triPurityBoostWeight
        detector.triCenterBlackWeight = tuning.triCenterBlackWeight
        detector.triColorScoreWeight = tuning.triColorScoreWeight
        detector.topNCandidatesPerColor = tuning.topNCandidatesPerColor
        detector.minPairDistanceMultiplier = tuning.minPairDistanceMultiplier
        detector.maxPairDistanceMultiplier = tuning.maxPairDistanceMultiplier
        detector.darkBackgroundBoost = tuning.darkBackgroundBoost
        detector.probabilityCandidateExponent = tuning.probabilityCandidateExponent
        detector.probabilityGroupExponent = tuning.probabilityGroupExponent
        detector.probabilityWeightFloor = tuning.probabilityWeightFloor
        detector.calibrationSearchStep = tuning.calibrationSearchStep
        detector.calibrationSearchRadius = tuning.calibrationSearchRadius
        detector.sampleTopPercent = tuning.sampleTopPercent

        pnpCalculator.estimatedFovDegrees = tuning.estimatedFovDegrees
        pnpCalculator.depthInitFallbackMm = tuning.pnpDepthInitFallbackMm
        pnpCalculator.solverMaxIterations = tuning.pnpSolverMaxIterations
        pnpCalculator.solverConvergenceErrorMm = tuning.pnpSolverConvergenceErrorMm
        pnpCalculator.solverLearningRate = tuning.pnpSolverLearningRate
        pnpCalculator.solverMinDepthMm = tuning.pnpSolverMinDepthMm
    }

    Box(modifier = Modifier.fillMaxSize()) {
        when {
            cameraPermissionState.status.isGranted -> {
                    Row(modifier = Modifier.fillMaxSize()) {
                        Box(
                            modifier = Modifier
                                .fillMaxHeight()
                                .aspectRatio(4f / 3f, matchHeightConstraintsFirst = true)
                        ) {
                            // 相机预览
                            CameraPreview(
                                modifier = Modifier.fillMaxSize(),
                                tuning = tuning,
                                isGridSizeAdjusting = isGridSliderDragged,
                                probabilityMatrix = probabilityMatrix,
                                probabilityEditEnabled = isProbabilityEditMode,
                                probabilityEditMode = probabilityEditMode,
                                probabilityEditStep = probabilityEditStep,
                                onProbabilityMatrixChange = {
                                    probabilityMatrix = it
                                },
                                onProbabilityCellSelected = { cell ->
                                    selectedProbabilityCell = cell
                                },
                                detector = detector,  // 传递detector实例
                                onFpsUpdate = { newFps ->
                                    fps = newFps
                                },
                                onBrightSpotsDetected = { spots, size ->
                                    detectedSpotsCount = spots.size
                                    detectedSpots = spots
                                    imageSize = size

                                    // 仅保留3点模式：RGB三色LED，计算完整6DOF姿态
                                    if (spots.size == 3 && size != null) {
                                        val redSpot = spots.find { it.color == LedColor.RED }
                                        val greenSpot = spots.find { it.color == LedColor.GREEN }
                                        val blueSpot = spots.find { it.color == LedColor.BLUE }

                                        pnpResult =
                                            if (redSpot != null && greenSpot != null && blueSpot != null) {
                                                pnpCalculator.calculate3PointPnP(
                                                    redPoint = redSpot.position,
                                                    greenPoint = greenSpot.position,
                                                    bluePoint = blueSpot.position,
                                                    triangleEdgeLength = tuning.knownTriangleEdgeLengthMm,
                                                    focalLength = null,
                                                    imageWidth = size.first,
                                                    imageHeight = size.second
                                                )
                                            } else {
                                                null  // 未检测到完整的RGB三色
                                            }
                                    } else {
                                        pnpResult = null
                                    }
                                },
                                onBitmapCaptured = { bitmap ->
                                    // 释放旧的bitmap
                                    calibrationBitmap?.recycle()
                                    calibrationBitmap = bitmap
                                },
                                captureBitmapForCalibration = isCalibrating
                            )

                            // 校准标记（校准模式下显示）
                            if (isCalibrating) {
                                // Canvas层：不拦截触摸事件，让事件穿透到CameraPreview进行ROI选择
                                Canvas(
                                    modifier = Modifier.fillMaxSize()
                                ) {
                                    val centerX = size.width / 2f
                                    val centerY = size.height / 2f
                                    // 使用屏幕最小尺寸计算，确保在横屏和竖屏下都有合适的大小
                                    val calibrationRadius = kotlin.math.min(size.width, size.height) * 0.20f

                                // 绘制校准圆框（彩虹渐变效果）
                                // 外圈 - 半透明白色
                                drawCircle(
                                    color = Color(0xAAFFFFFF),
                                    radius = calibrationRadius + 10f,
                                    center = Offset(centerX, centerY),
                                    style = Stroke(width = 4f)
                                )

                                // 中圈 - 彩色渐变提示
                                drawCircle(
                                    color = Color(0xCCFFFFFF),
                                    radius = calibrationRadius,
                                    center = Offset(centerX, centerY),
                                    style = Stroke(width = 8f)
                                )

                                // 内圈 - 半透明
                                drawCircle(
                                    color = Color(0x33FFFFFF),
                                    radius = calibrationRadius - 10f,
                                    center = Offset(centerX, centerY)
                                )

                                // 中心十字线
                                drawLine(
                                    color = Color(0xAAFFFFFF),
                                    start = Offset(centerX - 40f, centerY),
                                    end = Offset(centerX + 40f, centerY),
                                    strokeWidth = 3f
                                )
                                drawLine(
                                    color = Color(0xAAFFFFFF),
                                    start = Offset(centerX, centerY - 40f),
                                    end = Offset(centerX, centerY + 40f),
                                    strokeWidth = 3f
                                )

                                // 绘制RGB标识（圆框边缘）
                                drawContext.canvas.nativeCanvas.apply {
                                    val paint = android.graphics.Paint().apply {
                                        textSize = 36f
                                        textAlign = android.graphics.Paint.Align.CENTER
                                        isFakeBoldText = true
                                        setShadowLayer(8f, 0f, 0f, android.graphics.Color.BLACK)
                                    }

                                    // 红色标识（左）
                                    paint.color = android.graphics.Color.rgb(255, 51, 51)
                                    drawText("R", centerX - calibrationRadius - 50f, centerY + 12f, paint)

                                    // 绿色标识（上）
                                    paint.color = android.graphics.Color.rgb(51, 255, 51)
                                    drawText("G", centerX, centerY - calibrationRadius - 40f, paint)

                                    // 蓝色标识（右）
                                    paint.color = android.graphics.Color.rgb(51, 102, 255)
                                    drawText("B", centerX + calibrationRadius + 50f, centerY + 12f, paint)
                                }
                                }
                            }
                        }

                        // 右侧控制面板（常显，支持上下滚动）
                        Surface(
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxHeight(),
                            color = FlatUiColors.SidebarBackground
                        ) {
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .background(FlatUiColors.SidebarBackground)
                                    .padding(12.dp)
                            ) {
                            FlatIconActionButton(
                                modifier = Modifier
                                    .align(Alignment.TopEnd)
                                    .zIndex(2f),
                                icon = {
                                    Icon(
                                        imageVector = Icons.Filled.Settings,
                                        contentDescription = "设置",
                                        tint = FlatUiColors.TextPrimary
                                    )
                                },
                                onClick = {
                                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                                    showSettings = true
                                }
                            )

                            val scrollState = rememberScrollState()
                            // 控制面板部分（可滚动）
                            Column(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .verticalScroll(scrollState)
                                    .padding(top = 52.dp),
                                verticalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                if (isProbabilityEditMode) {
                                    FlatPanel {
                                        Text(
                                            text = "概率矩阵编辑中",
                                            color = FlatUiColors.TextPrimary,
                                            style = MaterialTheme.typography.titleMedium
                                        )
                                        Text(
                                            text = "直接在左侧摄像头画面的网格上点击或拖动即可编辑。",
                                            color = FlatUiColors.TextSecondary,
                                            style = MaterialTheme.typography.bodyMedium
                                        )
                                        selectedProbabilityCell?.let { (row, col) ->
                                            Text(
                                                text = "当前格: 行 ${row + 1}, 列 ${col + 1}, 值 ${"%.2f".format(probabilityMatrix.weightAt(row, col))}",
                                                color = FlatUiColors.TextMuted,
                                                style = MaterialTheme.typography.bodySmall
                                            )
                                        }
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                                        ) {
                                            ThemeModeButton(
                                                label = "增加",
                                                selected = probabilityEditMode == ProbabilityEditMode.Increase,
                                                onClick = {
                                                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                                                    probabilityEditMode = ProbabilityEditMode.Increase
                                                },
                                                modifier = Modifier.weight(1f)
                                            )
                                            ThemeModeButton(
                                                label = "减少",
                                                selected = probabilityEditMode == ProbabilityEditMode.Decrease,
                                                onClick = {
                                                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                                                    probabilityEditMode = ProbabilityEditMode.Decrease
                                                },
                                                modifier = Modifier.weight(1f)
                                            )
                                        }
                                        Text(
                                            text = "每次调整 ${"%.2f".format(probabilityEditStep)}",
                                            color = FlatUiColors.Accent,
                                            style = MaterialTheme.typography.titleSmall
                                        )
                                        Slider(
                                            value = probabilityEditStep,
                                            onValueChange = { probabilityEditStep = it },
                                            valueRange = 0.05f..1.0f,
                                            steps = 18
                                        )
                                        Button(
                                            onClick = {
                                                haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                                                isProbabilityEditMode = false
                                            },
                                            modifier = Modifier.fillMaxWidth(),
                                            colors = ButtonDefaults.buttonColors(
                                                containerColor = FlatUiColors.Accent,
                                                contentColor = FlatUiColors.TextOnAccent
                                            ),
                                            shape = FlatUiShapes.Control,
                                            elevation = ButtonDefaults.buttonElevation(defaultElevation = 0.dp)
                                        ) {
                                            Text("退出编辑模式")
                                        }
                                    }
                                }
                                // 控制面板内容
                                Column(
                                    verticalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    // 精度控制
                                    Text(
                                        text = "PRECISION",
                                        color = FlatUiColors.TextMuted,
                                        style = MaterialTheme.typography.labelSmall,
                                        modifier = Modifier.padding(top = 4.dp)
                                    )
                                    Column(
                                        verticalArrangement = Arrangement.spacedBy(4.dp)
                                    ) {
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Text(
                                                text = "${tuning.gridSize}×${tuning.gridSize}",
                                                color = FlatUiColors.Accent,
                                                style = MaterialTheme.typography.titleMedium
                                            )
                                            Text(
                                                text = "${tuning.gridSize * tuning.gridSize}",
                                                color = FlatUiColors.TextMuted,
                                                style = MaterialTheme.typography.bodySmall
                                            )
                                        }
                                        Slider(
                                            value = tuning.gridSize.toFloat(),
                                            onValueChange = { value ->
                                                val rounded = kotlin.math.round(value / 8f).toInt() * 8
                                                val newGridSize = rounded.coerceIn(32, 512)
                                                if (newGridSize != lastGridSizeHapticStep) {
                                                    haptic.performHapticFeedback(HapticFeedbackType.SegmentTick)
                                                    lastGridSizeHapticStep = newGridSize
                                                }
                                                tuning = tuning.copy(gridSize = newGridSize)
                                            },
                                            onValueChangeFinished = {
                                                haptic.performHapticFeedback(HapticFeedbackType.GestureEnd)
                                            },
                                            valueRange = 32f..512f,
                                            steps = ((512 - 32) / 32 - 1),
                                            interactionSource = gridSliderInteractionSource,
                                            modifier = Modifier.fillMaxWidth()
                                        )
                                    }

                                    // 曝光控制（横向）
                                    Text(
                                        text = "EXPOSURE",
                                        color = FlatUiColors.TextMuted,
                                        style = MaterialTheme.typography.labelSmall,
                                        modifier = Modifier.padding(top = 4.dp)
                                    )
                                    Surface(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .height(80.dp),
                                        shape = MaterialTheme.shapes.small,
                                        color = FlatUiColors.Panel
                                    ) {
                                        Row(
                                            modifier = Modifier.fillMaxSize(),
                                            horizontalArrangement = Arrangement.SpaceEvenly,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            IconButton(
                                                onClick = {
                                                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                                                    if (tuning.exposureCompensation > -6) {
                                                        tuning = tuning.copy(
                                                            exposureCompensation = tuning.exposureCompensation - 1
                                                        )
                                                    }
                                                }
                                            ) {
                                                Text(
                                                    "-",
                                                    style = MaterialTheme.typography.headlineMedium,
                                                    color = FlatUiColors.TextPrimary
                                                )
                                            }
                                            Text(
                                                text = when {
                                                    tuning.exposureCompensation > 0 -> "+${tuning.exposureCompensation}"
                                                    else -> "${tuning.exposureCompensation}"
                                                },
                                                color = FlatUiColors.Accent,
                                                style = MaterialTheme.typography.headlineLarge
                                            )
                                            IconButton(
                                                onClick = {
                                                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                                                    if (tuning.exposureCompensation < 6) {
                                                        tuning = tuning.copy(
                                                            exposureCompensation = tuning.exposureCompensation + 1
                                                        )
                                                    }
                                                }
                                            ) {
                                                Text(
                                                    "+",
                                                    style = MaterialTheme.typography.headlineMedium,
                                                    color = FlatUiColors.TextPrimary
                                                )
                                            }
                                        }
                                    }

                                    // ROI优化开关（仅在3点模式显示）
                                    if (targetSpotCount == 3) {
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Text(
                                                text = "ROI OPTIMIZATION",
                                                color = FlatUiColors.TextMuted,
                                                style = MaterialTheme.typography.labelSmall
                                            )
                                            Surface(
                                                modifier = Modifier
                                                    .height(36.dp)
                                                    .width(64.dp),
                                                shape = MaterialTheme.shapes.small,
                                                color = if (tuning.enableRoiOptimization) Color(0xFF2CA36B) else Color(
                                                    0xFFCCCCCC
                                                ),
                                                shadowElevation = 0.dp
                                            ) {
                                                TextButton(
                                                    onClick = {
                                                        haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                                                        tuning = tuning.copy(
                                                            enableRoiOptimization = !tuning.enableRoiOptimization
                                                        )
                                                    },
                                                    modifier = Modifier.fillMaxSize(),
                                                    contentPadding = PaddingValues(0.dp)
                                                ) {
                                                    Text(
                                                        text = if (tuning.enableRoiOptimization) "ON" else "OFF",
                                                        color = if (tuning.enableRoiOptimization) Color.White else Color(
                                                            0xFF666666
                                                        ),
                                                        style = MaterialTheme.typography.labelMedium
                                                    )
                                                }
                                            }
                                        }

                                        // 校准按钮（仅在3点模式显示）
                                        HorizontalDivider(
                                            modifier = Modifier.padding(vertical = 4.dp),
                                            thickness = DividerDefaults.Thickness,
                                            color = FlatUiColors.Border
                                        )
                                        Text(
                                            text = "CALIBRATION",
                                            color = FlatUiColors.TextMuted,
                                            style = MaterialTheme.typography.labelSmall
                                        )
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            // 清除校准按钮（仅在已校准状态显示）
                                            if (calibrationData != null && !isCalibrating) {
                                                Surface(
                                                    modifier = Modifier
                                                        .weight(1f)
                                                        .height(40.dp),
                                                    shape = MaterialTheme.shapes.small,
                                                    color = Color(0xFFE35D5B),
                                                    shadowElevation = 0.dp
                                                ) {
                                                    TextButton(
                                                        onClick = {
                                                            haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                                                            calibrationData = null
                                                            detector.setCalibration(null)
                                                           Log.d(
                                                                "Calibration",
                                                                "已清除校准数据"
                                                            )
                                                            Toast.makeText(
                                                                context,
                                                                "已清除校准",
                                                                Toast.LENGTH_SHORT
                                                            ).show()
                                                        },
                                                        modifier = Modifier.fillMaxSize(),
                                                        contentPadding = PaddingValues(0.dp)
                                                    ) {
                                                        Text(
                                                            text = "? 清除",
                                                            color = Color.White,
                                                            style = MaterialTheme.typography.labelMedium
                                                        )
                                                    }
                                                }
                                            }

                                            // 校准/确认按钮
                                            Surface(
                                                modifier = Modifier
                                                    .weight(1f)
                                                    .height(40.dp),
                                                shape = MaterialTheme.shapes.small,
                                                color = if (isCalibrating) Color(0xFFF0A229) else Color(
                                                    0xFF0078D4
                                                ),
                                                shadowElevation = 0.dp
                                            ) {
                                                TextButton(
                                                    onClick = {
                                                        haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                                                        if (isCalibrating) {
                                                            // 执行校准
                                                            try {
                                                                calibrationBitmap?.let { originalBitmap ->
                                                                    if (!originalBitmap.isRecycled) {
                                                                        // 立即创建一个不可变副本，防止在校准过程中被回收
                                                                        val bitmap =
                                                                            originalBitmap.copy(
                                                                                originalBitmap.config
                                                                                    ?: android.graphics.Bitmap.Config.ARGB_8888,
                                                                                false
                                                                            )

                                                                        val width = bitmap.width
                                                                        val height = bitmap.height

                                                                        // 定义校准圆框的范围（图像坐标）
                                                                        val centerX = width / 2
                                                                        val centerY = height / 2
                                                                        val calibrationRadius =
                                                                            (width * 0.25f).toInt()  // 圆框半径为图像宽度的25%

                                                                        Log.d(
                                                                            "Calibration",
                                                                            "=== 开始自动校准 ==="
                                                                        )
                                                                        Log.d(
                                                                            "Calibration",
                                                                            "图像尺寸: ${width}x${height}"
                                                                        )
                                                                        Log.d(
                                                                            "Calibration",
                                                                            "搜索区域: 中心($centerX,$centerY), 半径=$calibrationRadius"
                                                                        )

                                                                        // 在圆框内搜索最红、最绿、最蓝的点
                                                                        val result =
                                                                            detector.findRGBColorsInCircle(
                                                                                bitmap,
                                                                                centerX,
                                                                                centerY,
                                                                                calibrationRadius
                                                                            )

                                                                        // 使用完后立即回收副本
                                                                        bitmap.recycle()

                                                                        if (result != null) {
                                                                            val (redColor, greenColor, blueColor) = result

                                                                            // 显示找到的颜色详情
                                                                            fun analyzeColor(
                                                                                name: String,
                                                                                color: Triple<Float, Float, Float>
                                                                            ): String {
                                                                                val (r, g, b) = color
                                                                                val total = r + g + b
                                                                                val rPct =
                                                                                    (r / total * 100).toInt()
                                                                                val gPct =
                                                                                    (g / total * 100).toInt()
                                                                                val bPct =
                                                                                    (b / total * 100).toInt()
                                                                                return "$name: RGB(${r.toInt()},${g.toInt()},${b.toInt()}) 占比:R${rPct}% G${gPct}% B${bPct}% 亮度:${total.toInt()}"
                                                                            }

                                                                            Log.d(
                                                                                "Calibration",
                                                                                "=== 校准结果 ==="
                                                                            )
                                                                            Log.d(
                                                                                "Calibration",
                                                                                analyzeColor(
                                                                                    "?? 红色LED",
                                                                                    redColor
                                                                                )
                                                                            )
                                                                            Log.d(
                                                                                "Calibration",
                                                                                analyzeColor(
                                                                                    "?? 绿色LED",
                                                                                    greenColor
                                                                                )
                                                                            )
                                                                            Log.d(
                                                                                "Calibration",
                                                                                analyzeColor(
                                                                                    "?? 蓝色LED",
                                                                                    blueColor
                                                                                )
                                                                            )

                                                                            // 保存校准数据
                                                                            calibrationData =
                                                                                ColorCalibration(
                                                                                    redColor,
                                                                                    greenColor,
                                                                                    blueColor
                                                                                )
                                                                            detector.setCalibration(
                                                                                calibrationData
                                                                            )
                                                                            Log.d(
                                                                                "Calibration",
                                                                                "? 校准数据已保存，现在将使用这些颜色进行检测"
                                                                            )

                                                                            // 显示成功提示
                                                                            val totalBrightness =
                                                                                (redColor.first + redColor.second + redColor.third +
                                                                                        greenColor.first + greenColor.second + greenColor.third +
                                                                                        blueColor.first + blueColor.second + blueColor.third) / 3
                                                                            android.widget.Toast.makeText(
                                                                                context,
                                                                                "? 校准成功！平均亮度: ${totalBrightness.toInt()}",
                                                                                android.widget.Toast.LENGTH_SHORT
                                                                            ).show()
                                                                        } else {
                                                                            // 搜索失败
                                                                            Log.e(
                                                                                "Calibration",
                                                                                "? 未在圆框内找到有效的RGB颜色"
                                                                            )

                                                                            Toast.makeText(
                                                                                context,
                                                                                "校准失败！\n请确保RGB三色LED都在圆框内",
                                                                                android.widget.Toast.LENGTH_LONG
                                                                            ).show()

                                                                            // 清除之前的校准数据
                                                                            calibrationData = null
                                                                            detector.setCalibration(null)
                                                                        }
                                                                    }
                                                                }
                                                            } catch (e: Exception) {
                                                                e.printStackTrace()
                                                                android.widget.Toast.makeText(
                                                                    context,
                                                                    "校准出错：${e.message}",
                                                                    android.widget.Toast.LENGTH_LONG
                                                                ).show()
                                                            }
                                                        }
                                                        isCalibrating = !isCalibrating
                                                    },
                                                    modifier = Modifier.fillMaxSize(),
                                                    contentPadding = PaddingValues(0.dp)
                                                ) {
                                                    Text(
                                                        text = if (isCalibrating) "? 确认" else if (calibrationData != null) "? 已校准" else "校准",
                                                        color = Color.White,
                                                        style = MaterialTheme.typography.labelMedium
                                                    )
                                                }
                                            }
                                        }
                                    }

                                    // PnP距离设置（三色模式）
                                    if (targetSpotCount == 3) {
                                        HorizontalDivider(
                                            modifier = Modifier.padding(vertical = 4.dp),
                                            thickness = DividerDefaults.Thickness,
                                            color = FlatUiColors.Border
                                        )
                                        Text(
                                            text = "△ EDGE LENGTH",
                                            color = FlatUiColors.TextMuted,
                                            style = MaterialTheme.typography.labelSmall
                                        )
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            IconButton(
                                                onClick = {
                                                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                                                    if (tuning.knownTriangleEdgeLengthMm > 10f) {
                                                        tuning = tuning.copy(
                                                            knownTriangleEdgeLengthMm = tuning.knownTriangleEdgeLengthMm - 10f
                                                        )
                                                    }
                                                },
                                                modifier = Modifier.size(40.dp)
                                            ) {
                                                Text(
                                                    "-",
                                                    style = MaterialTheme.typography.headlineMedium,
                                                    color = FlatUiColors.TextPrimary
                                                )
                                            }
                                            Column(
                                                horizontalAlignment = Alignment.CenterHorizontally,
                                                modifier = Modifier.weight(1f)
                                            ) {
                                                Text(
                                                    text = "%.0f".format(tuning.knownTriangleEdgeLengthMm),
                                                    color = FlatUiColors.Accent,
                                                    style = MaterialTheme.typography.headlineMedium
                                                )
                                                Text(
                                                    text = "mm",
                                                    color = FlatUiColors.TextMuted,
                                                    style = MaterialTheme.typography.labelSmall
                                                )
                                            }
                                            IconButton(
                                                onClick = {
                                                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                                                    tuning = tuning.copy(
                                                        knownTriangleEdgeLengthMm = tuning.knownTriangleEdgeLengthMm + 10f
                                                    )
                                                },
                                                modifier = Modifier.size(40.dp)
                                            ) {
                                                Text(
                                                    "+",
                                                    style = MaterialTheme.typography.headlineMedium,
                                                    color = FlatUiColors.TextPrimary
                                                )
                                            }
                                        }
                                    }

                                    HorizontalDivider(
                                        modifier = Modifier.padding(vertical = 8.dp),
                                        thickness = DividerDefaults.Thickness, color = FlatUiColors.Border
                                    )

                                    // 状态信息部分
                                    Column(
                                        verticalArrangement = Arrangement.spacedBy(12.dp)
                                    ) {
                                        // 状态信息标题
                                        Text(
                                            text = "STATUS",
                                            color = FlatUiColors.TextMuted,
                                            style = MaterialTheme.typography.labelSmall
                                        )

                                        // 检测状态
                                        Row(
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                                        ) {
                                            Surface(
                                                modifier = Modifier.size(12.dp),
                                                shape = MaterialTheme.shapes.small,
                                                color = if (detectedSpotsCount > 0) Color(0xFF2CA36B) else Color(
                                                    0xFFCCCCCC
                                                ),
                                                shadowElevation = 0.dp
                                            ) {}
                                            Column(modifier = Modifier.weight(1f)) {
                                                Text(
                                                    text = "DETECTED",
                                                    color = FlatUiColors.TextMuted,
                                                    style = MaterialTheme.typography.labelSmall
                                                )
                                                Text(
                                                    text = "$detectedSpotsCount / $targetSpotCount",
                                                    color = if (detectedSpotsCount == targetSpotCount) Color(
                                                        0xFF4CAF50
                                                    ) else Color(0xFFF0A229),
                                                    style = MaterialTheme.typography.titleLarge
                                                )
                                            }
                                        }

                                        // 网格和区块信息（横向排列）
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween
                                        ) {
                                            Column {
                                                Text(
                                                    text = "GRID",
                                                    color = FlatUiColors.TextMuted,
                                                    style = MaterialTheme.typography.labelSmall
                                                )
                                                Text(
                                                    text = "${tuning.gridSize}×${tuning.gridSize}",
                                                    color = FlatUiColors.TextPrimary,
                                                    style = MaterialTheme.typography.titleMedium
                                                )
                                            }

                                            Column(horizontalAlignment = Alignment.End) {
                                                Text(
                                                    text = "BLOCKS",
                                                    color = FlatUiColors.TextMuted,
                                                    style = MaterialTheme.typography.labelSmall
                                                )
                                                Text(
                                                    text = "${tuning.gridSize * tuning.gridSize}",
                                                    color = FlatUiColors.TextPrimary,
                                                    style = MaterialTheme.typography.titleMedium
                                                )
                                            }
                                        }

                                        // 帧率显示
                                        Column {
                                            Text(
                                                text = "FPS",
                                                color = FlatUiColors.TextMuted,
                                                style = MaterialTheme.typography.labelSmall
                                            )
                                            Text(
                                                text = "%.1f".format(fps),
                                                color = if (fps > 20f) Color(0xFF2CA36B) else if (fps > 10f) Color(
                                                    0xFFFF9800
                                                ) else Color(0xFFE35D5B),
                                                style = MaterialTheme.typography.titleLarge
                                            )
                                        }

                                        // PnP结果显示（三色模式）
                                        if (pnpResult != null) {
                                            HorizontalDivider(
                                                modifier = Modifier.padding(vertical = 12.dp),
                                                thickness = DividerDefaults.Thickness,
                                                color = FlatUiColors.Border
                                            )

                                            // 距离、方位角、仰角（横向排列）
                                            Row(
                                                modifier = Modifier.fillMaxWidth(),
                                                horizontalArrangement = Arrangement.SpaceBetween
                                            ) {
                                                Column {
                                                    Text(
                                                        text = "DISTANCE",
                                                        color = FlatUiColors.TextMuted,
                                                        style = MaterialTheme.typography.labelSmall
                                                    )
                                                    Text(
                                                        text = pnpCalculator.formatDistance(pnpResult!!.distance),
                                                        color = FlatUiColors.Accent,
                                                        style = MaterialTheme.typography.titleMedium
                                                    )
                                                }

                                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                                    Text(
                                                        text = "AZIMUTH",
                                                        color = FlatUiColors.TextMuted,
                                                        style = MaterialTheme.typography.labelSmall
                                                    )
                                                    Text(
                                                        text = pnpCalculator.formatAngle(pnpResult!!.azimuth),
                                                        color = FlatUiColors.TextPrimary,
                                                        style = MaterialTheme.typography.titleMedium
                                                    )
                                                }

                                                Column(horizontalAlignment = Alignment.End) {
                                                    Text(
                                                        text = "ELEVATION",
                                                        color = FlatUiColors.TextMuted,
                                                        style = MaterialTheme.typography.labelSmall
                                                    )
                                                    Text(
                                                        text = pnpCalculator.formatAngle(pnpResult!!.elevation),
                                                        color = FlatUiColors.TextPrimary,
                                                        style = MaterialTheme.typography.titleMedium
                                                    )
                                                }
                                            }

                                            HorizontalDivider(
                                                modifier = Modifier.padding(vertical = 8.dp),
                                                thickness = DividerDefaults.Thickness,
                                                color = FlatUiColors.Border
                                            )

                                            if (pnpResult!!.pose6DOF != null) {
                                                // 3点模式：显示完整6DOF姿态
                                                Column(
                                                    verticalArrangement = Arrangement.spacedBy(8.dp)
                                                ) {
                                                    Text(
                                                        text = "6DOF POSE",
                                                        color = FlatUiColors.Accent,
                                                        style = MaterialTheme.typography.titleSmall
                                                    )

                                                    Column(
                                                        verticalArrangement = Arrangement.spacedBy(4.dp)
                                                    ) {
                                                        Row(
                                                            modifier = Modifier.fillMaxWidth(),
                                                            horizontalArrangement = Arrangement.SpaceBetween
                                                        ) {
                                                            Text(
                                                                text = "CENTER",
                                                                color = Color(0xFF95A5A6),
                                                                style = MaterialTheme.typography.labelSmall
                                                            )
                                                            Text(
                                                                text = pnpCalculator.formatPoint3D(
                                                                    pnpResult!!.pose6DOF!!.position
                                                                ),
                                                                color = FlatUiColors.Accent,
                                                                style = MaterialTheme.typography.bodySmall
                                                            )
                                                        }
                                                    }

                                                    HorizontalDivider(
                                                        modifier = Modifier.padding(vertical = 4.dp),
                                                        thickness = DividerDefaults.Thickness,
                                                        color = FlatUiColors.Border
                                                    )

                                                    Text(
                                                        text = "ORIENTATION",
                                                        color = FlatUiColors.TextMuted,
                                                        style = MaterialTheme.typography.labelSmall
                                                    )

                                                    Row(
                                                        modifier = Modifier.fillMaxWidth(),
                                                        horizontalArrangement = Arrangement.SpaceBetween
                                                    ) {
                                                        Column {
                                                            Text(
                                                                text = "ROLL",
                                                                color = FlatUiColors.TextMuted,
                                                                style = MaterialTheme.typography.labelSmall
                                                            )
                                                            Text(
                                                                text = pnpCalculator.formatAngle(
                                                                    pnpResult!!.pose6DOF!!.roll
                                                                ),
                                                                color = Color(0xFFE35D5B),
                                                                style = MaterialTheme.typography.bodyMedium
                                                            )
                                                        }

                                                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                                            Text(
                                                                text = "PITCH",
                                                                color = FlatUiColors.TextMuted,
                                                                style = MaterialTheme.typography.labelSmall
                                                            )
                                                            Text(
                                                                text = pnpCalculator.formatAngle(
                                                                    pnpResult!!.pose6DOF!!.pitch
                                                                ),
                                                                color = Color(0xFF2CA36B),
                                                                style = MaterialTheme.typography.bodyMedium
                                                            )
                                                        }

                                                        Column(horizontalAlignment = Alignment.End) {
                                                            Text(
                                                                text = "YAW",
                                                                color = FlatUiColors.TextMuted,
                                                                style = MaterialTheme.typography.labelSmall
                                                            )
                                                            Text(
                                                                text = pnpCalculator.formatAngle(
                                                                    pnpResult!!.pose6DOF!!.yaw
                                                                ),
                                                                color = Color(0xFF2F8FCE),
                                                                style = MaterialTheme.typography.bodyMedium
                                                            )
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            androidx.compose.animation.AnimatedVisibility(
                                visible = showSettings,
                                enter = slideInHorizontally(
                                    animationSpec = tween(durationMillis = 260),
                                    initialOffsetX = { fullWidth -> fullWidth }
                                ) + fadeIn(animationSpec = tween(durationMillis = 220)),
                                exit = slideOutHorizontally(
                                    animationSpec = tween(durationMillis = 220),
                                    targetOffsetX = { fullWidth -> fullWidth }
                                ) + fadeOut(animationSpec = tween(durationMillis = 180)),
                                modifier = Modifier
                                    .fillMaxSize()
                                    .zIndex(20f)
                            ) {
                                SettingsScreen(
                                    onDismiss = { showSettings = false },
                                    themeMode = themeMode,
                                    onThemeModeChange = onThemeModeChange,
                                    tuning = tuning,
                                    onTuningChange = { tuning = it },
                                    probabilityMatrix = probabilityMatrix,
                                    onProbabilityMatrixEditRequest = {
                                        selectedProbabilityCell = null
                                        isProbabilityEditMode = true
                                        showSettings = false
                                    },
                                    modifier = Modifier.fillMaxSize()
                                )
                            }
                            }
                        }
                    }
            }
            cameraPermissionState.status.shouldShowRationale -> {
                // 需要向用户解释为什么需要相机权限
                PermissionRationaleContent(
                    onRequestPermission = { cameraPermissionState.launchPermissionRequest() }
                )
            }
            else -> {
                // 首次请求权限
                PermissionRequestContent(
                    onRequestPermission = { cameraPermissionState.launchPermissionRequest() }
                )
            }
        }
    }
}

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
                .fillMaxSize()
                .padding(20.dp),
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
private fun SettingsOverviewContent(
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
                onCheckedChange = { onTuningChange(tuning.copy(enableRoiOptimization = it)) }
            )
            FloatSliderSetting(
                title = "检测阈值",
                description = "亮点候选最低得分。升高可减少误检，降低可提升弱光目标通过率。",
                value = tuning.detectionThreshold,
                valueText = "%.0f".format(tuning.detectionThreshold),
                range = 10f..200f,
                steps = 18,
                onValueChange = { onTuningChange(tuning.copy(detectionThreshold = it)) }
            )
            IntSliderSetting(
                title = "检测网格",
                description = "每帧采样网格边长。越大越精细，但计算量近似按平方增加。",
                value = tuning.gridSize,
                valueText = "${tuning.gridSize} x ${tuning.gridSize}",
                range = 32..512,
                step = 8,
                onValueChange = { onTuningChange(tuning.copy(gridSize = it)) }
            )
            IntSliderSetting(
                title = "工作分辨率上限",
                description = "图像会先缩放到此最大边长再做检测。更高更精细，更低更省算力。",
                value = tuning.workingMaxDimension,
                valueText = "${tuning.workingMaxDimension}px",
                range = 320..1920,
                step = 32,
                onValueChange = { onTuningChange(tuning.copy(workingMaxDimension = it)) }
            )
            IntSliderSetting(
                title = "曝光补偿",
                description = "直接影响画面亮度和高光饱和，间接影响识别稳定性。",
                value = tuning.exposureCompensation,
                valueText = tuning.exposureCompensation.toString(),
                range = -6..6,
                step = 1,
                onValueChange = { onTuningChange(tuning.copy(exposureCompensation = it)) }
            )
        }

        SettingsSection(
            title = "ROI 搜索",
            description = "控制跟踪目标后的局部搜索范围、扩张速度和失败回退策略。"
        ) {
            FloatSliderSetting(
                title = "初始半径倍率",
                description = "以上一帧三点外接圆半径为基础扩大多少倍作为首个 ROI。",
                value = tuning.roiInitialRadiusMultiplier,
                valueText = "%.2f".format(tuning.roiInitialRadiusMultiplier),
                range = 0.5f..4f,
                steps = 13,
                onValueChange = { onTuningChange(tuning.copy(roiInitialRadiusMultiplier = it)) }
            )
            FloatSliderSetting(
                title = "扩张倍率",
                description = "ROI 每次未命中后的放大比例。更大能更快重新找回目标，但更耗时。",
                value = tuning.roiExpandFactor,
                valueText = "%.2f".format(tuning.roiExpandFactor),
                range = 1.05f..2f,
                steps = 18,
                onValueChange = { onTuningChange(tuning.copy(roiExpandFactor = it)) }
            )
            FloatSliderSetting(
                title = "最大搜索半径比例",
                description = "ROI 最大可扩到画面长边的比例，过小可能丢目标，过大则接近全图搜索。",
                value = tuning.roiMaxRadiusRatio,
                valueText = "%.2f".format(tuning.roiMaxRadiusRatio),
                range = 0.1f..1f,
                steps = 17,
                onValueChange = { onTuningChange(tuning.copy(roiMaxRadiusRatio = it)) }
            )
            IntSliderSetting(
                title = "最大扩张次数",
                description = "单帧局部搜索最多扩几轮，限制最坏情况下的计算时间。",
                value = tuning.roiMaxExpansions,
                valueText = tuning.roiMaxExpansions.toString(),
                range = 1..20,
                step = 1,
                onValueChange = { onTuningChange(tuning.copy(roiMaxExpansions = it)) }
            )
            FloatSliderSetting(
                title = "稳定性容差",
                description = "与上一轮三点半径变化的允许比例。越低越严格，越高越容易接受突变。",
                value = tuning.roiStabilityTolerance,
                valueText = "%.2f".format(tuning.roiStabilityTolerance),
                range = 0f..1f,
                steps = 19,
                onValueChange = { onTuningChange(tuning.copy(roiStabilityTolerance = it)) }
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
                        tuning.copy(
                            roiRecentAttempts = attempts,
                            roiFailureThreshold = tuning.roiFailureThreshold.coerceAtMost(attempts)
                        )
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
                onValueChange = { onTuningChange(tuning.copy(roiFailureThreshold = it)) }
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
                onValueChange = { onTuningChange(tuning.copy(refineIterations = it)) }
            )
            FloatSliderSetting(
                title = "细化半径倍率",
                description = "原图细化窗口大小，越大越不容易偏移丢点，但更易受背景干扰。",
                value = tuning.refineRadiusMultiplier,
                valueText = "%.2f".format(tuning.refineRadiusMultiplier),
                range = 0.5f..4f,
                steps = 13,
                onValueChange = { onTuningChange(tuning.copy(refineRadiusMultiplier = it)) }
            )
            IntSliderSetting(
                title = "最小细化半径",
                description = "细化窗口的像素下限，防止小网格时窗口过小。",
                value = tuning.refineMinRadiusPx,
                valueText = "${tuning.refineMinRadiusPx}px",
                range = 1..20,
                step = 1,
                onValueChange = { onTuningChange(tuning.copy(refineMinRadiusPx = it)) }
            )
            FloatSliderSetting(
                title = "背景抑制权重",
                description = "越偏向黑背景的像素会额外加分，适合亮点在黑底上的场景。",
                value = tuning.refineBackgroundDarkness,
                valueText = "%.2f".format(tuning.refineBackgroundDarkness),
                range = 0f..2f,
                steps = 19,
                onValueChange = { onTuningChange(tuning.copy(refineBackgroundDarkness = it)) }
            )
            FloatSliderSetting(
                title = "高斯聚焦系数",
                description = "越小越强调中心区域，越大越均匀地参考整个细化窗口。",
                value = tuning.refineGaussianSigmaFactor,
                valueText = "%.2f".format(tuning.refineGaussianSigmaFactor),
                range = 0.1f..1.5f,
                steps = 13,
                onValueChange = { onTuningChange(tuning.copy(refineGaussianSigmaFactor = it)) }
            )
        }

        SettingsSection(
            title = "颜色筛选",
            description = "控制候选区域亮度、面积和 HSV 相似度的门槛。"
        ) {
            FloatSliderSetting("像素最小亮度", "校准取色时忽略过暗像素。", tuning.minPixelBrightness, "%.0f".format(tuning.minPixelBrightness), 0f..100f, 19) {
                onTuningChange(tuning.copy(minPixelBrightness = it))
            }
            FloatSliderSetting("区域最小亮度", "候选区域平均亮度下限，过低会被直接丢弃。", tuning.minTotalBrightness, "%.0f".format(tuning.minTotalBrightness), 0f..255f, 24) {
                onTuningChange(tuning.copy(minTotalBrightness = it))
            }
            FloatSliderSetting("动态阈值下限", "局部最亮像素不足时的保底阈值。", tuning.dynamicThresholdMin, "%.0f".format(tuning.dynamicThresholdMin), 0f..255f, 24) {
                onTuningChange(tuning.copy(dynamicThresholdMin = it))
            }
            FloatSliderSetting("动态阈值比例", "取区域峰值亮度的多少作为亮像素阈值。", tuning.dynamicThresholdRatio, "%.2f".format(tuning.dynamicThresholdRatio), 0.05f..1f, 18) {
                onTuningChange(tuning.copy(dynamicThresholdRatio = it))
            }
            IntSliderSetting("最少亮像素数", "候选区域至少包含多少个亮像素。", tuning.minPixelCount, tuning.minPixelCount.toString(), 0..20, 1) {
                onTuningChange(tuning.copy(minPixelCount = it))
            }
            IntSliderSetting("检测器细化半径下限", "候选点在检测器内部二次细化时使用的最小半径。", tuning.detectorMinRefineRadius, "${tuning.detectorMinRefineRadius}px", 1..20, 1) {
                onTuningChange(tuning.copy(detectorMinRefineRadius = it))
            }
            FloatSliderSetting("亮像素占比上限", "候选区域里亮像素占比过高时视为大块光斑并过滤。", tuning.maxBrightPixelRatio, "%.2f".format(tuning.maxBrightPixelRatio), 0.01f..0.5f, 24) {
                onTuningChange(tuning.copy(maxBrightPixelRatio = it))
            }
            IntSliderSetting("亮像素数量兜底上限", "小窗口场景下的亮像素绝对数量上限。", tuning.maxBrightPixelCountMin, tuning.maxBrightPixelCountMin.toString(), 1..50, 1) {
                onTuningChange(tuning.copy(maxBrightPixelCountMin = it))
            }
            FloatSliderSetting("最小饱和度", "过滤偏灰偏白高光，降低镜面反射误检。", tuning.hsvMinSaturation, "%.2f".format(tuning.hsvMinSaturation), 0f..1f, 19) {
                onTuningChange(tuning.copy(hsvMinSaturation = it))
            }
            FloatSliderSetting("最小明度", "过滤过暗噪声。", tuning.hsvMinValue, "%.2f".format(tuning.hsvMinValue), 0f..1f, 19) {
                onTuningChange(tuning.copy(hsvMinValue = it))
            }
            FloatSliderSetting("色相权重", "HSV 相似度中色相差异的影响强度。", tuning.hsvHueWeight, "%.2f".format(tuning.hsvHueWeight), 0f..3f, 29) {
                onTuningChange(tuning.copy(hsvHueWeight = it))
            }
            FloatSliderSetting("饱和度权重", "HSV 相似度中饱和度差异的影响强度。", tuning.hsvSatWeight, "%.2f".format(tuning.hsvSatWeight), 0f..3f, 29) {
                onTuningChange(tuning.copy(hsvSatWeight = it))
            }
            FloatSliderSetting("明度权重", "HSV 相似度中明度差异的影响强度。", tuning.hsvValWeight, "%.2f".format(tuning.hsvValWeight), 0f..3f, 29) {
                onTuningChange(tuning.copy(hsvValWeight = it))
            }
            FloatSliderSetting("HSV 锐度", "越大越挑剔，相似度会衰减得更快。", tuning.hsvSharpness, "%.2f".format(tuning.hsvSharpness), 1f..20f, 18) {
                onTuningChange(tuning.copy(hsvSharpness = it))
            }
            FloatSliderSetting("暗背景增益", "候选周围越暗，整体亮度加权越高。", tuning.darkBackgroundBoost, "%.2f".format(tuning.darkBackgroundBoost), 0f..2f, 19) {
                onTuningChange(tuning.copy(darkBackgroundBoost = it))
            }
        }

        SettingsSection(
            title = "三点组合评分",
            description = "控制 RGB 三点组合时的几何约束、颜色纯度和概率矩阵影响。"
        ) {
            IntSliderSetting("每色候选数", "每种颜色保留多少个候选点参与三点组合。更高更稳，但组合数会暴涨。", tuning.topNCandidatesPerColor, tuning.topNCandidatesPerColor.toString(), 1..64, 1) {
                onTuningChange(tuning.copy(topNCandidatesPerColor = it))
            }
            FloatSliderSetting("最小点距倍率", "约束三点之间最短允许距离，避免把噪点拼成一组。", tuning.minPairDistanceMultiplier, "%.2f".format(tuning.minPairDistanceMultiplier), 0.5f..10f, 18) {
                onTuningChange(tuning.copy(minPairDistanceMultiplier = it))
            }
            FloatSliderSetting("最大点距倍率", "约束三点之间最大允许距离，避免把分散目标拼成一组。", tuning.maxPairDistanceMultiplier, "%.2f".format(tuning.maxPairDistanceMultiplier), 2f..40f, 18) {
                onTuningChange(tuning.copy(maxPairDistanceMultiplier = it))
            }
            FloatSliderSetting("形状惩罚", "越高越偏向接近等边三角形的三点组合。", tuning.triShapePenaltyWeight, "%.2f".format(tuning.triShapePenaltyWeight), 0f..2f, 19) {
                onTuningChange(tuning.copy(triShapePenaltyWeight = it))
            }
            FloatSliderSetting("距离惩罚", "越高越严格限制边长是否落在期望范围内。", tuning.triGeometryPenaltyWeight, "%.2f".format(tuning.triGeometryPenaltyWeight), 0f..3f, 29) {
                onTuningChange(tuning.copy(triGeometryPenaltyWeight = it))
            }
            FloatSliderSetting("纯度加分", "越高越强调单色明显强于另外两色。", tuning.triPurityBoostWeight, "%.2f".format(tuning.triPurityBoostWeight), 0f..2f, 19) {
                onTuningChange(tuning.copy(triPurityBoostWeight = it))
            }
            FloatSliderSetting("中心偏黑加分", "越高越偏向三点中间为暗区的结构。", tuning.triCenterBlackWeight, "%.2f".format(tuning.triCenterBlackWeight), 0f..3f, 29) {
                onTuningChange(tuning.copy(triCenterBlackWeight = it))
            }
            FloatSliderSetting("颜色总分权重", "整体放大或缩小颜色分数在最终组合中的影响。", tuning.triColorScoreWeight, "%.2f".format(tuning.triColorScoreWeight), 0f..3f, 29) {
                onTuningChange(tuning.copy(triColorScoreWeight = it))
            }
            FloatSliderSetting("候选概率指数", "候选阶段概率矩阵影响强度，越高越压低低概率区域。", tuning.probabilityCandidateExponent, "%.2f".format(tuning.probabilityCandidateExponent), 0f..3f, 29) {
                onTuningChange(tuning.copy(probabilityCandidateExponent = it))
            }
            FloatSliderSetting("组合概率指数", "三点组合阶段概率矩阵影响强度。", tuning.probabilityGroupExponent, "%.2f".format(tuning.probabilityGroupExponent), 0f..3f, 29) {
                onTuningChange(tuning.copy(probabilityGroupExponent = it))
            }
            FloatSliderSetting("概率权重下限", "避免某些区域权重过低而被完全排除。", tuning.probabilityWeightFloor, "%.3f".format(tuning.probabilityWeightFloor), 0.001f..0.2f, 19) {
                onTuningChange(tuning.copy(probabilityWeightFloor = it))
            }
        }

        SettingsSection(
            title = "校准与 PnP",
            description = "控制颜色校准取样方式，以及姿态计算里的焦距与迭代求解参数。"
        ) {
            IntSliderSetting("校准搜索步长", "校准时在圆区域内隔多少像素取一个样点。步长越小越准，也更耗时。", tuning.calibrationSearchStep, tuning.calibrationSearchStep.toString(), 1..30, 1) {
                onTuningChange(tuning.copy(calibrationSearchStep = it))
            }
            IntSliderSetting("校准搜索半径", "每个校准样点周围取色窗口的半径。", tuning.calibrationSearchRadius, "${tuning.calibrationSearchRadius}px", 1..40, 1) {
                onTuningChange(tuning.copy(calibrationSearchRadius = it))
            }
            FloatSliderSetting("校准高亮像素比例", "取色时只平均最亮部分像素，避免背景污染。", tuning.sampleTopPercent, "%.2f".format(tuning.sampleTopPercent), 0.01f..0.8f, 19) {
                onTuningChange(tuning.copy(sampleTopPercent = it))
            }
            FloatSliderSetting("三点边长", "已知 RGB 三点构成的实际边长，直接影响距离解算结果。", tuning.knownTriangleEdgeLengthMm, "%.0f mm".format(tuning.knownTriangleEdgeLengthMm), 10f..1000f, 19) {
                onTuningChange(tuning.copy(knownTriangleEdgeLengthMm = it))
            }
            FloatSliderSetting("估算视场角", "用于在无真实焦距时估算相机焦距，直接影响 PnP 距离尺度。", tuning.estimatedFovDegrees, "%.1f°".format(tuning.estimatedFovDegrees), 20f..120f, 19) {
                onTuningChange(tuning.copy(estimatedFovDegrees = it))
            }
            FloatSliderSetting("深度初值", "PnP 求解初始化深度，主要影响收敛速度与稳定性。", tuning.pnpDepthInitFallbackMm, "%.0f mm".format(tuning.pnpDepthInitFallbackMm), 100f..5000f, 24) {
                onTuningChange(tuning.copy(pnpDepthInitFallbackMm = it))
            }
            IntSliderSetting("PnP 最大迭代数", "求解器最多迭代次数。更高更容易收敛，但更耗时。", tuning.pnpSolverMaxIterations, tuning.pnpSolverMaxIterations.toString(), 1..200, 1) {
                onTuningChange(tuning.copy(pnpSolverMaxIterations = it))
            }
            FloatSliderSetting("PnP 收敛误差", "误差低于该值就提前停止。越低越精确，但可能迭代更久。", tuning.pnpSolverConvergenceErrorMm, "%.3f".format(tuning.pnpSolverConvergenceErrorMm), 0.001f..5f, 19) {
                onTuningChange(tuning.copy(pnpSolverConvergenceErrorMm = it))
            }
            FloatSliderSetting("PnP 学习率", "每轮调整深度的步长。过低收敛慢，过高可能震荡。", tuning.pnpSolverLearningRate, "%.2f".format(tuning.pnpSolverLearningRate), 0.01f..1f, 19) {
                onTuningChange(tuning.copy(pnpSolverLearningRate = it))
            }
            FloatSliderSetting("PnP 最小深度", "限制求解结果深度不能低于该值，避免不合理解。", tuning.pnpSolverMinDepthMm, "%.0f mm".format(tuning.pnpSolverMinDepthMm), 1f..1000f, 19) {
                onTuningChange(tuning.copy(pnpSolverMinDepthMm = it))
            }
        }
    }
}

@Composable
private fun SettingsSection(
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
private fun FloatSliderSetting(
    title: String,
    description: String,
    value: Float,
    valueText: String,
    range: ClosedFloatingPointRange<Float>,
    steps: Int,
    onValueChange: (Float) -> Unit
) {
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
            onValueChange = onValueChange,
            valueRange = range,
            steps = steps
        )
    }
}

@Composable
private fun IntSliderSetting(
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
private fun BooleanSettingRow(
    title: String,
    description: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
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
                ThemeModeButton(label = "关", selected = !checked, onClick = { onCheckedChange(false) })
                ThemeModeButton(label = "开", selected = checked, onClick = { onCheckedChange(true) })
            }
        }
    }
}

enum class ProbabilityEditMode {
    Increase,
    Decrease
}

@Composable
fun PermissionRequestContent(onRequestPermission: () -> Unit) {
    val haptic = LocalHapticFeedback.current
    PermissionCard(
        title = "需要相机权限",
        body = "此应用需要访问您的相机来检测亮点。",
        actionLabel = "授予相机权限",
        onAction = {
            haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
            onRequestPermission()
        }
    )
}

@Composable
fun PermissionRationaleContent(onRequestPermission: () -> Unit) {
    val haptic = LocalHapticFeedback.current
    PermissionCard(
        title = "需要相机权限",
        body = "为了实时检测画面中的亮点，我们需要访问您的相机。\n\n没有相机权限，应用将无法正常工作。",
        actionLabel = "授予权限",
        onAction = {
            haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
            onRequestPermission()
        }
    )
}

enum class ThemeMode {
    Light,
    Dark
}

private data class FlatPalette(
    val appBackground: Color,
    val sidebarBackground: Color,
    val panel: Color,
    val border: Color,
    val accent: Color,
    val textPrimary: Color,
    val textSecondary: Color,
    val textMuted: Color,
    val textOnAccent: Color
)

private object FlatUiColors {
    private val light = FlatPalette(
        appBackground = Color(0xFFF4F6F8),
        sidebarBackground = Color(0xFFEAEFF3),
        panel = Color(0xFFFFFFFF),
        border = Color(0xFFD6DEE6),
        accent = Color(0xFF147EFB),
        textPrimary = Color(0xFF122033),
        textSecondary = Color(0xFF536276),
        textMuted = Color(0xFF738296),
        textOnAccent = Color(0xFFF8FBFF)
    )

    private val dark = FlatPalette(
        appBackground = Color(0xFF0F1722),
        sidebarBackground = Color(0xFF16202D),
        panel = Color(0xFF1B2635),
        border = Color(0xFF2B3A4E),
        accent = Color(0xFF56A3FF),
        textPrimary = Color(0xFFF3F7FC),
        textSecondary = Color(0xFFB7C4D4),
        textMuted = Color(0xFF8A9AAF),
        textOnAccent = Color(0xFF08111C)
    )

    private var palette by mutableStateOf(light)

    fun update(themeMode: ThemeMode) {
        palette = if (themeMode == ThemeMode.Dark) dark else light
    }

    val AppBackground get() = palette.appBackground
    val SidebarBackground get() = palette.sidebarBackground
    val Panel get() = palette.panel
    val Border get() = palette.border
    val Accent get() = palette.accent
    val TextPrimary get() = palette.textPrimary
    val TextSecondary get() = palette.textSecondary
    val TextMuted get() = palette.textMuted
    val TextOnAccent get() = palette.textOnAccent
}

private object FlatUiShapes {
    val Panel = RoundedCornerShape(4.dp)
    val Control = RoundedCornerShape(4.dp)
}

@Composable
private fun TriDotFlatTheme(
    themeMode: ThemeMode,
    content: @Composable () -> Unit
) {
    FlatUiColors.update(themeMode)
    MaterialTheme(
        colorScheme = if (themeMode == ThemeMode.Dark) {
            androidx.compose.material3.darkColorScheme(
                background = FlatUiColors.AppBackground,
                surface = FlatUiColors.Panel,
                primary = FlatUiColors.Accent,
                onPrimary = FlatUiColors.TextOnAccent,
                onSurface = FlatUiColors.TextPrimary
            )
        } else {
            lightColorScheme(
            background = FlatUiColors.AppBackground,
            surface = FlatUiColors.Panel,
            primary = FlatUiColors.Accent,
            onPrimary = FlatUiColors.TextOnAccent,
            onSurface = FlatUiColors.TextPrimary
            )
        },
        content = content
    )
}

@Composable
private fun FlatPanel(content: @Composable () -> Unit) {
    Surface(
        color = FlatUiColors.Panel,
        shape = FlatUiShapes.Panel,
        border = BorderStroke(1.dp, FlatUiColors.Border),
        shadowElevation = 0.dp
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            content()
        }
    }
}

@Composable
private fun FlatIconActionButton(
    onClick: () -> Unit,
    icon: @Composable () -> Unit,
    modifier: Modifier = Modifier
) {
    IconButton(
        modifier = modifier,
        colors = IconButtonDefaults.iconButtonColors(
            containerColor = FlatUiColors.Panel,
            contentColor = FlatUiColors.TextPrimary
        ),
        shape = FlatUiShapes.Control,
        onClick = onClick
    ) {
        icon()
    }
}

@Composable
private fun ThemeModeButton(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Button(
        onClick = onClick,
        modifier = modifier,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (selected) FlatUiColors.Accent else FlatUiColors.Panel,
            contentColor = if (selected) FlatUiColors.TextOnAccent else FlatUiColors.TextPrimary
        ),
        border = if (selected) null else BorderStroke(1.dp, FlatUiColors.Border),
        shape = FlatUiShapes.Control,
        elevation = ButtonDefaults.buttonElevation(defaultElevation = 0.dp)
    ) {
        Text(label)
    }
}

@Composable
private fun PermissionCard(
    title: String,
    body: String,
    actionLabel: String,
    onAction: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(FlatUiColors.AppBackground)
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Surface(
            color = FlatUiColors.Panel,
            shape = FlatUiShapes.Panel,
            border = BorderStroke(1.dp, FlatUiColors.Border),
            shadowElevation = 0.dp
        ) {
            Column(
                modifier = Modifier
                    .width(320.dp)
                    .padding(24.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.headlineMedium,
                    color = FlatUiColors.TextPrimary,
                    textAlign = TextAlign.Center
                )
                Text(
                    text = body,
                    style = MaterialTheme.typography.bodyLarge,
                    color = FlatUiColors.TextSecondary,
                    textAlign = TextAlign.Center
                )
                Button(
                    onClick = onAction,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = FlatUiColors.Accent,
                        contentColor = FlatUiColors.TextOnAccent
                    ),
                    shape = FlatUiShapes.Control,
                    elevation = ButtonDefaults.buttonElevation(defaultElevation = 0.dp)
                ) {
                    Text(actionLabel)
                }
            }
        }
    }
}







