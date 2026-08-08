package org.example.tridotpnp


import org.example.tridotpnp.ble.BleImuClient
import org.example.tridotpnp.camera.CameraPreview
import org.example.tridotpnp.config.AppTuningSettings
import org.example.tridotpnp.config.ProbabilityMatrix32x24
import org.example.tridotpnp.config.buildManualProbabilityMatrix32x24
import org.example.tridotpnp.tracking.FrameTrackingSnapshot
import org.example.tridotpnp.tracking.PnPDistanceCalculator
import org.example.tridotpnp.tracking.TrackedNodeState
import org.example.tridotpnp.tracking.VisionObservation
import org.example.tridotpnp.vision.BrightSpot
import org.example.tridotpnp.vision.ColorCalibration
import android.Manifest
import android.os.Build
import android.util.Log
import android.widget.Toast
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsDraggedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberMultiplePermissionsState
import com.google.accompanist.permissions.rememberPermissionState
import com.google.accompanist.permissions.shouldShowRationale
import kotlinx.coroutines.flow.collectLatest
import org.example.tridotpnp.ui.DetectionControlSection
import org.example.tridotpnp.ui.FlatIconActionButton
import org.example.tridotpnp.ui.FlatPanel
import org.example.tridotpnp.ui.FlatUiColors
import org.example.tridotpnp.ui.FlatUiShapes
import org.example.tridotpnp.ui.PermissionRequestContent
import org.example.tridotpnp.ui.PermissionRationaleContent
import org.example.tridotpnp.ui.ProbabilityEditMode
import org.example.tridotpnp.ui.SettingsScreen
import org.example.tridotpnp.ui.StatusSection
import org.example.tridotpnp.ui.ThemeMode
import org.example.tridotpnp.ui.ThemeModeButton
import org.example.tridotpnp.ui.TriDotFlatTheme

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
    var visionObservation by remember { mutableStateOf<VisionObservation?>(null) }
    var trackedNodes by remember { mutableStateOf<List<TrackedNodeState>>(emptyList()) }
    var imageSize by remember { mutableStateOf<Pair<Int, Int>?>(null) }
    var fps by remember { mutableFloatStateOf(0f) }
    var showSettings by remember { mutableStateOf(false) }
    var isProbabilityEditMode by remember { mutableStateOf(false) }
    var probabilityEditMode by remember { mutableStateOf(ProbabilityEditMode.Decrease) }
    var probabilityEditStep by remember { mutableFloatStateOf(0.2f) }
    var selectedProbabilityCell by remember { mutableStateOf<Pair<Int, Int>?>(null) }
    var lastGridSizeHapticStep by remember { mutableIntStateOf(tuning.gridSize) }
    val gridSliderInteractionSource = remember { MutableInteractionSource() }
    val isGridSliderDragged by gridSliderInteractionSource.collectIsDraggedAsState()

    var isCalibrating by remember { mutableStateOf(false) }
    var calibrationData by remember { mutableStateOf<ColorCalibration?>(null) }
    var calibrationBitmap by remember { mutableStateOf<android.graphics.Bitmap?>(null) }

    val context = LocalContext.current
    val haptic = LocalHapticFeedback.current
    val session = remember { AppSession(context.applicationContext) }
    val bleConnected by session.bleClient.connectedDevices.collectAsState()
    val bleScanning by session.bleClient.scanning.collectAsState()
    val currentTuning = rememberUpdatedState(tuning)

    val blePermissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        listOf(
            Manifest.permission.BLUETOOTH_SCAN,
            Manifest.permission.BLUETOOTH_CONNECT
        )
    } else {
        listOf(Manifest.permission.ACCESS_FINE_LOCATION)
    }
    val blePermissionState = rememberMultiplePermissionsState(blePermissions)

    LaunchedEffect(tuning) {
        session.applyTuning(tuning)
    }

    LaunchedEffect(tuning.enableBleImu, blePermissionState.allPermissionsGranted) {
        if (tuning.enableBleImu) {
            if (blePermissionState.allPermissionsGranted) {
                session.bleClient.startScan()
            } else {
                blePermissionState.launchMultiplePermissionRequest()
            }
        } else {
            session.bleClient.disconnectAll()
        }
    }

    LaunchedEffect(tuning.enableBleImu, tuning.enableEskfFusion) {
        if (!tuning.enableBleImu || !tuning.enableEskfFusion) return@LaunchedEffect
        session.bleClient.samples.collectLatest { sample ->
            val s = sample ?: return@collectLatest
            session.nodeTracker.onImu(s)
        }
    }

    DisposableEffect(Unit) {
        onDispose { session.dispose() }
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
                            detector = session.detector,
                            onFpsUpdate = { newFps ->
                                fps = newFps
                            },
                            onSpotsAnalyzed = { spots, size, candidates ->
                                session.computeTracking(
                                    spots = spots,
                                    imageSize = size,
                                    tuning = currentTuning.value,
                                    candidates = candidates
                                )
                            },
                            onBrightSpotsDetected = { spots, size, snap ->
                                detectedSpotsCount = spots.size
                                detectedSpots = spots
                                imageSize = size
                                if (size == null) {
                                    pnpResult = null
                                    visionObservation = null
                                    trackedNodes = emptyList()
                                } else {
                                    trackedNodes = snap?.trackedNodes ?: emptyList()
                                    visionObservation = snap?.visionObservation
                                    pnpResult = snap?.pnpResult
                                }
                            },
                            onBitmapCaptured = { bitmap ->
                                calibrationBitmap?.recycle()
                                calibrationBitmap = bitmap
                            },
                            captureBitmapForCalibration = isCalibrating
                        )

                        if (isCalibrating) {
                            CalibrationOverlay()
                        }
                    }

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
                            Column(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .verticalScroll(scrollState)
                                    .padding(top = 52.dp),
                                verticalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                if (isProbabilityEditMode) {
                                    ProbabilityEditSidebar(
                                        probabilityMatrix = probabilityMatrix,
                                        selectedProbabilityCell = selectedProbabilityCell,
                                        probabilityEditMode = probabilityEditMode,
                                        onProbabilityEditModeChange = { probabilityEditMode = it },
                                        probabilityEditStep = probabilityEditStep,
                                        onProbabilityEditStepChange = { probabilityEditStep = it },
                                        onExit = { isProbabilityEditMode = false }
                                    )
                                }
                                Column(
                                    verticalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    DetectionControlSection(
                                        tuning = tuning,
                                        onTuningChange = { tuning = it },
                                        lastGridSizeHapticStep = lastGridSizeHapticStep,
                                        onGridHapticStepChange = { lastGridSizeHapticStep = it },
                                        gridSliderInteractionSource = gridSliderInteractionSource,
                                        targetSpotCount = targetSpotCount,
                                        calibrationData = calibrationData,
                                        isCalibrating = isCalibrating,
                                        onClearCalibration = {
                                            calibrationData = null
                                            session.clearCalibration()
                                            Log.d("Calibration", "已清除校准数据")
                                            Toast.makeText(context, "已清除校准", Toast.LENGTH_SHORT).show()
                                        },
                                        onCalibrationAction = {
                                            if (isCalibrating) {
                                                calibrationData = session.executeCalibration(
                                                    context = context,
                                                    calibrationBitmap = calibrationBitmap
                                                )
                                            }
                                            isCalibrating = !isCalibrating
                                        }
                                    )

                                    StatusSection(
                                        detectedSpotsCount = detectedSpotsCount,
                                        targetSpotCount = targetSpotCount,
                                        tuning = tuning,
                                        fps = fps,
                                        pnpResult = pnpResult,
                                        pnpCalculator = session.pnpCalculator,
                                        visionObservation = visionObservation,
                                        trackedNodes = trackedNodes,
                                        bleConnectedCount = bleConnected.size,
                                        bleScanning = bleScanning
                                    )
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
                PermissionRationaleContent(
                    onRequestPermission = { cameraPermissionState.launchPermissionRequest() }
                )
            }
            else -> {
                PermissionRequestContent(
                    onRequestPermission = { cameraPermissionState.launchPermissionRequest() }
                )
            }
        }
    }
}

@Composable
private fun CalibrationOverlay() {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val centerX = size.width / 2f
        val centerY = size.height / 2f
        val calibrationRadius = kotlin.math.min(size.width, size.height) * 0.20f

        drawCircle(
            color = Color(0xAAFFFFFF),
            radius = calibrationRadius + 10f,
            center = Offset(centerX, centerY),
            style = Stroke(width = 4f)
        )
        drawCircle(
            color = Color(0xCCFFFFFF),
            radius = calibrationRadius,
            center = Offset(centerX, centerY),
            style = Stroke(width = 8f)
        )
        drawCircle(
            color = Color(0x33FFFFFF),
            radius = calibrationRadius - 10f,
            center = Offset(centerX, centerY)
        )
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

        drawContext.canvas.nativeCanvas.apply {
            val paint = android.graphics.Paint().apply {
                textSize = 36f
                textAlign = android.graphics.Paint.Align.CENTER
                isFakeBoldText = true
                setShadowLayer(8f, 0f, 0f, android.graphics.Color.BLACK)
            }
            paint.color = android.graphics.Color.rgb(255, 51, 51)
            drawText("R", centerX - calibrationRadius - 50f, centerY + 12f, paint)
            paint.color = android.graphics.Color.rgb(51, 255, 51)
            drawText("G", centerX, centerY - calibrationRadius - 40f, paint)
            paint.color = android.graphics.Color.rgb(51, 102, 255)
            drawText("B", centerX + calibrationRadius + 50f, centerY + 12f, paint)
        }
    }
}

@Composable
private fun ProbabilityEditSidebar(
    probabilityMatrix: ProbabilityMatrix32x24,
    selectedProbabilityCell: Pair<Int, Int>?,
    probabilityEditMode: ProbabilityEditMode,
    onProbabilityEditModeChange: (ProbabilityEditMode) -> Unit,
    probabilityEditStep: Float,
    onProbabilityEditStepChange: (Float) -> Unit,
    onExit: () -> Unit
) {
    val haptic = LocalHapticFeedback.current
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
                    onProbabilityEditModeChange(ProbabilityEditMode.Increase)
                },
                modifier = Modifier.weight(1f)
            )
            ThemeModeButton(
                label = "减少",
                selected = probabilityEditMode == ProbabilityEditMode.Decrease,
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                    onProbabilityEditModeChange(ProbabilityEditMode.Decrease)
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
            onValueChange = onProbabilityEditStepChange,
            valueRange = 0.05f..1.0f,
            steps = 18
        )
        Button(
            onClick = {
                haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                onExit()
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
