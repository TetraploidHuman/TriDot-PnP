package org.example.tridotpnp

import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.MotionEvent
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectDragGesturesAfterLongPress
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.*
import androidx.compose.material3.Text
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.pointerInteropFilter
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.FocusMeteringAction
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import androidx.core.graphics.scale
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

@androidx.camera.camera2.interop.ExperimentalCamera2Interop
@OptIn(ExperimentalCamera2Interop::class)
@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    exposureCompensation: Int = 0,
    gridSize: Int = 50,
    isGridSizeAdjusting: Boolean = false,
    probabilityMatrix: ProbabilityMatrix32x24? = null,
    probabilityEditEnabled: Boolean = false,
    probabilityEditMode: ProbabilityEditMode = ProbabilityEditMode.Decrease,
    probabilityEditStep: Float = 0.2f,
    onProbabilityMatrixChange: (ProbabilityMatrix32x24) -> Unit = {},
    onProbabilityCellSelected: (Pair<Int, Int>?) -> Unit = {},
    detector: BrightSpotDetector,  // 必须在工程中提供实现
    enableRoiOptimization: Boolean = true,  // 是否启用ROI优化
    onBrightSpotsDetected: (List<BrightSpot>, Pair<Int, Int>?) -> Unit = { _, _ -> },
    onBitmapCaptured: (Bitmap) -> Unit = {},
    captureBitmapForCalibration: Boolean = false,
    onFpsUpdate: (Float) -> Unit = {},  // 帧率更新回调
    // 分区参数（可调）
    maxDepth: Int = 13,            // 分区树最大深度（更大值 -> 更细）
    splitRatio: Float = 0.618f,   // 分割比例
    minLeafSizePx: Int = 3,       // 叶子最小边长（像素）
    maxDrawDepth: Int = 13         // 实际绘制的最大深度（<= maxDepth）
) {
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    var brightSpots by remember { mutableStateOf<List<BrightSpot>>(emptyList()) }
    var previewSize by remember { mutableStateOf<Pair<Int, Int>?>(null) }
    var camera by remember { mutableStateOf<Camera?>(null) }
    var previewViewRef by remember { mutableStateOf<PreviewView?>(null) }
    var zoomRatioDisplay by remember { mutableFloatStateOf(1f) }
    var minZoomRatioDisplay by remember { mutableFloatStateOf(1f) }
    var maxZoomRatioDisplay by remember { mutableFloatStateOf(1f) }
    var focusMarkerVisible by remember { mutableStateOf(false) }
    var focusMarkerColor by remember { mutableStateOf(Color.White) }
    val focusMarkerRotation = remember { Animatable(0f) }
    val focusMarkerAlpha = remember { Animatable(1f) }
    val focusMarkerGapOffset = remember { Animatable(0f) }
    var focusMarkerAnimJob by remember { mutableStateOf<Job?>(null) }
    var showGridPreview by remember { mutableStateOf(false) }
    var hasObservedGridSizeChange by remember { mutableStateOf(false) }
    var halfPressActive by remember { mutableStateOf(false) }
    var selectedProbabilityDisplayCell by remember { mutableStateOf<Pair<Int, Int>?>(null) }
    var lastEditedProbabilityDisplayCell by remember { mutableStateOf<Pair<Int, Int>?>(null) }
    var dragAnchorProbabilityDisplayCell by remember { mutableStateOf<Pair<Int, Int>?>(null) }
    var liveProbabilityMatrix by remember(probabilityMatrix) { mutableStateOf(probabilityMatrix) }
    val forceGlobalSearchRequested = remember { AtomicBoolean(false) }

    // 跟踪上一个三色点的三个位置（用于ROI优化）
    var lastTriSpots by remember { mutableStateOf<Triple<Offset, Offset, Offset>?>(null) }
    var lastTriSpotsImageSize by remember { mutableStateOf<Pair<Int, Int>?>(null) }

    // ROI可视化信息（原图坐标）
    var roiVisualizationInfo by remember { mutableStateOf<RoiVisualizationInfo?>(null) }

    // 手动选择的ROI区域（屏幕坐标）- 用于优先识别
    var manualRoiStart by remember { mutableStateOf<Offset?>(null) }
    var manualRoiEnd by remember { mutableStateOf<Offset?>(null) }
    var manualRoiImageCoords by remember { mutableStateOf<ManualRoiCoords?>(null) } // left, top, right, bottom (图像坐标)
    var isDragging by remember { mutableStateOf(false) }

    // 限制识别区域（屏幕坐标）- 只在这个区域内识别
    var limitRegionStart by remember { mutableStateOf<Offset?>(null) }
    var limitRegionEnd by remember { mutableStateOf<Offset?>(null) }
    var limitRegionImageCoords by remember { mutableStateOf<ManualRoiCoords?>(null) } // left, top, right, bottom (图像坐标)
    var limitRegionScreenRect by remember { mutableStateOf<Rect?>(null) } // 限制区域的屏幕坐标Rect，用于判断触摸点是否在区域内
    var isDraggingLimitRegion by remember { mutableStateOf(false) }
    var isLongPressing by remember { mutableStateOf(false) } // 标记是否正在长按

    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val mainHandler = remember { Handler(Looper.getMainLooper()) }
    val coroutineScope = rememberCoroutineScope()

    // 帧率计算相关 - 使用Atomic类型确保线程安全
    val frameCount = remember { AtomicInteger(0) }
    val lastFrameTime = remember { AtomicLong(System.nanoTime()) }

    // 使用rememberUpdatedState确保闭包中始终使用最新的回调
    val currentOnFpsUpdate = rememberUpdatedState(onFpsUpdate)

    // 使用rememberUpdatedState确保闭包中始终使用最新的值
    val currentExposureCompensation = rememberUpdatedState(exposureCompensation)
    val currentGridSize = rememberUpdatedState(gridSize)
    val currentProbabilityMatrix = rememberUpdatedState(probabilityMatrix)
    val currentEnableRoiOptimization = rememberUpdatedState(enableRoiOptimization)
    val currentCaptureBitmapForCalibration = rememberUpdatedState(captureBitmapForCalibration)
    var skipFrames by remember { mutableIntStateOf(0) }

    // ROI优化关闭时清除跟踪状态和可视化信息
    LaunchedEffect(enableRoiOptimization) {
        if (!enableRoiOptimization) {
            lastTriSpots = null
            lastTriSpotsImageSize = null
            roiVisualizationInfo = null
        }
    }

    LaunchedEffect(probabilityEditEnabled) {
        if (!probabilityEditEnabled) {
            selectedProbabilityDisplayCell = null
            lastEditedProbabilityDisplayCell = null
            dragAnchorProbabilityDisplayCell = null
            onProbabilityCellSelected(null)
        }
    }

    LaunchedEffect(gridSize, isGridSizeAdjusting) {
        if (!hasObservedGridSizeChange) {
            hasObservedGridSizeChange = true
            return@LaunchedEffect
        }
        showGridPreview = true
        if (!isGridSizeAdjusting) {
            delay(500)
            showGridPreview = false
        }
    }

    // 当曝光补偿改变时更新相机设置
    LaunchedEffect(exposureCompensation) {
        camera?.let { cam ->
            val cameraControl = cam.cameraControl

            // 先锁定AE
            val camera2Control = Camera2CameraControl.from(cameraControl)
            val lockRequest = CaptureRequestOptions.Builder()
                .setCaptureRequestOption(CaptureRequest.CONTROL_AE_LOCK, true)
                .setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                .build()
            camera2Control.setCaptureRequestOptions(lockRequest)

            // 再设置曝光补偿
            cameraControl.setExposureCompensationIndex(exposureCompensation)
            android.util.Log.d("Camera", "AE locked with exposure: $exposureCompensation")
        }
    }

    LaunchedEffect(Unit) {
        VolumeKeyZoomController.zoomSteps.collectLatest { step ->
            val cam = camera ?: return@collectLatest
            val zoomState = cam.cameraInfo.zoomState.value ?: return@collectLatest
            val currentZoom = zoomState.zoomRatio
            val factor = 1.08f
            val targetZoom = if (step > 0) currentZoom * factor else currentZoom / factor
            val clampedZoom = targetZoom.coerceIn(zoomState.minZoomRatio, zoomState.maxZoomRatio)
            cam.cameraControl.setZoomRatio(clampedZoom)
            zoomRatioDisplay = clampedZoom
            minZoomRatioDisplay = zoomState.minZoomRatio
            maxZoomRatioDisplay = zoomState.maxZoomRatio
        }
    }
    val context = androidx.compose.ui.platform.LocalContext.current
    val haptic = LocalHapticFeedback.current
    LaunchedEffect(Unit) {
        ShutterKeyController.events.collectLatest { event ->
            when (event) {
                ShutterKeyEvent.HALF_PRESS_DOWN -> {
                    if (halfPressActive) return@collectLatest
                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                    halfPressActive = true
                    val cam = camera ?: return@collectLatest
                    val pv = previewViewRef ?: return@collectLatest
                    focusMarkerAnimJob?.cancel()
                    focusMarkerVisible = true
                    focusMarkerColor = Color.White
                    coroutineScope.launch {
                        focusMarkerRotation.snapTo(0f)
                        focusMarkerAlpha.snapTo(1f)
                        focusMarkerGapOffset.snapTo(0f)
                        // 出现瞬间做一次间隔脉动，模拟“对焦吸附”效果
                        focusMarkerGapOffset.animateTo(12f, animationSpec = tween(durationMillis = 135))
                        focusMarkerGapOffset.animateTo(-6f, animationSpec = tween(durationMillis = 135))
                        focusMarkerGapOffset.animateTo(0f, animationSpec = tween(durationMillis = 180))
                    }
                    // 对焦点取“实际预览显示区域”的中心（FIT_START 下不是整个 View 的中心）
                    val (focusX, focusY) = previewSize?.let { (imageW, imageH) ->
                        val viewW = pv.width.toFloat().coerceAtLeast(1f)
                        val viewH = pv.height.toFloat().coerceAtLeast(1f)
                        val scale = kotlin.math.min(viewW / imageW.toFloat(), viewH / imageH.toFloat())
                        Pair(imageW * scale / 2f, imageH * scale / 2f)
                    } ?: Pair(pv.width / 2f, pv.height / 2f)
                    val point = pv.meteringPointFactory.createPoint(focusX, focusY)
                    val action = FocusMeteringAction.Builder(point).build()
                    val focusFuture = cam.cameraControl.startFocusAndMetering(action)
                    focusFuture.addListener(
                        {
                            val success = runCatching { focusFuture.get().isFocusSuccessful }.getOrDefault(false)
                            focusMarkerAnimJob?.cancel()
                            focusMarkerAnimJob = coroutineScope.launch {
                                focusMarkerColor = if (success) Color(0xFF4CAF50) else Color(0xFFFF9800)
                                focusMarkerVisible = true
                                focusMarkerRotation.snapTo(0f)
                                focusMarkerAlpha.snapTo(1f)
                                focusMarkerGapOffset.snapTo(0f)

                                if (success) {
                                    launch {
                                        focusMarkerRotation.animateTo(
                                            targetValue = 180f,
                                            animationSpec = tween(durationMillis = 420)
                                        )
                                    }
                                    focusMarkerAlpha.animateTo(
                                        targetValue = 0f,
                                        animationSpec = tween(durationMillis = 420, delayMillis = 40)
                                    )
                                } else {
                                    focusMarkerAlpha.animateTo(
                                        targetValue = 0f,
                                        animationSpec = tween(durationMillis = 320, delayMillis = 480)
                                    )
                                }
                                focusMarkerVisible = false
                            }
                        },
                        ContextCompat.getMainExecutor(context)
                    )
                    Log.d("ShutterKey", "HALF_PRESS_DOWN captured")
                }

                ShutterKeyEvent.FULL_PRESS_DOWN -> {
                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                    forceGlobalSearchRequested.set(true)
                    roiVisualizationInfo = null
                    Log.d("ShutterKey", "FULL_PRESS_DOWN captured")
                }

                ShutterKeyEvent.HALF_PRESS_UP -> {
                    halfPressActive = false
                    Log.d("ShutterKey", "HALF_PRESS_UP captured")
                }

                ShutterKeyEvent.FULL_PRESS_UP -> {
                    Log.d("ShutterKey", "FULL_PRESS_UP captured")
                }
            }
        }
    }

    DisposableEffect(Unit) {
        onDispose { cameraExecutor.shutdown() }
    }

    // ---------- 分区树类型与函数（定义在 @Composable 作用域） ----------
    data class PlacedRect(
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float,
        val depth: Int,
        val verticalSplit: Boolean,
        val children: List<PlacedRect> = emptyList()
    ) {
        fun contains(x: Float, y: Float) = x in left..right && y >= top && y <= bottom
    }

    fun buildPartitionTree(
        l: Float, t: Float, r: Float, b: Float,
        depth: Int,
        maxDepthParam: Int,
        splitRatioParam: Float,
        minLeafSizePxParam: Int,
        targetPoint: Offset? = null  // 目标点位置，用于决定分割方向
    ): PlacedRect {
        val w = r - l
        val h = b - t
        if (depth >= maxDepthParam || w <= minLeafSizePxParam || h <= minLeafSizePxParam) {
            return PlacedRect(l, t, r, b, depth, verticalSplit = true, children = emptyList())
        }
        val vertical = w >= h
        val baseRatio = splitRatioParam.coerceIn(0.2f, 0.8f)

        // 根据目标点位置动态调整分割比例
        val ratio = if (targetPoint != null) {
            if (vertical) {
                // 垂直分割：如果点在左边，让左边短（使用1-ratio）；如果点在右边，让右边短（使用ratio）
                val pointX = targetPoint.x.coerceIn(l, r)
                val relativeX = (pointX - l) / w  // 点在矩形中的相对位置 [0, 1]
                if (relativeX < 0.5f) {
                    // 点在左边，让左边短（使用较小的比例，比如1-ratio）
                    1f - baseRatio
                } else {
                    // 点在右边，让右边短（使用较大的比例）
                    baseRatio
                }
            } else {
                // 水平分割：如果点在上边，让上边短（使用1-ratio）；如果点在下边，让下边短（使用ratio）
                val pointY = targetPoint.y.coerceIn(t, b)
                val relativeY = (pointY - t) / h  // 点在矩形中的相对位置 [0, 1]
                if (relativeY < 0.5f) {
                    // 点在上边，让上边短
                    1f - baseRatio
                } else {
                    // 点在下边，让下边短
                    baseRatio
                }
            }.coerceIn(0.2f, 0.8f)
        } else {
            baseRatio
        }

        return if (vertical) {
            val cut = l + w * ratio
            val leftRect = buildPartitionTree(l, t, cut, b, depth + 1, maxDepthParam, splitRatioParam, minLeafSizePxParam, targetPoint)
            val rightRect = buildPartitionTree(cut, t, r, b, depth + 1, maxDepthParam, splitRatioParam, minLeafSizePxParam, targetPoint)
            PlacedRect(l, t, r, b, depth, verticalSplit = true, children = listOf(leftRect, rightRect))
        } else {
            val cut = t + h * ratio
            val topRect = buildPartitionTree(l, t, r, cut, depth + 1, maxDepthParam, splitRatioParam, minLeafSizePxParam, targetPoint)
            val bottomRect = buildPartitionTree(l, cut, r, b, depth + 1, maxDepthParam, splitRatioParam, minLeafSizePxParam, targetPoint)
            PlacedRect(l, t, r, b, depth, verticalSplit = false, children = listOf(topRect, bottomRect))
        }
    }

    fun findPathToPoint(root: PlacedRect, x: Float, y: Float): List<PlacedRect> {
        val path = mutableListOf<PlacedRect>()
        var node: PlacedRect? = root
        while (node != null) {
            path.add(node)
            if (node.children.isEmpty()) break
            node = node.children.find { it.contains(x, y) }
            if (node == null) break
        }
        return path
    }

    /**
     * 查找ROI区域内与空间分割树相交的所有叶子节点（最小单位）
     * @param root 分割树根节点
     * @param roiLeft ROI左边界（原图坐标）
     * @param roiTop ROI上边界（原图坐标）
     * @param roiRight ROI右边界（原图坐标）
     * @param roiBottom ROI下边界（原图坐标）
     * @return ROI区域内的叶子节点列表
     */
    fun findLeafNodesInRoi(
        root: PlacedRect,
        roiLeft: Float,
        roiTop: Float,
        roiRight: Float,
        roiBottom: Float
    ): List<PlacedRect> {
        val result = mutableListOf<PlacedRect>()

        fun collectLeaves(node: PlacedRect) {
            // 检查节点是否与ROI区域相交
            if (node.right < roiLeft || node.left > roiRight ||
                node.bottom < roiTop || node.top > roiBottom) {
                return // 不相交，跳过
            }

            // 如果是叶子节点，添加到结果中
            if (node.children.isEmpty()) {
                result.add(node)
                return
            }

            // 如果有子节点，递归检查子节点
            node.children.forEach { child ->
                collectLeaves(child)
            }
        }

        collectLeaves(root)
        return result
    }

    // 计算目标点位置（所有检测到的点的中心），用于动态调整分割方向
    val targetPointForPartition = remember(brightSpots, previewSize) {
        if (brightSpots.isNotEmpty() && previewSize != null) {
            // 使用所有点的中心作为目标点
            val centerX = brightSpots.map { it.position.x }.average().toFloat()
            val centerY = brightSpots.map { it.position.y }.average().toFloat()
            Offset(centerX, centerY)
        } else {
            null
        }
    }

    // 在 previewSize 可用时，用 remember 缓存 rootRect（避免每帧重建）
    // 现在根据目标点位置动态构建分区树，让短的一边朝向点
    val rootRectCached = remember(previewSize, maxDepth, splitRatio, minLeafSizePx, targetPointForPartition) {
        if (previewSize == null) null
        else {
            val (w, h) = previewSize!!
            buildPartitionTree(0f, 0f, w.toFloat(), h.toFloat(), 0, maxDepth, splitRatio, minLeafSizePx, targetPointForPartition)
        }
    }
    var latestRoiInfoForAnimation by remember { mutableStateOf<RoiVisualizationInfo?>(null) }
    LaunchedEffect(roiVisualizationInfo) {
        if (roiVisualizationInfo != null) {
            latestRoiInfoForAnimation = roiVisualizationInfo
        }
    }
    val roiTargetRectInImage = remember(
        roiVisualizationInfo,
        rootRectCached,
        previewSize,
        brightSpots,
        maxDepth,
        splitRatio,
        minLeafSizePx,
        targetPointForPartition
    ) {
        val roiInfo = roiVisualizationInfo ?: return@remember null
        val imageSize = previewSize ?: return@remember null
        val (imageWidth, imageHeight) = imageSize
        val currentTargetPoint = if (brightSpots.isNotEmpty()) {
            val centerX = brightSpots.map { it.position.x }.average().toFloat()
            val centerY = brightSpots.map { it.position.y }.average().toFloat()
            Offset(centerX, centerY)
        } else {
            targetPointForPartition
        }
        val rootRect = rootRectCached ?: buildPartitionTree(
            0f, 0f, imageWidth.toFloat(), imageHeight.toFloat(), 0,
            maxDepth, splitRatio, minLeafSizePx, currentTargetPoint
        )
        val leafNodes = findLeafNodesInRoi(
            root = rootRect,
            roiLeft = roiInfo.left,
            roiTop = roiInfo.top,
            roiRight = roiInfo.right,
            roiBottom = roiInfo.bottom
        )
        if (leafNodes.isEmpty()) {
            null
        } else {
            Rect(
                left = leafNodes.minOf { it.left },
                top = leafNodes.minOf { it.top },
                right = leafNodes.maxOf { it.right },
                bottom = leafNodes.maxOf { it.bottom }
            )
        }
    }
    val roiAnimatedWidth by animateFloatAsState(
        targetValue = (roiTargetRectInImage?.right ?: 0f) - (roiTargetRectInImage?.left ?: 0f),
        animationSpec = tween(durationMillis = 120),
        label = "roiAnimatedWidth"
    )
    val roiAnimatedHeight by animateFloatAsState(
        targetValue = (roiTargetRectInImage?.bottom ?: 0f) - (roiTargetRectInImage?.top ?: 0f),
        animationSpec = tween(durationMillis =120),
        label = "roiAnimatedHeight"
    )
    val animatedRoiCenterX by animateFloatAsState(
        targetValue = ((roiTargetRectInImage?.left ?: 0f) + (roiTargetRectInImage?.right ?: 0f)) / 2f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioNoBouncy,
            stiffness = Spring.StiffnessHigh
        ),
        label = "animatedRoiCenterX"
    )
    val animatedRoiCenterY by animateFloatAsState(
        targetValue = ((roiTargetRectInImage?.top ?: 0f) + (roiTargetRectInImage?.bottom ?: 0f)) / 2f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioNoBouncy,
            stiffness = Spring.StiffnessHigh
        ),
        label = "animatedRoiCenterY"
    )
    val animatedRoiLeft = animatedRoiCenterX - roiAnimatedWidth / 2f
    val animatedRoiTop = animatedRoiCenterY - roiAnimatedHeight / 2f
    val animatedRoiRight = animatedRoiCenterX + roiAnimatedWidth / 2f
    val animatedRoiBottom = animatedRoiCenterY + roiAnimatedHeight / 2f
    val roiVisibilityAlpha by animateFloatAsState(
        targetValue = if (roiTargetRectInImage != null) 1f else 0f,
        animationSpec = tween(durationMillis = 180),
        label = "roiVisibilityAlpha"
    )
    val roiPulseTransition = rememberInfiniteTransition(label = "roiPulse")
    val roiPulseAlpha by roiPulseTransition.animateFloat(
        initialValue = 0.85f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 900),
            repeatMode = RepeatMode.Reverse
        ),
        label = "roiPulseAlpha"
    )

    val previewInteractionModifier = if (probabilityEditEnabled) {
        modifier
    } else {
        modifier
        .pointerInput(Unit) {
            // 长按+拖动：创建限制识别区域
            detectDragGesturesAfterLongPress(
                onDragStart = { offset ->
                    mainHandler.post {
                        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                        isLongPressing = true
                        isDraggingLimitRegion = true
                        limitRegionStart = offset
                        limitRegionEnd = offset
                        limitRegionImageCoords = null
                    }
                },
                onDrag = { change, dragAmount ->
                    change.consume()
                    mainHandler.post {
                        limitRegionEnd = change.position
                    }
                },
                onDragEnd = {
                    mainHandler.post {
                        isDraggingLimitRegion = false
                        isLongPressing = false
                    }
                }
            )
        }
        .pointerInput(limitRegionScreenRect) {
            // 普通拖动：创建优先识别区域（原有的manualRoi功能）
            detectDragGestures(
                onDragStart = { offset ->
                    // 只有在没有长按拖动时才创建优先识别区域
                    // 如果有限制区域，检查拖动起始点是否在限制区域内
                    val allowDrag = if (limitRegionScreenRect != null) {
                        limitRegionScreenRect!!.contains(offset)
                    } else {
                        true // 没有限制区域，允许拖动
                    }

                    mainHandler.post {
                        if (!isDraggingLimitRegion && !isLongPressing && allowDrag) {
                            isDragging = true
                            manualRoiStart = offset
                            manualRoiEnd = offset
                            manualRoiImageCoords = null
                        }
                    }
                },
                onDrag = { change, dragAmount ->
                    // 拖拽过程中更新结束点
                    if (!isDraggingLimitRegion && !isLongPressing) {
                        change.consume()
                        mainHandler.post {
                            manualRoiEnd = change.position
                        }
                    }
                },
                onDragEnd = {
                    // 拖拽结束时，标记拖拽结束（在Canvas中计算图像坐标）
                    mainHandler.post {
                        isDragging = false
                    }
                }
            )
        }
        .pointerInput(limitRegionImageCoords) {
            // 单击：解除限制区域（仅在有限制区域时生效）
            detectTapGestures(
                onTap = { offset ->
                    // 如果有限制区域，单击任意位置解除限制
                    mainHandler.post {
                        if (limitRegionImageCoords != null) {
                            limitRegionStart = null
                            limitRegionEnd = null
                            limitRegionImageCoords = null
                            limitRegionScreenRect = null // 清除屏幕坐标Rect
                        }
                    }
                }
            )
        }
    }

    Box(modifier = previewInteractionModifier) {
        // 相机预览视图
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                val previewView = PreviewView(ctx)
                // 使用不裁剪的缩放策略，便于一致的矩阵映射
                previewView.scaleType = PreviewView.ScaleType.FIT_START
                previewViewRef = previewView

                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    // 预览用例
                    val preview = Preview.Builder()
                        .build()
                        .also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }

                    // 图像分析用例 - 用于检测亮点
                    val imageAnalyzer = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                        .also { analysis ->
                            analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                                if (skipFrames > 0) {
                                    skipFrames -= 1
                                    imageProxy.close()
                                } else {
                                    // 计算帧率（线程安全）
                                    val currentTime = System.nanoTime()
                                    val count = frameCount.incrementAndGet()
                                    val elapsedNs = currentTime - lastFrameTime.get()
                                    // 每秒更新一次帧率
                                    if (elapsedNs >= 1_000_000_000L) { // 1秒
                                        val fps = count * 1_000_000_000f / elapsedNs
                                        frameCount.set(0)
                                        lastFrameTime.set(currentTime)
                                        // 在主线程中通知UI更新帧率
                                        mainHandler.post {
                                            currentOnFpsUpdate.value(fps)
                                        }
                                    }

                                    // 获取当前手动ROI坐标和限制区域坐标（通过原子引用或直接使用，因为这是主线程上下文）
                                    val currentManualRoi = manualRoiImageCoords
                                    val currentLimitRegion = limitRegionImageCoords


                                    processImage(
                                        imageProxy = imageProxy,
                                        detector = detector,
                                        gridSize = currentGridSize.value,
                                        probabilityMatrix = currentProbabilityMatrix.value,
                                        enableRoiOptimization = currentEnableRoiOptimization.value,
                                        lastTriSpots = if (currentEnableRoiOptimization.value) lastTriSpots else null,
                                        lastTriSpotsImageSize = if (currentEnableRoiOptimization.value) lastTriSpotsImageSize else null,
                                        manualRoi = currentManualRoi,
                                        limitRegion = currentLimitRegion,
                                        forceGlobalSearch = forceGlobalSearchRequested.getAndSet(false),
                                        onRoiInfo = { info ->
                                            // 在主线程中更新ROI可视化信息
                                            mainHandler.post {
                                                roiVisualizationInfo = info
                                            }
                                        },
                                        onSpotsDetected = { spots, size, foundInManualRoi ->
                                            brightSpots = spots
                                            previewSize = size

                                            // 如果在手动ROI中找到三色点，清除手动ROI并隐藏显示框
                                            if (foundInManualRoi && spots.size == 3) {
                                                mainHandler.post {
                                                    manualRoiStart = null
                                                    manualRoiEnd = null
                                                    manualRoiImageCoords = null
                                                }
                                            }

                                            // 仅在启用ROI优化时更新跟踪的三色点位置
                                            if (currentEnableRoiOptimization.value) {
                                                if (spots.size == 3) {
                                                    var redSpot: BrightSpot? = null
                                                    var greenSpot: BrightSpot? = null
                                                    var blueSpot: BrightSpot? = null
                                                    spots.forEach { spot ->
                                                        when (spot.color) {
                                                            LedColor.RED -> redSpot = spot
                                                            LedColor.GREEN -> greenSpot = spot
                                                            LedColor.BLUE -> blueSpot = spot
                                                            else -> {}
                                                        }
                                                    }
                                                    if (redSpot != null && greenSpot != null && blueSpot != null) {
                                                        // 存储三个点的位置
                                                        lastTriSpots = Triple(
                                                            redSpot.position,
                                                            greenSpot.position,
                                                            blueSpot.position
                                                        )
                                                        lastTriSpotsImageSize = size
                                                    } else {
                                                        // 如果找不到三个完整的颜色点，清除跟踪
                                                        lastTriSpots = null
                                                        lastTriSpotsImageSize = null
                                                    }
                                                } else {
                                                    // 如果不是三色模式或未找到三个点，清除跟踪
                                                    lastTriSpots = null
                                                    lastTriSpotsImageSize = null
                                                }
                                            }
                                            onBrightSpotsDetected(spots, size)
                                        },
                                        captureBitmapForCalibration = currentCaptureBitmapForCalibration.value,
                                        onBitmapCaptured = onBitmapCaptured
                                    )
                                    imageProxy.close()
                                }
                            }
                        }

                    // 选择后置摄像头
                    val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                    try {
                        // 解绑所有用例
                        cameraProvider.unbindAll()

                        // 绑定用例到相机并获取Camera对象
                        val cam = cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            cameraSelector,
                            preview,
                            imageAnalyzer
                        )

                        camera = cam
                        zoomRatioDisplay = cam.cameraInfo.zoomState.value?.zoomRatio ?: 1f
                        minZoomRatioDisplay = cam.cameraInfo.zoomState.value?.minZoomRatio ?: 1f
                        maxZoomRatioDisplay = cam.cameraInfo.zoomState.value?.maxZoomRatio ?: 1f
                        val camera2Info = Camera2CameraInfo.from(cam.cameraInfo)
                        val ranges = camera2Info.getCameraCharacteristic(
                            CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES
                        )

                        android.util.Log.d("FPS", "AE_AVAILABLE_FPS_RANGES=${ranges?.joinToString()}")
                        // 设置固定曝光
                        val cameraControl = cam.cameraControl
                        val cameraInfo = cam.cameraInfo

                        // 获取曝光补偿范围
                        val exposureState = cameraInfo.exposureState
                        val minExposure = exposureState.exposureCompensationRange.lower
                        val maxExposure = exposureState.exposureCompensationRange.upper

                        // 设置初始曝光补偿
                        cameraControl.setExposureCompensationIndex(currentExposureCompensation.value)

                        // 启用AE锁定（如果支持）
                        try {
                            cameraControl.enableTorch(false) // 关闭闪光灯
                        } catch (e: Exception) {
                            // 某些设备不支持闪光灯控制
                        }

                    } catch (e: Exception) {
                        e.printStackTrace()
                    }
                }, ContextCompat.getMainExecutor(ctx))

                previewView
            }
        )

        // 在预览视图上绘制亮点标记与分区指示线
        Canvas(modifier = Modifier.fillMaxSize()) {
            val pv = previewViewRef
            if (previewSize != null && pv != null) {
                val (imageWidth, imageHeight) = previewSize!!
                // 构建从图像坐标到画布坐标的变换矩阵（匹配 FIT_START 策略）
                val scale = kotlin.math.min(size.width / imageWidth, size.height / imageHeight)
                val dx = 0f
                val dy = 0f
                val transform = Matrix().apply {
                    postScale(scale, scale)
                    postTranslate(dx, dy)
                }

                // 概率矩阵可视化：编辑模式下显示更明确的整格网格，普通模式维持轻量覆盖。
                probabilityMatrix?.let { matrix ->
                    val rows = matrix.rowsForImage(imageWidth, imageHeight)
                    val cols = matrix.colsForImage(imageWidth, imageHeight)
                    val cellWidth = imageWidth.toFloat() / cols.toFloat()
                    val cellHeight = imageHeight.toFloat() / rows.toFloat()

                    val rowStep = 1
                    val colStep = 1
                    val lowWeight = ProbabilityMatrix32x24.MIN_WEIGHT
                    val highWeight = ProbabilityMatrix32x24.MAX_WEIGHT
                    val weightRange = (highWeight - lowWeight).coerceAtLeast(0.001f)
                    val lowColor = if (probabilityEditEnabled) Color(0xFFF8FBFF) else Color(0xFFF2F2F2)
                    val highColor = if (probabilityEditEnabled) Color(0xFF147EFB) else Color(0xFFF2F2F2)

                    for (row in 0 until rows step rowStep) {
                        for (col in 0 until cols step colStep) {
                            val weight = matrix.weightAtGridCell(row, col, imageWidth, imageHeight)
                            val normalized = ((weight - lowWeight) / weightRange).coerceIn(0f, 1f)
                            val alpha = if (probabilityEditEnabled) {
                                (0.22f + normalized * 0.48f).coerceIn(0.12f, 0.70f)
                            } else {
                                (0.50f * (1f - normalized)).coerceIn(0.02f, 0.50f)
                            }

                            val left = dx + col * cellWidth * scale
                            val top = dy + row * cellHeight * scale
                            val drawW = cellWidth * scale * colStep
                            val drawH = cellHeight * scale * rowStep

                            drawRect(
                                color = if (probabilityEditEnabled) {
                                    androidx.compose.ui.graphics.lerp(lowColor, highColor, normalized).copy(alpha = alpha)
                                } else {
                                    lowColor.copy(alpha = alpha)
                                },
                                topLeft = Offset(left, top),
                                size = androidx.compose.ui.geometry.Size(drawW, drawH)
                            )
                        }
                    }

                    if (probabilityEditEnabled) {
                        for (row in 0..rows) {
                            val y = dy + row * cellHeight * scale
                            drawLine(
                                color = Color.White.copy(alpha = if (row % 8 == 0) 0.80f else 0.35f),
                                start = Offset(dx, y),
                                end = Offset(dx + imageWidth * scale, y),
                                strokeWidth = if (row % 8 == 0) 1.4f else 0.5f
                            )
                        }
                        for (col in 0..cols) {
                            val x = dx + col * cellWidth * scale
                            drawLine(
                                color = Color.White.copy(alpha = if (col % 8 == 0) 0.80f else 0.35f),
                                start = Offset(x, dy),
                                end = Offset(x, dy + imageHeight * scale),
                                strokeWidth = if (col % 8 == 0) 1.4f else 0.5f
                            )
                        }

                        selectedProbabilityDisplayCell?.let { (row, col) ->
                            drawRect(
                                color = Color(0xFFFFF176),
                                topLeft = Offset(
                                    dx + col * cellWidth * scale,
                                    dy + row * cellHeight * scale
                                ),
                                size = androidx.compose.ui.geometry.Size(
                                    cellWidth * scale,
                                    cellHeight * scale
                                ),
                                style = Stroke(width = 2f)
                            )
                        }
                    }
                }

                // 使用缓存的分区树（已经基于 previewSize 构建或为 null）
                // 如果缓存中没有，使用当前帧的brightSpots计算目标点
                val currentTargetPoint = if (brightSpots.isNotEmpty()) {
                    val centerX = brightSpots.map { it.position.x }.average().toFloat()
                    val centerY = brightSpots.map { it.position.y }.average().toFloat()
                    Offset(centerX, centerY)
                } else {
                    targetPointForPartition
                }
                val rootRect = rootRectCached ?: buildPartitionTree(
                    0f, 0f, imageWidth.toFloat(), imageHeight.toFloat(), 0,
                    maxDepth, splitRatio, minLeafSizePx, currentTargetPoint
                )

                // ---------- 优化：一次性查找三色点，避免重复查找 ----------
                var triCenterImg: Offset? = null
                var redSpot: BrightSpot? = null
                var greenSpot: BrightSpot? = null
                var blueSpot: BrightSpot? = null
                if (brightSpots.size == 3) {
                    brightSpots.forEach { spot ->
                        when (spot.color) {
                            LedColor.RED -> redSpot = spot
                            LedColor.GREEN -> greenSpot = spot
                            LedColor.BLUE -> blueSpot = spot
                            else -> {}
                        }
                    }
                    if (redSpot != null && greenSpot != null && blueSpot != null) {
                        val cx = (redSpot.position.x + greenSpot.position.x + blueSpot.position.x) / 3f
                        val cy = (redSpot.position.y + greenSpot.position.y + blueSpot.position.y) / 3f
                        triCenterImg = Offset(cx, cy)
                    }
                }

                val pBuf = FloatArray(8) // 复用FloatArray，避免频繁分配
//                val baseColor = Color(0xFF222222) // 预定义基础颜色
//                partitionRects.forEach { info ->
//                    pBuf[0] = info.node.left; pBuf[1] = info.node.top
//                    pBuf[2] = info.node.right; pBuf[3] = info.node.top
//                    pBuf[4] = info.node.right; pBuf[5] = info.node.bottom
//                    pBuf[6] = info.node.left; pBuf[7] = info.node.bottom
//                    transform.mapPoints(pBuf)
//
//                    // 使用drawRect的stroke替代4条drawLine（性能更好）
//                    drawRect(
//                        color = baseColor.copy(alpha = info.alpha),
//                        topLeft = Offset(pBuf[0], pBuf[1]),
//                        size = androidx.compose.ui.geometry.Size(pBuf[2] - pBuf[0], pBuf[5] - pBuf[1]),
//                        style = Stroke(width = info.strokeW)
//                    )
//                } //灰色基准线，我觉得已经有了空间分割树可视化，就不需要这个了

                // ---------- 优化：如果存在三色中心，则突出显示包含中心的路径 ----------
                if (triCenterImg != null) {
                    val path = findPathToPoint(rootRect, triCenterImg.x, triCenterImg.y)
                    val filteredPath = path.filter { it.depth <= maxDrawDepth }
                    if (filteredPath.isNotEmpty()) {
                        val lastIdx = filteredPath.lastIndex
                        filteredPath.forEachIndexed { idx, node ->
                            pBuf[0] = node.left; pBuf[1] = node.top
                            pBuf[2] = node.right; pBuf[3] = node.top
                            pBuf[4] = node.right; pBuf[5] = node.bottom
                            pBuf[6] = node.left; pBuf[7] = node.bottom
                            transform.mapPoints(pBuf)

//                            val depth = node.depth
//                            val ratio = 1f - (depth.toFloat() / (maxDepth.toFloat() + 1f))
//                            val strokeW = (6f * ratio).coerceIn(1f, 8f)
                            val strokeW = 3f //此处使用等宽线段，如果需要随着深度增加而改变粗细，可启用上述代码
                            val strokeCol = if (idx == lastIdx) {
                                Color(0xFFFFFF00) // 复用颜色常量
                            } else {
                                //Color(0xFFFFCC33).copy(alpha = (0.95f - idx * 0.06f).coerceIn(0.2f, 0.98f))
                                //Color(0xFFFFCC33)
                                Color(0xFFAAF0FF).copy(alpha = (idx * 0.06f).coerceIn(0.6f, 0.98f))
                            }

                            drawRect(
                                color = strokeCol,
                                topLeft = Offset(pBuf[0], pBuf[1]),
                                size = androidx.compose.ui.geometry.Size(pBuf[2] - pBuf[0], pBuf[5] - pBuf[1]),
                                style = Stroke(width = strokeW)
                            )

                            // 若为叶子则填充
                            if (node.children.isEmpty()) {
                                drawRect(
                                    color = Color(0xC99AF2FF),
                                    topLeft = Offset(pBuf[0], pBuf[1]),
                                    size = androidx.compose.ui.geometry.Size(pBuf[2] - pBuf[0], pBuf[5] - pBuf[1])
                                )
                            }
                        }
                    }
                }

                // ---------- 优化：绘制三色点到中心的连线与中心点 ----------
                if (brightSpots.size == 3 && triCenterImg != null && redSpot != null && greenSpot != null && blueSpot != null) {
                    val pointBuf = FloatArray(2) // 用于单点变换
                    pointBuf[0] = redSpot.position.x; pointBuf[1] = redSpot.position.y; transform.mapPoints(pointBuf); val redP = Offset(pointBuf[0], pointBuf[1])
                    pointBuf[0] = greenSpot.position.x; pointBuf[1] = greenSpot.position.y; transform.mapPoints(pointBuf); val greenP = Offset(pointBuf[0], pointBuf[1])
                    pointBuf[0] = blueSpot.position.x; pointBuf[1] = blueSpot.position.y; transform.mapPoints(pointBuf); val blueP = Offset(pointBuf[0], pointBuf[1])
                    pointBuf[0] = triCenterImg.x; pointBuf[1] = triCenterImg.y; transform.mapPoints(pointBuf); val centerP = Offset(pointBuf[0], pointBuf[1])

                    // 复用颜色常量
                    drawLine(color = Color(0xCCFF3333), start = redP, end = centerP, strokeWidth = 4f)
                    drawLine(color = Color(0xCC33FF33), start = greenP, end = centerP, strokeWidth = 4f)
                    drawLine(color = Color(0xCC3366FF), start = blueP, end = centerP, strokeWidth = 4f)

                    drawCircle(color = Color(0xFFFFFF00), radius = 10f, center = centerP)
                    drawCircle(color = Color.White, radius = 6f, center = centerP, style = Stroke(width = 2f))
                    drawCircle(color = Color(0xFFFFFF00), radius = 3f, center = centerP)
                }

                // ---------- 优化：绘制检测到的 brightSpots 标记（circle + cross） ----------
                // 预定义颜色常量，避免重复创建
                val redColor = Color(0xFFFF3333)
                val greenColor = Color(0xFF33FF33)
                val blueColor = Color(0xFF3366FF)
                val whiteColor = Color.White

                val spotBuf = FloatArray(2) // 复用FloatArray
                brightSpots.forEach { spot ->
                    spotBuf[0] = spot.position.x
                    spotBuf[1] = spot.position.y
                    transform.mapPoints(spotBuf)
                    val x = spotBuf[0]
                    val y = spotBuf[1]
                    val markColor = when (spot.color) {
                        LedColor.RED -> redColor
                        LedColor.GREEN -> greenColor
                        LedColor.BLUE -> blueColor
                        else -> whiteColor
                    }
                    drawCircle(color = markColor, radius = 14f, center = Offset(x, y), style = Stroke(width = 4f))
                    drawCircle(color = whiteColor, radius = 10f, center = Offset(x, y), style = Stroke(width = 1.6f))
                    drawLine(color = markColor, start = Offset(x - 12, y), end = Offset(x + 12, y), strokeWidth = 2.6f)
                    drawLine(color = markColor, start = Offset(x, y - 12), end = Offset(x, y + 12), strokeWidth = 2.6f)
                    drawCircle(color = markColor, radius = 3f, center = Offset(x, y))
                }

                // ---------- 限制识别区域：处理拖拽结束后的坐标计算 ----------
                if (!isDraggingLimitRegion && limitRegionStart != null && limitRegionEnd != null && limitRegionImageCoords == null) {
                    // 拖拽刚结束，计算图像坐标
                    val start = limitRegionStart!!
                    val end = limitRegionEnd!!

                    // 计算矩形边界（确保顺序正确）
                    val minX = kotlin.math.min(start.x, end.x)
                    val maxX = kotlin.math.max(start.x, end.x)
                    val minY = kotlin.math.min(start.y, end.y)
                    val maxY = kotlin.math.max(start.y, end.y)

                    // 构建逆变换：从画布坐标到图像坐标
                    val invScale = 1f / scale
                    val imageLeft = ((minX - dx) * invScale).coerceIn(0f, imageWidth.toFloat())
                    val imageTop = ((minY - dy) * invScale).coerceIn(0f, imageHeight.toFloat())
                    val imageRight = ((maxX - dx) * invScale).coerceIn(0f, imageWidth.toFloat())
                    val imageBottom = ((maxY - dy) * invScale).coerceIn(0f, imageHeight.toFloat())

                    // 确保矩形有效（至少有一定的尺寸）
                    if (imageRight > imageLeft && imageBottom > imageTop &&
                        (imageRight - imageLeft) * (imageBottom - imageTop) > 100f) {
                        // 在主线程中更新图像坐标（需要访问画布size，所以在这里计算）
                        mainHandler.post {
                            limitRegionImageCoords = ManualRoiCoords(
                                imageLeft,
                                imageTop,
                                imageRight,
                                imageBottom
                            )
                        }
                    } else {
                        // 矩形太小，清除
                        mainHandler.post {
                            limitRegionStart = null
                            limitRegionEnd = null
                            limitRegionImageCoords = null
                        }
                    }
                }

                // ---------- 绘制限制识别区域的半透明遮罩（区域外遮罩，区域内正常显示） ----------
                if (limitRegionImageCoords != null || (limitRegionStart != null && limitRegionEnd != null)) {
                    val regionRect = if (limitRegionImageCoords != null) {
                        // 使用图像坐标转换为屏幕坐标
                        val topLeftBuf = FloatArray(2)
                        topLeftBuf[0] = limitRegionImageCoords!!.left
                        topLeftBuf[1] = limitRegionImageCoords!!.top
                        transform.mapPoints(topLeftBuf)
                        val topLeft = Offset(topLeftBuf[0], topLeftBuf[1])

                        val bottomRightBuf = FloatArray(2)
                        bottomRightBuf[0] = limitRegionImageCoords!!.right
                        bottomRightBuf[1] = limitRegionImageCoords!!.bottom
                        transform.mapPoints(bottomRightBuf)
                        val bottomRight = Offset(bottomRightBuf[0], bottomRightBuf[1])

                        Rect(topLeft, bottomRight)
                    } else {
                        // 使用屏幕坐标（拖动中）
                        val start = limitRegionStart!!
                        val end = limitRegionEnd!!
                        val minX = kotlin.math.min(start.x, end.x)
                        val maxX = kotlin.math.max(start.x, end.x)
                        val minY = kotlin.math.min(start.y, end.y)
                        val maxY = kotlin.math.max(start.y, end.y)
                        Rect(minX, minY, maxX, maxY)
                    }

                    // 更新限制区域的屏幕坐标Rect（用于判断触摸点是否在区域内）
                    mainHandler.post {
                        limitRegionScreenRect = regionRect
                    }

                    // 绘制半透明遮罩（区域外遮罩）
                    // 上半部分
                    if (regionRect.top > 0) {
                        drawRect(
                            color = Color(0x80000000),
                            topLeft = Offset(0f, 0f),
                            size = androidx.compose.ui.geometry.Size(size.width, regionRect.top)
                        )
                    }
                    // 下半部分
                    if (regionRect.bottom < size.height) {
                        drawRect(
                            color = Color(0x80000000),
                            topLeft = Offset(0f, regionRect.bottom),
                            size = androidx.compose.ui.geometry.Size(size.width, size.height - regionRect.bottom)
                        )
                    }
                    // 左侧
                    if (regionRect.left > 0) {
                        drawRect(
                            color = Color(0x80000000),
                            topLeft = Offset(0f, regionRect.top),
                            size = androidx.compose.ui.geometry.Size(regionRect.left, regionRect.height)
                        )
                    }
                    // 右侧
                    if (regionRect.right < size.width) {
                        drawRect(
                            color = Color(0x80000000),
                            topLeft = Offset(regionRect.right, regionRect.top),
                            size = androidx.compose.ui.geometry.Size(size.width - regionRect.right, regionRect.height)
                        )
                    }

                    // 绘制限制区域的边框
                    val strokeColor = Color(0xFFFFAA00) // 橙色边框以区分
                    val strokeWidth = 5f
                    val cornerLength = (kotlin.math.min(regionRect.width, regionRect.height) * 0.15f).coerceIn(15f, 40f)

                    // 左上角 ┌
                    drawLine(
                        color = strokeColor,
                        start = Offset(regionRect.left, regionRect.top),
                        end = Offset(regionRect.left + cornerLength, regionRect.top),
                        strokeWidth = strokeWidth
                    )
                    drawLine(
                        color = strokeColor,
                        start = Offset(regionRect.left, regionRect.top),
                        end = Offset(regionRect.left, regionRect.top + cornerLength),
                        strokeWidth = strokeWidth
                    )

                    // 右上角 ┐
                    drawLine(
                        color = strokeColor,
                        start = Offset(regionRect.right, regionRect.top),
                        end = Offset(regionRect.right - cornerLength, regionRect.top),
                        strokeWidth = strokeWidth
                    )
                    drawLine(
                        color = strokeColor,
                        start = Offset(regionRect.right, regionRect.top),
                        end = Offset(regionRect.right, regionRect.top + cornerLength),
                        strokeWidth = strokeWidth
                    )

                    // 左下角 └
                    drawLine(
                        color = strokeColor,
                        start = Offset(regionRect.left, regionRect.bottom),
                        end = Offset(regionRect.left + cornerLength, regionRect.bottom),
                        strokeWidth = strokeWidth
                    )
                    drawLine(
                        color = strokeColor,
                        start = Offset(regionRect.left, regionRect.bottom),
                        end = Offset(regionRect.left, regionRect.bottom - cornerLength),
                        strokeWidth = strokeWidth
                    )

                    // 右下角 ┘
                    drawLine(
                        color = strokeColor,
                        start = Offset(regionRect.right, regionRect.bottom),
                        end = Offset(regionRect.right - cornerLength, regionRect.bottom),
                        strokeWidth = strokeWidth
                    )
                    drawLine(
                        color = strokeColor,
                        start = Offset(regionRect.right, regionRect.bottom),
                        end = Offset(regionRect.right, regionRect.bottom - cornerLength),
                        strokeWidth = strokeWidth
                    )
                }

                // ---------- 手动选择的ROI区域：处理拖拽结束后的坐标计算和绘制 ----------
                if (!isDragging && manualRoiStart != null && manualRoiEnd != null && manualRoiImageCoords == null) {
                    // 拖拽刚结束，计算图像坐标
                    val start = manualRoiStart!!
                    val end = manualRoiEnd!!

                    // 计算矩形边界（确保顺序正确）
                    val minX = kotlin.math.min(start.x, end.x)
                    val maxX = kotlin.math.max(start.x, end.x)
                    val minY = kotlin.math.min(start.y, end.y)
                    val maxY = kotlin.math.max(start.y, end.y)

                    // 构建逆变换：从画布坐标到图像坐标
                    val invScale = 1f / scale
                    val imageLeft = ((minX - dx) * invScale).coerceIn(0f, imageWidth.toFloat())
                    val imageTop = ((minY - dy) * invScale).coerceIn(0f, imageHeight.toFloat())
                    val imageRight = ((maxX - dx) * invScale).coerceIn(0f, imageWidth.toFloat())
                    val imageBottom = ((maxY - dy) * invScale).coerceIn(0f, imageHeight.toFloat())

                    // 确保矩形有效（至少有一定的尺寸）
                    if (imageRight > imageLeft && imageBottom > imageTop &&
                        (imageRight - imageLeft) * (imageBottom - imageTop) > 100f) {
                        // 在主线程中更新图像坐标（需要访问画布size，所以在这里计算）
                        mainHandler.post {
                            manualRoiImageCoords = ManualRoiCoords(
                                imageLeft,
                                imageTop,
                                imageRight,
                                imageBottom
                            )
                        }
                    } else {
                        // 矩形太小，清除
                        mainHandler.post {
                            manualRoiStart = null
                            manualRoiEnd = null
                            manualRoiImageCoords = null
                        }
                    }
                }

                // 绘制手动选择的矩形框（屏幕坐标）
                if (manualRoiStart != null && manualRoiEnd != null) {
                    val start = manualRoiStart!!
                    val end = manualRoiEnd!!

                    // 计算矩形边界
                    val minX = kotlin.math.min(start.x, end.x)
                    val maxX = kotlin.math.max(start.x, end.x)
                    val minY = kotlin.math.min(start.y, end.y)
                    val maxY = kotlin.math.max(start.y, end.y)

                    val rectWidth = maxX - minX
                    val rectHeight = maxY - minY

                    // 绘制半透明背景
                    drawRect(
                        color = Color(0x3300AAFF),
                        topLeft = Offset(minX, minY),
                        size = androidx.compose.ui.geometry.Size(rectWidth, rectHeight)
                    )

                    // 绘制边框
                    val strokeColor = Color(0xFF00AAFF)
                    val strokeWidth = 4f

                    // 计算角标记长度
                    val cornerLength = (kotlin.math.min(rectWidth, rectHeight) * 0.15f).coerceIn(15f, 40f)

                    // 绘制四个角的L形标记
                    // 左上角 ┌
                    drawLine(
                        color = strokeColor,
                        start = Offset(minX, minY),
                        end = Offset(minX + cornerLength, minY),
                        strokeWidth = strokeWidth
                    )
                    drawLine(
                        color = strokeColor,
                        start = Offset(minX, minY),
                        end = Offset(minX, minY + cornerLength),
                        strokeWidth = strokeWidth
                    )

                    // 右上角 ┐
                    drawLine(
                        color = strokeColor,
                        start = Offset(maxX, minY),
                        end = Offset(maxX - cornerLength, minY),
                        strokeWidth = strokeWidth
                    )
                    drawLine(
                        color = strokeColor,
                        start = Offset(maxX, minY),
                        end = Offset(maxX, minY + cornerLength),
                        strokeWidth = strokeWidth
                    )

                    // 左下角 └
                    drawLine(
                        color = strokeColor,
                        start = Offset(minX, maxY),
                        end = Offset(minX + cornerLength, maxY),
                        strokeWidth = strokeWidth
                    )
                    drawLine(
                        color = strokeColor,
                        start = Offset(minX, maxY),
                        end = Offset(minX, maxY - cornerLength),
                        strokeWidth = strokeWidth
                    )

                    // 右下角 ┘
                    drawLine(
                        color = strokeColor,
                        start = Offset(maxX, maxY),
                        end = Offset(maxX - cornerLength, maxY),
                        strokeWidth = strokeWidth
                    )
                    drawLine(
                        color = strokeColor,
                        start = Offset(maxX, maxY),
                        end = Offset(maxX, maxY - cornerLength),
                        strokeWidth = strokeWidth
                    )
                }

                // ---------- ROI可视化：绘制带尺寸过渡动画的ROI区域矩形框 ----------
                val roiInfoForDraw = latestRoiInfoForAnimation
                if (roiInfoForDraw != null && roiVisibilityAlpha > 0.01f) {
                    val roiMinLeft = animatedRoiLeft
                    val roiMinTop = animatedRoiTop
                    val roiMaxRight = animatedRoiRight
                    val roiMaxBottom = animatedRoiBottom
                    if (roiMaxRight > roiMinLeft && roiMaxBottom > roiMinTop) {
                        // 根据是否找到目标设置颜色
                        val roiColor = if (roiInfoForDraw.found) {
                            Color(0xFF00FF00) // 找到目标：绿色
                        } else {
                            // 未找到：根据扩大次数变化颜色（从黄色到橙色到红色）
                            val expansionRatio = (roiInfoForDraw.expansionCount.toFloat() / 10f).coerceIn(0f, 1f)
                            when {
                                expansionRatio < 0.33f -> Color(0xFFFFFF00) // 黄色
                                expansionRatio < 0.66f -> Color(0xFFFF6600) // 橙色
                                else -> Color(0xFFFF0000) // 红色
                            }
                        }

                        // 根据扩大次数设置透明度
                        val baseRoiAlpha = if (roiInfoForDraw.found) {
                            0.7f
                        } else {
                            (0.3f + roiInfoForDraw.expansionCount * 0.05f).coerceAtMost(0.8f)
                        }

                        // 变换ROI矩形坐标到画布坐标
                        val topLeftBuf = FloatArray(2)
                        topLeftBuf[0] = roiMinLeft
                        topLeftBuf[1] = roiMinTop
                        transform.mapPoints(topLeftBuf)
                        val topLeft = Offset(topLeftBuf[0], topLeftBuf[1])

                        val topRightBuf = FloatArray(2)
                        topRightBuf[0] = roiMaxRight
                        topRightBuf[1] = roiMinTop
                        transform.mapPoints(topRightBuf)
                        val topRight = Offset(topRightBuf[0], topRightBuf[1])

                        val bottomLeftBuf = FloatArray(2)
                        bottomLeftBuf[0] = roiMinLeft
                        bottomLeftBuf[1] = roiMaxBottom
                        transform.mapPoints(bottomLeftBuf)
                        val bottomLeft = Offset(bottomLeftBuf[0], bottomLeftBuf[1])

                        val bottomRightBuf = FloatArray(2)
                        bottomRightBuf[0] = roiMaxRight
                        bottomRightBuf[1] = roiMaxBottom
                        transform.mapPoints(bottomRightBuf)
                        val bottomRight = Offset(bottomRightBuf[0], bottomRightBuf[1])

                        // 计算角标记的长度（根据矩形大小动态调整，最小15像素，最大矩形边长的20%）
                        val roiWidth = (topRight.x - topLeft.x).coerceAtLeast(1f)
                        val roiHeight = (bottomLeft.y - topLeft.y).coerceAtLeast(1f)
                        val cornerLength = (kotlin.math.min(roiWidth, roiHeight) * 0.15f).coerceIn(15f, 40f)

                        val strokeWidth = 5f
                        val animatedRoiAlpha = (baseRoiAlpha * roiVisibilityAlpha * roiPulseAlpha).coerceIn(0f, 1f)
                        val roiColorWithAlpha = roiColor.copy(alpha = animatedRoiAlpha)

                        // 绘制四个角的方框（每个角由两条线段组成L形）
                        // 左上角 ┌
                        drawLine(
                            color = roiColorWithAlpha,
                            start = topLeft,
                            end = Offset(topLeft.x + cornerLength, topLeft.y),
                            strokeWidth = strokeWidth
                        )
                        drawLine(
                            color = roiColorWithAlpha,
                            start = topLeft,
                            end = Offset(topLeft.x, topLeft.y + cornerLength),
                            strokeWidth = strokeWidth
                        )

                        // 右上角 ┐
                        drawLine(
                            color = roiColorWithAlpha,
                            start = topRight,
                            end = Offset(topRight.x - cornerLength, topRight.y),
                            strokeWidth = strokeWidth
                        )
                        drawLine(
                            color = roiColorWithAlpha,
                            start = topRight,
                            end = Offset(topRight.x, topRight.y + cornerLength),
                            strokeWidth = strokeWidth
                        )

                        // 左下角 └
                        drawLine(
                            color = roiColorWithAlpha,
                            start = bottomLeft,
                            end = Offset(bottomLeft.x + cornerLength, bottomLeft.y),
                            strokeWidth = strokeWidth
                        )
                        drawLine(
                            color = roiColorWithAlpha,
                            start = bottomLeft,
                            end = Offset(bottomLeft.x, bottomLeft.y - cornerLength),
                            strokeWidth = strokeWidth
                        )

                        // 右下角 ┘
                        drawLine(
                            color = roiColorWithAlpha,
                            start = bottomRight,
                            end = Offset(bottomRight.x - cornerLength, bottomRight.y),
                            strokeWidth = strokeWidth
                        )
                        drawLine(
                            color = roiColorWithAlpha,
                            start = bottomRight,
                            end = Offset(bottomRight.x, bottomRight.y - cornerLength),
                            strokeWidth = strokeWidth
                        )
                    }
                }
            }
        }

        val gridPreviewAlpha by animateFloatAsState(
            targetValue = if (showGridPreview) 1f else 0f,
            animationSpec = tween(durationMillis = 220),
            label = "grid_preview_alpha"
        )

        if (gridPreviewAlpha > 0.01f) {
            Canvas(modifier = Modifier.fillMaxSize()) {
                val imageSize = previewSize ?: return@Canvas
                val (imageWidth, imageHeight) = imageSize
                val previewScale = min(size.width / imageWidth.toFloat(), size.height / imageHeight.toFloat())
                val displayedWidth = imageWidth * previewScale
                val displayedHeight = imageHeight * previewScale
                val stepX = displayedWidth / gridSize.toFloat()
                val stepY = displayedHeight / gridSize.toFloat()
                val gridColor = Color.White.copy(alpha = 0.24f * gridPreviewAlpha)
                val borderColor = Color.White.copy(alpha = 0.58f * gridPreviewAlpha)

                if (stepX > 0f && stepY > 0f) {
                    for (column in 0..gridSize) {
                        val x = stepX * column
                        drawLine(
                            color = if (column == 0 || column == gridSize) borderColor else gridColor,
                            start = Offset(x, 0f),
                            end = Offset(x, displayedHeight),
                            strokeWidth = if (column == 0 || column == gridSize) 1.5f else 1f
                        )
                    }
                    for (row in 0..gridSize) {
                        val y = stepY * row
                        drawLine(
                            color = if (row == 0 || row == gridSize) borderColor else gridColor,
                            start = Offset(0f, y),
                            end = Offset(displayedWidth, y),
                            strokeWidth = if (row == 0 || row == gridSize) 1.5f else 1f
                        )
                    }
                }
            }
        }

        if (focusMarkerVisible) {
            Canvas(modifier = Modifier.fillMaxSize()) {
                // 图标中心：使用预览实际显示区域中心，避免在 FIT_START + 留白时偏右/偏下
                val (cx, cy) = previewSize?.let { (imageW, imageH) ->
                    val scale = min(size.width / imageW.toFloat(), size.height / imageH.toFloat())
                    Pair(imageW * scale / 2f, imageH * scale / 2f)
                } ?: Pair(size.width / 2f, size.height / 2f)

                // 中间空心圆半径：越大整体图标越醒目
                val radius = 12f
                // 中间空心圆线宽：越粗越有“机械对焦框”质感
                val ringStroke = 4.8f
                // 辐条与圆之间的间隔：出现时会做脉动变化，模拟对焦过程
                val lineGap = (21f + focusMarkerGapOffset.value).coerceIn(6f, 36f)
                // 辐条长度：越长越有方向感
                val lineLen = 64f
                // 辐条线宽：越粗越强调对焦状态
                val lineStroke = 3.6f
                val alpha = focusMarkerAlpha.value.coerceIn(0f, 1f)
                val color = focusMarkerColor.copy(alpha = alpha)
                // 第一根辐条朝向正上方（-90°），其余每隔120°
                val baseAngles = floatArrayOf(-90f, 30f, 150f)
                val rotation = focusMarkerRotation.value

                drawCircle(
                    color = color,
                    radius = radius,
                    center = Offset(cx, cy),
                    style = Stroke(width = ringStroke)
                )

                baseAngles.forEach { base ->
                    val deg = base + rotation
                    val rad = Math.toRadians(deg.toDouble())
                    val ux = kotlin.math.cos(rad).toFloat()
                    val uy = kotlin.math.sin(rad).toFloat()

                    val start = Offset(
                        x = cx + ux * (radius + lineGap),
                        y = cy + uy * (radius + lineGap)
                    )
                    val end = Offset(
                        x = cx + ux * (radius + lineGap + lineLen),
                        y = cy + uy * (radius + lineGap + lineLen)
                    )
                    drawLine(color = color, start = start, end = end, strokeWidth = lineStroke)
                }
            }
        }

        if (probabilityEditEnabled && probabilityMatrix != null) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInteropFilter { event ->
                        fun offsetToDisplayCell(x: Float, y: Float): Pair<Int, Int>? {
                            val sizeInfo = previewSize ?: return null
                            val imageWidth = sizeInfo.first
                            val imageHeight = sizeInfo.second
                            val drawWidth = previewViewRef?.width?.toFloat()?.coerceAtLeast(1f) ?: return null
                            val drawHeight = previewViewRef?.height?.toFloat()?.coerceAtLeast(1f) ?: return null
                            val scale = min(drawWidth / imageWidth.toFloat(), drawHeight / imageHeight.toFloat())
                            val imageDrawWidth = imageWidth * scale
                            val imageDrawHeight = imageHeight * scale
                            if (x < 0f || y < 0f || x > imageDrawWidth || y > imageDrawHeight) {
                                return null
                            }

                            val rows = probabilityMatrix.rowsForImage(imageWidth, imageHeight)
                            val cols = probabilityMatrix.colsForImage(imageWidth, imageHeight)
                            val cellWidth = imageDrawWidth / cols.toFloat()
                            val cellHeight = imageDrawHeight / rows.toFloat()
                            val col = (x / cellWidth).toInt().coerceIn(0, cols - 1)
                            val row = (y / cellHeight).toInt().coerceIn(0, rows - 1)
                            return row to col
                        }

                        fun displayPath(from: Pair<Int, Int>, to: Pair<Int, Int>): List<Pair<Int, Int>> {
                            val rowDelta = to.first - from.first
                            val colDelta = to.second - from.second
                            val steps = max(abs(rowDelta), abs(colDelta))
                            if (steps == 0) return listOf(from)

                            return buildList {
                                for (step in 0..steps) {
                                    val t = step.toFloat() / steps.toFloat()
                                    val row = kotlin.math.round(from.first + rowDelta * t).toInt()
                                    val col = kotlin.math.round(from.second + colDelta * t).toInt()
                                    val cell = row to col
                                    if (lastOrNull() != cell) add(cell)
                                }
                            }
                        }

                        fun applyProbabilityEditPath(targetCell: Pair<Int, Int>) {
                            val sizeInfo = previewSize ?: return
                            val imageWidth = sizeInfo.first
                            val imageHeight = sizeInfo.second
                            val delta = if (probabilityEditMode == ProbabilityEditMode.Increase) {
                                probabilityEditStep
                            } else {
                                -probabilityEditStep
                            }
                            val startCell = dragAnchorProbabilityDisplayCell ?: targetCell
                            if (dragAnchorProbabilityDisplayCell != null && startCell == targetCell) {
                                selectedProbabilityDisplayCell = targetCell
                                return
                            }
                            val pathCells = if (dragAnchorProbabilityDisplayCell == null) {
                                listOf(targetCell)
                            } else {
                                displayPath(startCell, targetCell).drop(1)
                            }
                            if (pathCells.isEmpty()) return
                            var nextMatrix = liveProbabilityMatrix ?: probabilityMatrix

                            pathCells.forEach { (row, col) ->
                                nextMatrix = nextMatrix.updateDisplayedCell(
                                    row = row,
                                    col = col,
                                    imageWidth = imageWidth,
                                    imageHeight = imageHeight,
                                    delta = delta
                                )
                                lastEditedProbabilityDisplayCell = row to col
                            }

                            liveProbabilityMatrix = nextMatrix
                            selectedProbabilityDisplayCell = targetCell
                            dragAnchorProbabilityDisplayCell = targetCell
                            val mappedCell = if (imageWidth > imageHeight) {
                                targetCell.second to targetCell.first
                            } else {
                                targetCell
                            }
                            onProbabilityCellSelected(mappedCell)
                            onProbabilityMatrixChange(nextMatrix)
                            haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                        }

                        when (event.actionMasked) {
                            MotionEvent.ACTION_DOWN -> {
                                lastEditedProbabilityDisplayCell = null
                                dragAnchorProbabilityDisplayCell = null
                                offsetToDisplayCell(event.x, event.y)?.let { applyProbabilityEditPath(it) }
                                true
                            }
                            MotionEvent.ACTION_MOVE -> {
                                for (index in 0 until event.historySize) {
                                    offsetToDisplayCell(
                                        event.getHistoricalX(index),
                                        event.getHistoricalY(index)
                                    )?.let { applyProbabilityEditPath(it) }
                                }
                                offsetToDisplayCell(event.x, event.y)?.let { applyProbabilityEditPath(it) }
                                true
                            }
                            MotionEvent.ACTION_UP,
                            MotionEvent.ACTION_CANCEL -> {
                                lastEditedProbabilityDisplayCell = null
                                dragAnchorProbabilityDisplayCell = null
                                true
                            }
                            else -> true
                        }
                    }
            )
        }

        val zoomProgressTarget = remember(zoomRatioDisplay, minZoomRatioDisplay, maxZoomRatioDisplay) {
            val minZoom = minZoomRatioDisplay.coerceAtLeast(0.01f)
            val maxZoom = maxZoomRatioDisplay.coerceAtLeast(minZoom + 0.0001f)
            val currentZoom = zoomRatioDisplay.coerceIn(minZoom, maxZoom)

            val relativeZoom = (currentZoom / minZoom).coerceAtLeast(1f)
            val maxRelativeZoom = (maxZoom / minZoom).coerceAtLeast(1.0001f)

            (ln(relativeZoom) / ln(maxRelativeZoom)).coerceIn(0f, 1f)
        }
        val animatedZoomProgress by animateFloatAsState(
            targetValue = zoomProgressTarget,
            animationSpec = tween(durationMillis = 220),
            label = "zoom_indicator_progress"
        )

        BoxWithConstraints(
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 6.dp)
        ) {
            val minBarWidth = 24.dp
            val barWidth = minBarWidth + (maxWidth - minBarWidth) * animatedZoomProgress

            Column(
                modifier = Modifier
                    .align(Alignment.BottomEnd),
                horizontalAlignment = Alignment.End,
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Text(
                    text = "${"%.1f".format(zoomRatioDisplay)}x",
                    color = Color.White,
                )
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(4.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .align(Alignment.CenterEnd)
                            .fillMaxWidth()
                            .height(4.dp)
                            .background(
                                color = Color(0x40FFFFFF),
                                shape = RoundedCornerShape(4.dp)
                            )
                    )
                    Box(
                        modifier = Modifier
                            .align(Alignment.CenterEnd)
                            .width(barWidth)
                            .height(4.dp)
                            .background(
                                color = Color.White,
                                shape = RoundedCornerShape(4.dp)
                            )
                    )
                }
            }
        }
    }
}

/**
 * ROI范围数据类
 */
private data class RoiRange(val left: Int, val top: Int, val right: Int, val bottom: Int)

/**
 * 手动选择的ROI坐标数据类
 */
private data class ManualRoiCoords(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

/**
 * ROI可视化信息数据类（原图坐标）
 */
private data class RoiVisualizationInfo(
    val left: Float,        // ROI左边界（原图坐标）
    val top: Float,         // ROI上边界（原图坐标）
    val right: Float,       // ROI右边界（原图坐标）
    val bottom: Float,      // ROI下边界（原图坐标）
    val expansionCount: Int,   // 扩大次数
    val found: Boolean         // 是否找到目标
)

/**
 * 计算三个点形成的三角形的外接圆半径
 */
private fun calculateCircumradius(
    p1: Offset,
    p2: Offset,
    p3: Offset
): Float {
    // 计算三条边的长度
    val a = kotlin.math.sqrt((p2.x - p3.x) * (p2.x - p3.x) + (p2.y - p3.y) * (p2.y - p3.y))
    val b = kotlin.math.sqrt((p1.x - p3.x) * (p1.x - p3.x) + (p1.y - p3.y) * (p1.y - p3.y))
    val c = kotlin.math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y))

    // 计算三角形面积（海伦公式）
    val s = (a + b + c) / 2f
    val area = kotlin.math.sqrt((s * (s - a) * (s - b) * (s - c)).coerceAtLeast(0f))

    // 如果面积为0或太小，使用最大边长作为直径
    if (area < 1e-6f) {
        val maxEdge = kotlin.math.max(kotlin.math.max(a, b), c)
        return maxEdge / 2f
    }

    // 外接圆半径公式: R = (a * b * c) / (4 * area)
    val radius = (a * b * c) / (4f * area)
    return radius
}

/**
 * 处理图像并检测亮点
 */
private fun processImage(
    imageProxy: ImageProxy,
    detector: BrightSpotDetector,
    gridSize: Int = 50,
    probabilityMatrix: ProbabilityMatrix32x24? = null,
    enableRoiOptimization: Boolean = true,
    lastTriSpots: Triple<Offset, Offset, Offset>? = null,
    lastTriSpotsImageSize: Pair<Int, Int>? = null,
    manualRoi: ManualRoiCoords? = null,
    limitRegion: ManualRoiCoords? = null,
    forceGlobalSearch: Boolean = false,
    onRoiInfo: ((RoiVisualizationInfo?) -> Unit)? = null,
    onSpotsDetected: (List<BrightSpot>, Pair<Int, Int>, Boolean) -> Unit,
    captureBitmapForCalibration: Boolean = false,
    onBitmapCaptured: (Bitmap) -> Unit = {}
) {
    try {
        val targetSpotCount = 3
        val bitmap = imageProxy.toBitmap()
        val rotatedBitmap = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees.toFloat())

        val maxDim = 1280f
        val origW = rotatedBitmap.width
        val origH = rotatedBitmap.height
        val scaleDown = kotlin.math.min(1f, maxDim / kotlin.math.max(origW, origH))
        val workingBitmap = if (scaleDown < 1f) {
            rotatedBitmap.scale(
                (origW * scaleDown).toInt().coerceAtLeast(1),
                (origH * scaleDown).toInt().coerceAtLeast(1)
            )
        } else rotatedBitmap

        val workingScale = kotlin.math.min(
            workingBitmap.width.toFloat() / rotatedBitmap.width,
            workingBitmap.height.toFloat() / rotatedBitmap.height
        )

        // 限制区域映射到 workingBitmap
        data class LimitRegionWorking(val left: Int, val top: Int, val right: Int, val bottom: Int)
        val effectiveLimitRegion = if (forceGlobalSearch) null else limitRegion
        val effectiveManualRoi = if (forceGlobalSearch) null else manualRoi

        val limitRegionWorking = effectiveLimitRegion?.let {
            val w = workingBitmap.width
            val h = workingBitmap.height
            val l = (it.left * workingScale).toInt().coerceIn(0, w - 1)
            val t = (it.top * workingScale).toInt().coerceIn(0, h - 1)
            val r = (it.right * workingScale).toInt().coerceIn(l + 1, w - 1)
            val b = (it.bottom * workingScale).toInt().coerceIn(t + 1, h - 1)
            LimitRegionWorking(l, t, r, b)
        }

        // 上一次三色点映射到当前图像
        val initialRoiInfo = if (!forceGlobalSearch && enableRoiOptimization && lastTriSpots != null && lastTriSpotsImageSize != null) {
            val (lastW, lastH) = lastTriSpotsImageSize
            val (currW, currH) = rotatedBitmap.width to rotatedBitmap.height
            val scaleX = currW.toFloat() / lastW
            val scaleY = currH.toFloat() / lastH
            val p1 = Offset(lastTriSpots.first.x * scaleX, lastTriSpots.first.y * scaleY)
            val p2 = Offset(lastTriSpots.second.x * scaleX, lastTriSpots.second.y * scaleY)
            val p3 = Offset(lastTriSpots.third.x * scaleX, lastTriSpots.third.y * scaleY)
            val centerX = (p1.x + p2.x + p3.x) / 3f
            val centerY = (p1.y + p2.y + p3.y) / 3f
            val radius = calculateCircumradius(p1, p2, p3) * 2f
            Triple(centerX * workingScale, centerY * workingScale, radius * workingScale)
        } else null

        var lastTriRadius: Float? = null // 记录上一个外接圆半径
        val recentAttempts = mutableListOf<Boolean>() // 记录最近尝试的结果，成功为true，失败为false
        val maxRecentAttempts = 7  // 最近的尝试次数
        val failureThreshold = 4  // 失败次数阈值，如果失败次数大于此阈值，则触发全局搜索

        val (spotsScaled, foundInManualRoi) = if (effectiveManualRoi != null) {
            // 手动ROI的逻辑与之前相同，保持不变
            val roiLeftOrig = effectiveManualRoi.left
            val roiTopOrig = effectiveManualRoi.top
            val roiRightOrig = effectiveManualRoi.right
            val roiBottomOrig = effectiveManualRoi.bottom
            val workingWidth = workingBitmap.width
            val workingHeight = workingBitmap.height

            var roiLeft = (roiLeftOrig * workingScale).toInt().coerceIn(0, workingWidth - 1)
            var roiTop = (roiTopOrig * workingScale).toInt().coerceIn(0, workingHeight - 1)
            var roiRight = (roiRightOrig * workingScale).toInt().coerceAtLeast(roiLeft + 1).coerceAtMost(workingWidth - 1)
            var roiBottom = (roiBottomOrig * workingScale).toInt().coerceAtLeast(roiTop + 1).coerceAtMost(workingHeight - 1)

            // 与限制区域取交集
            if (limitRegionWorking != null) {
                roiLeft = maxOf(roiLeft, limitRegionWorking.left)
                roiTop = maxOf(roiTop, limitRegionWorking.top)
                roiRight = minOf(roiRight, limitRegionWorking.right)
                roiBottom = minOf(roiBottom, limitRegionWorking.bottom)
                if (roiRight <= roiLeft || roiBottom <= roiTop) {
                    roiLeft = limitRegionWorking.left
                    roiTop = limitRegionWorking.top
                    roiRight = limitRegionWorking.right
                    roiBottom = limitRegionWorking.bottom
                }
            }

            onRoiInfo?.invoke(null)

            val manualRoiSpots = detector.detectBrightSpots(
                bitmap = workingBitmap,
                threshold = 80f,
                gridSize = gridSize,                roiLeft = roiLeft,
                roiTop = roiTop,
                roiRight = roiRight,
                roiBottom = roiBottom,
                probabilityMatrix = probabilityMatrix
            )

            val foundFlag = manualRoiSpots.size == 3
            Pair(manualRoiSpots, foundFlag)

        } else if (initialRoiInfo != null) {
            // ROI 扩展 + 三色点稳定性判断
            val (centerX, centerY, initialRoiRadius) = initialRoiInfo
            var currentRadius = initialRoiRadius
            val workingWidth = workingBitmap.width
            val workingHeight = workingBitmap.height
            val maxRadiusLimit = limitRegionWorking?.let {
                maxOf(it.right - it.left, it.bottom - it.top) * 0.5f
            }
            val maxAllowedRadius = min(max(workingWidth, workingHeight) * 0.4f, maxRadiusLimit ?: Float.MAX_VALUE)

            var foundSpots: List<BrightSpot>? = null
            var finalRoiInfo: RoiVisualizationInfo? = null
            var expansionCount = 0
            val maxExpansions = 10

            while (foundSpots == null && currentRadius <= maxAllowedRadius && expansionCount < maxExpansions) {
                var roiLeft = (centerX - currentRadius).toInt().coerceAtLeast(0)
                var roiTop = (centerY - currentRadius).toInt().coerceAtLeast(0)
                var roiRight = (centerX + currentRadius).toInt().coerceAtMost(workingWidth - 1)
                var roiBottom = (centerY + currentRadius).toInt().coerceAtMost(workingHeight - 1)

                // 与限制区域取交集
                if (limitRegionWorking != null) {
                    roiLeft = maxOf(roiLeft, limitRegionWorking.left)
                    roiTop = maxOf(roiTop, limitRegionWorking.top)
                    roiRight = minOf(roiRight, limitRegionWorking.right)
                    roiBottom = minOf(roiBottom, limitRegionWorking.bottom)
                    if (roiRight <= roiLeft || roiBottom <= roiTop) {
                        currentRadius *= 1.3f
                        expansionCount++
                        continue
                    }
                }

                val roiSpots = detector.detectBrightSpots(
                    bitmap = workingBitmap,
                    threshold = 80f,
                    gridSize = gridSize,                    roiLeft = roiLeft,
                    roiTop = roiTop,
                    roiRight = roiRight,
                    roiBottom = roiBottom,
                    probabilityMatrix = probabilityMatrix
                )

                // 三色点稳定性判断
                val isStable = if (roiSpots.size == 3 && lastTriRadius != null) {
                    val currRadius = calculateCircumradius(roiSpots[0].position, roiSpots[1].position, roiSpots[2].position)
                    val changeRatio = abs(currRadius - lastTriRadius) / lastTriRadius
                    changeRatio <= 0.3f
                } else roiSpots.size == 3

                if (roiSpots.size == 3 && isStable) {
                    foundSpots = roiSpots
                    val originalRoiLeft = roiLeft / workingScale
                    val originalRoiTop = roiTop / workingScale
                    val originalRoiRight = roiRight / workingScale
                    val originalRoiBottom = roiBottom / workingScale
                    finalRoiInfo = RoiVisualizationInfo(
                        left = originalRoiLeft,
                        top = originalRoiTop,
                        right = originalRoiRight,
                        bottom = originalRoiBottom,
                        expansionCount = expansionCount,
                        found = true
                    )
                    // 记录当前外接圆半径
                    lastTriRadius = calculateCircumradius(roiSpots[0].position, roiSpots[1].position, roiSpots[2].position)
                    // 记录成功
                    recentAttempts.add(true)
                    if (recentAttempts.size > maxRecentAttempts) {
                        recentAttempts.removeAt(0)
                    }
                } else {
                    currentRadius *= 1.3f
                    expansionCount++
                    // 记录失败
                    recentAttempts.add(false)
                    if (recentAttempts.size > maxRecentAttempts) {
                        recentAttempts.removeAt(0)
                    }
                }

                // 如果最近7次中失败超过4次，触发全局搜索
                if (recentAttempts.count { !it } >= failureThreshold) {
                    foundSpots = null
                    onRoiInfo?.invoke(null)
                    break
                }
            }

            // 如果找不到三色点，清空ROI并开始全图搜索
            if (foundSpots == null) {
                onRoiInfo?.invoke(null)
                val searchResult = limitRegionWorking?.let {
                    detector.detectBrightSpots(
                        bitmap = workingBitmap,
                        threshold = 80f,
                        gridSize = gridSize,                        roiLeft = it.left,
                        roiTop = it.top,
                        roiRight = it.right,
                        roiBottom = it.bottom,
                        probabilityMatrix = probabilityMatrix
                    )
                } ?: detector.detectBrightSpots(
                    bitmap = workingBitmap,
                    threshold = 80f,
                    gridSize = gridSize,                    probabilityMatrix = probabilityMatrix
                )
                Pair(searchResult, false)
            } else {
                finalRoiInfo?.let { onRoiInfo?.invoke(it) }
                Pair(foundSpots, false)
            }

        } else {
            // 全图搜索逻辑保持原样
            onRoiInfo?.invoke(null)
            val searchResult = limitRegionWorking?.let {
                detector.detectBrightSpots(
                    bitmap = workingBitmap,
                    threshold = 80f,
                    gridSize = gridSize,                    roiLeft = it.left,
                    roiTop = it.top,
                    roiRight = it.right,
                    roiBottom = it.bottom,
                    probabilityMatrix = probabilityMatrix
                )
            } ?: detector.detectBrightSpots(
                bitmap = workingBitmap,
                threshold = 80f,
                gridSize = gridSize,                probabilityMatrix = probabilityMatrix
            )
            Pair(searchResult, false)
        }

        val invScale = if (scaleDown < 1f) 1f / scaleDown else 1f
        val spotsMapped = spotsScaled.map { s -> s.copy(position = Offset(s.position.x * invScale, s.position.y * invScale)) }

        val spots = limitRegion?.let { region ->
            spotsMapped.filter { it.position.x in region.left..region.right && it.position.y in region.top..region.bottom }
        } ?: spotsMapped

        val refinedSpots = refineSpotsOnOriginal(rotatedBitmap, spots, gridSize)
        onSpotsDetected(refinedSpots, Pair(rotatedBitmap.width, rotatedBitmap.height), foundInManualRoi)

        if (captureBitmapForCalibration) {
            onBitmapCaptured(rotatedBitmap.copy(rotatedBitmap.config ?: Bitmap.Config.ARGB_8888, false))
        }

        if (rotatedBitmap != bitmap) rotatedBitmap.recycle()
        if (workingBitmap != rotatedBitmap) workingBitmap.recycle()
        bitmap.recycle()

    } catch (e: Exception) {
        e.printStackTrace()
    }
}

// 辅助函数
private fun Offset.getDistance(): Float = kotlin.math.sqrt(x * x + y * y)
private operator fun Offset.minus(other: Offset) = Offset(x - other.x, y - other.y)

private fun refineSpotsOnOriginal(
    bitmap: Bitmap,
    spots: List<BrightSpot>,
    gridSize: Int
): List<BrightSpot> {
    if (spots.isEmpty()) return spots
    val width = bitmap.width
    val height = bitmap.height
    val step = kotlin.math.max(1, kotlin.math.max(width / gridSize, height / gridSize))
    val radius = (step * 2f).toInt().coerceAtLeast(6).coerceAtMost(kotlin.math.min(width, height) / 6)

    fun weightFor(pixel: Int, prefer: LedColor, backgroundDarkness: Float = 0.5f): Float {
        val r = ((pixel shr 16) and 0xFF).toFloat()
        val g = ((pixel shr 8) and 0xFF).toFloat()
        val b = (pixel and 0xFF).toFloat()

        // 亮度与目标色突出度
        val colorScore = when (prefer) {
            LedColor.RED -> (r - (g + b) / 2f).coerceAtLeast(0f)
            LedColor.GREEN -> (g - (r + b) / 2f).coerceAtLeast(0f)
            LedColor.BLUE -> (b - (r + g) / 2f).coerceAtLeast(0f)
            else -> (r + g + b).coerceAtLeast(0f)
        }

        // 背景黑色权重（V越低越黑，额外加分）
        val v = (r + g + b) / (255f * 3f)  // 0..1
        val backgroundScore = (1f - v).pow(2) * backgroundDarkness

        return colorScore * (1f + backgroundScore)
    }

    fun refineOne(cx: Float, cy: Float, prefer: LedColor): Offset {
        var centerX = cx
        var centerY = cy
        repeat(2) {
            val x0 = (centerX - radius).toInt().coerceAtLeast(0)
            val y0 = (centerY - radius).toInt().coerceAtLeast(0)
            val x1 = (centerX + radius).toInt().coerceAtMost(width - 1)
            val y1 = (centerY + radius).toInt().coerceAtMost(height - 1)
            val wRect = x1 - x0 + 1
            val hRect = y1 - y0 + 1
            if (wRect <= 0 || hRect <= 0) return@repeat
            val buf = IntArray(wRect * hRect)
            bitmap.getPixels(buf, 0, wRect, x0, y0, wRect, hRect)
            var sumW = 0f
            var sumX = 0f
            var sumY = 0f
            // 轻度径向加权，抑制边缘（高斯近似常数系数）
            val radF = radius.toFloat()
            val invTwoSigma2 = 1f / (2f * (radF * 0.6f) * (radF * 0.6f))
            for (yy in 0 until hRect) {
                val row = yy * wRect
                val yAbs = (y0 + yy).toFloat()
                for (xx in 0 until wRect) {
                    val xAbs = (x0 + xx).toFloat()
                    val w = weightFor(buf[row + xx], prefer, backgroundDarkness = 0.6f)
                    if (w <= 0f) continue
                    val dx = xAbs - centerX
                    val dy = yAbs - centerY
                    val radial = kotlin.math.exp(-(dx * dx + dy * dy) * invTwoSigma2)
                    val ww = w * radial
                    sumW += ww
                    sumX += ww * xAbs
                    sumY += ww * yAbs
                }
            }
            if (sumW > 0f) {
                centerX = sumX / sumW
                centerY = sumY / sumW
            } else {
                return Offset(centerX, centerY)
            }
        }
        return Offset(centerX, centerY)
    }

    return spots.map { s ->
        val prefer = when {
            s.color != LedColor.UNKNOWN -> s.color
            s.redIntensity >= s.greenIntensity && s.redIntensity >= s.blueIntensity -> LedColor.RED
            s.greenIntensity >= s.redIntensity && s.greenIntensity >= s.blueIntensity -> LedColor.GREEN
            else -> LedColor.BLUE
        }
        val refined = refineOne(s.position.x, s.position.y, prefer)
        s.copy(position = refined)
    }
}

/**
 * 旋转Bitmap以匹配设备方向
 */
private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
    if (degrees == 0f) return bitmap

    val matrix = Matrix()
    matrix.postRotate(degrees)

    return Bitmap.createBitmap(
        bitmap,
        0,
        0,
        bitmap.width,
        bitmap.height,
        matrix,
        true
    )
}


