package org.example.tridotpnp

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import androidx.compose.ui.geometry.Offset
import androidx.core.graphics.scale
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.tan

internal data class RoiRange(val left: Int, val top: Int, val right: Int, val bottom: Int)

internal data class ManualRoiCoords(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

internal data class RoiVisualizationInfo(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val expansionCount: Int,
    val found: Boolean
)

private data class PreparedFrame(
    val rotatedBitmap: Bitmap,
    val workingBitmap: Bitmap,
    val scaleDown: Float,
    val workingScale: Float
)

private data class WorkingRegion(val left: Int, val top: Int, val right: Int, val bottom: Int)

private data class DetectionResult(
    val spots: List<BrightSpot>,
    val foundInManualRoi: Boolean
)

private enum class DetectionMode {
    ManualRoi,
    AdaptiveRoi,
    Global
}

internal fun calculateCircumradius(
    p1: Offset,
    p2: Offset,
    p3: Offset
): Float {
    val a = sqrt((p2.x - p3.x) * (p2.x - p3.x) + (p2.y - p3.y) * (p2.y - p3.y))
    val b = sqrt((p1.x - p3.x) * (p1.x - p3.x) + (p1.y - p3.y) * (p1.y - p3.y))
    val c = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y))

    val s = (a + b + c) / 2f
    val area = sqrt((s * (s - a) * (s - b) * (s - c)).coerceAtLeast(0f))
    if (area < 1e-6f) {
        return max(max(a, b), c) / 2f
    }

    return (a * b * c) / (4f * area)
}

/** 远距前端调度：far-primary 下默认每 N 帧跑一次，主线不可信时立即跑。 */
internal object FarAssistScheduler {
    fun shouldRun(frameCounter: Long, mainPlausible: Boolean, period: Int): Boolean {
        if (period <= 1) return true
        if (!mainPlausible) return true
        return frameCounter % period.toLong() == 0L
    }
}

private var farFrameCounter = 0L

internal fun processImage(
    imageProxy: ImageProxy,
    detector: BrightSpotDetector,
    tuning: AppTuningSettings = AppTuningSettings(),
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
        val bitmap = imageProxy.toBitmap()
        val frame = prepareFrame(bitmap, imageProxy, tuning)
        val effectiveLimitRegion = if (forceGlobalSearch) null else limitRegion
        val effectiveManualRoi = if (forceGlobalSearch) null else manualRoi
        val limitRegionWorking = effectiveLimitRegion?.toWorkingRegion(
            frame.workingScale,
            frame.workingBitmap.width,
            frame.workingBitmap.height
        )

        val detectionMode = resolveDetectionMode(
            enableRoiOptimization = enableRoiOptimization,
            manualRoi = effectiveManualRoi,
            lastTriSpots = lastTriSpots,
            lastTriSpotsImageSize = lastTriSpotsImageSize
        )
        // 端侧小检测器：assets 里有权重才启用（失败静默退化）
        farSpotDetector.classifier = DotClassifierAssets.get()
        val detectionResult = runDetection(
            mode = detectionMode,
            detector = detector,
            frame = frame,
            tuning = tuning,
            manualRoi = effectiveManualRoi,
            lastTriSpots = lastTriSpots,
            lastTriSpotsImageSize = lastTriSpotsImageSize,
            limitRegionWorking = limitRegionWorking,
            probabilityMatrix = probabilityMatrix,
            onRoiInfo = onRoiInfo
        )

        val spots = mapSpotsBackToOriginal(
            spots = detectionResult.spots,
            scaleDown = frame.scaleDown,
            limitRegion = limitRegion
        )
        val refinedSpots = refineSpotsOnOriginal(frame.rotatedBitmap, spots, tuning)
        val expectedEdgePx = expectedTripleEdgePx(
            frame.rotatedBitmap.width, lastTriSpots, lastTriSpotsImageSize
        )
        // 主线三点「凑齐」不等于「正确」：几何不自洽时同样交给远距辅助重找一次。
        val mainTriplePlausible = refinedSpots.size == 3 &&
            TripleConsistency.isPlausible(refinedSpots.map { floatArrayOf(it.position.x, it.position.y) }.toTypedArray(), expectedEdgePx)
        farSpotDetector.maxDim = tuning.farFrontendMaxDim
        farFrameCounter += 1
        val farRunDue = FarAssistScheduler.shouldRun(
            frameCounter = farFrameCounter,
            mainPlausible = mainTriplePlausible,
            period = tuning.farAssistPeriod
        )
        val farAssist = if (tuning.farDotAssistEnabled &&
            (if (tuning.farDotPrimary) farRunDue else !mainTriplePlausible)
        ) {
            farDotAssist(
                bitmap = frame.rotatedBitmap,
                spots = refinedSpots,
                tuning = tuning,
                lastTriSpots = lastTriSpots,
                lastTriSpotsImageSize = lastTriSpotsImageSize
            )
        } else {
            null
        }
        val farPlausible = farAssist != null &&
            TripleConsistency.isPlausible(
                farAssist.spots.map { floatArrayOf(it.position.x, it.position.y) }.toTypedArray(),
                expectedEdgePx
            )
        val source = TripleChooser.choose(
            mainPlausible = mainTriplePlausible,
            farAvailable = farAssist != null,
            farValidated = farAssist?.validated == true,
            farPlausible = farPlausible,
            farPrimary = tuning.farDotPrimary
        )
        val finalSpots = if (source == TripleSource.FAR && farAssist != null) farAssist.spots else refinedSpots
        onSpotsDetected(
            finalSpots,
            frame.rotatedBitmap.width to frame.rotatedBitmap.height,
            detectionResult.foundInManualRoi
        )

        captureCalibrationBitmapIfNeeded(
            captureBitmapForCalibration = captureBitmapForCalibration,
            rotatedBitmap = frame.rotatedBitmap,
            onBitmapCaptured = onBitmapCaptured
        )

        recyclePreparedFrame(bitmap, frame)
    } catch (e: Exception) {
        e.printStackTrace()
    }
}


/**
 * 三点几何自洽性判断（主线检测结果的兜底校验）。
 *
 * 主线 `BrightSpotDetector` 在远距/强泛光下可能给出三个「高分但几何荒谬」的点，
 * 这种情况 `refinedSpots.size == 3`，远距辅助不会触发，错误位姿会直接进 PnP。
 * 这里先用等边性与「上一帧预测的视边长」做廉价校验，不合规才让远距辅助接管。
 */
internal object TripleConsistency {
    /** 三点两两距离。 */
    fun edgeLengths(points: Array<FloatArray>): FloatArray {
        if (points.size != 3) return FloatArray(0)
        val d01 = hypot(points[0][0] - points[1][0], points[0][1] - points[1][1])
        val d12 = hypot(points[1][0] - points[2][0], points[1][1] - points[2][1])
        val d20 = hypot(points[2][0] - points[0][0], points[2][1] - points[0][1])
        return floatArrayOf(d01, d12, d20)
    }

    fun meanEdge(points: Array<FloatArray>): Float {
        val e = edgeLengths(points)
        if (e.isEmpty()) return 0f
        return (e[0] + e[1] + e[2]) / 3f
    }

    /**
     * @param expectedEdgePx 上一帧预测的视边长（<=1 表示无先验，只做形状校验）
     * @param tolerance 与 expectedEdgePx 允许的相对偏差
     */
    fun isPlausible(points: Array<FloatArray>, expectedEdgePx: Float, tolerance: Float = 0.45f): Boolean {
        if (points.size != 3) return false
        val e = edgeLengths(points)
        val mean = (e[0] + e[1] + e[2]) / 3f
        if (mean < 6f) return false
        val spread = (e.max() - e.min()) / mean
        if (spread > 0.5f) return false
        if (expectedEdgePx > 1f) {
            val ratio = mean / expectedEdgePx
            if (ratio < 1f - tolerance || ratio > 1f + tolerance) return false
        }
        return true
    }
}

/** 三点来源：主线亮斑检测 or 对比度前端（远距辅助）。 */
internal enum class TripleSource { MAIN, FAR }

/**
 * 主线 vs 对比度前端的三点选择策略（纯逻辑，便于单测）。
 *
 * 依据（Python 端侧链路模拟）：主线在 realistic 域 det@10≈0.22、对比度前端 ≈0.80，
 * 因此默认「对比度前端优先」，但只在它给出可信结果时（给了 PnP 位姿校验，或几何自洽）。
 * 关掉 farPrimary 则退回旧语义：只有主线不可信时才用对比度前端。
 */
internal object TripleChooser {
    fun choose(
        mainPlausible: Boolean,
        farAvailable: Boolean,
        farValidated: Boolean,
        farPlausible: Boolean,
        farPrimary: Boolean
    ): TripleSource {
        if (!farAvailable || !farPlausible) return TripleSource.MAIN
        if (farPrimary) return TripleSource.FAR
        return if (mainPlausible) TripleSource.MAIN else TripleSource.FAR
    }
}

private val farSpotDetector = FarSpotDetector()

/** 对比度前端的结果 + 这次是否用上了 PnP 位姿校验。 */
private data class FarAssist(val spots: List<BrightSpot>, val validated: Boolean)

/**
 * 远距辅助：常规检测凑不齐三点时，用「颜色差 + 形态学顶帽 + 噪声自适应阈值」前端再试一次。
 * 光斑在中远距离只有 1-2px，绝对亮度门与 720p 降采样都会把它抹掉；
 * 这条路径不做绝对亮度判断，阈值由画面自身的高频噪声与亮尾决定，最后还要过等边几何校验。
 */
private fun farDotAssist(
    bitmap: Bitmap,
    spots: List<BrightSpot>,
    tuning: AppTuningSettings,
    lastTriSpots: Triple<Offset, Offset, Offset>?,
    lastTriSpotsImageSize: Pair<Int, Int>?
): FarAssist? {
    val validator = farTripleValidator(bitmap, tuning)
    val expectedEdgePx = expectedTripleEdgePx(bitmap.width, lastTriSpots, lastTriSpotsImageSize)
    return try {
        farSpotDetector.detect(
            bitmap = bitmap,
            expectedEdgePx = expectedEdgePx,
            predictedSpots = lastTriSpots?.toList(),
            validator = validator
        )?.takeIf { it.size == 3 }?.let { FarAssist(it, validated = validator != null) }
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

/** 上一帧三点在本帧分辨率下的平均视边长；无先验返回 0。 */
private fun expectedTripleEdgePx(
    width: Int,
    lastTriSpots: Triple<Offset, Offset, Offset>?,
    lastTriSpotsImageSize: Pair<Int, Int>?
): Float {
    if (lastTriSpots == null || lastTriSpotsImageSize == null) return 0f
    val (a, b, c) = lastTriSpots
    val e1 = hypot(a.x - b.x, a.y - b.y)
    val e2 = hypot(b.x - c.x, b.y - c.y)
    val e3 = hypot(c.x - a.x, c.y - a.y)
    return ((e1 + e2 + e3) / 3f) * (width.toFloat() / lastTriSpotsImageSize.first.toFloat())
}

/**
 * 用上一帧位姿给远距回退检测做候选校验：单帧里几何上都像三角形时，
 * 只有「PnP 解出的平移/旋转与上一帧连续」的那组才被接受。
 */
private fun farTripleValidator(bitmap: Bitmap, tuning: AppTuningSettings): TripleValidator? =
    buildFarTripleValidator(
        width = bitmap.width,
        height = bitmap.height,
        fovDegrees = tuning.estimatedFovDegrees,
        edgeMm = tuning.knownTriangleEdgeLengthMm,
        workMaxDim = farSpotDetector.maxDim,
        tracker = AppTemporalTracker.tracker
    )

/**
 * 用上一帧位姿给远距回退检测做候选校验：单帧里几何上都像三角形时，
 * 只有「PnP 解出的平移/旋转与上一帧连续」的那组才被接受。
 */
internal fun buildFarTripleValidator(
    width: Int,
    height: Int,
    fovDegrees: Float,
    edgeMm: Float,
    workMaxDim: Int,
    tracker: PnPTemporalTracker
): TripleValidator? {
    val prior = tracker.currentSnapshot() ?: return null
    val priorRot = prior.rotation ?: return null
    val workScale = min(1f, workMaxDim.toFloat() / max(width, height).toFloat())
    val frame = (fovDegrees.toDouble().coerceIn(10.0, 170.0) * PI / 180.0)
    val fx = (width / (2.0 * tan(frame / 2.0))).toFloat()
    val cx = width / 2f
    val cy = height / 2f
    val depth = sqrt(
        prior.translationMm[0] * prior.translationMm[0] +
            prior.translationMm[1] * prior.translationMm[1] +
            prior.translationMm[2] * prior.translationMm[2]
    )
    val transGate = max(150f, 0.25f * depth)
    val rotGate = 25f
    return TripleValidator { tri ->
        val u = FloatArray(3) { ((tri[it].u / workScale) - cx) / fx }
        val v = FloatArray(3) { ((tri[it].v / workScale) - cy) / fx }
        // P3P 有两个镜像解：逐个试，取与上一帧最一致的那个（否则会锁死在镜像根上）
        var bestCost = Float.MAX_VALUE
        for (pose in tracker.posesForTriple(u, v, edgeMm)) {
            val t = pose.second
            val tErr = sqrt(
                (t[0] - prior.translationMm[0]) * (t[0] - prior.translationMm[0]) +
                    (t[1] - prior.translationMm[1]) * (t[1] - prior.translationMm[1]) +
                    (t[2] - prior.translationMm[2]) * (t[2] - prior.translationMm[2])
            )
            if (tErr > transGate) continue
            val ang = tracker.rotationAngleDeg(pose.first, priorRot)
            if (ang > rotGate) continue
            val cost = tErr / transGate + ang / rotGate
            if (cost < bestCost) bestCost = cost
        }
        if (bestCost == Float.MAX_VALUE) return@TripleValidator null
        bestCost
    }
}

private fun resolveDetectionMode(
    enableRoiOptimization: Boolean,
    manualRoi: ManualRoiCoords?,
    lastTriSpots: Triple<Offset, Offset, Offset>?,
    lastTriSpotsImageSize: Pair<Int, Int>?
): DetectionMode {
    if (manualRoi != null) return DetectionMode.ManualRoi
    if (enableRoiOptimization && lastTriSpots != null && lastTriSpotsImageSize != null) {
        return DetectionMode.AdaptiveRoi
    }
    return DetectionMode.Global
}

private fun runDetection(
    mode: DetectionMode,
    detector: BrightSpotDetector,
    frame: PreparedFrame,
    tuning: AppTuningSettings,
    manualRoi: ManualRoiCoords?,
    lastTriSpots: Triple<Offset, Offset, Offset>?,
    lastTriSpotsImageSize: Pair<Int, Int>?,
    limitRegionWorking: WorkingRegion?,
    probabilityMatrix: ProbabilityMatrix32x24?,
    onRoiInfo: ((RoiVisualizationInfo?) -> Unit)?
): DetectionResult {
    return when (mode) {
        DetectionMode.ManualRoi -> {
            onRoiInfo?.invoke(null)
            detectSpotsInManualRoi(
                detector = detector,
                frame = frame,
                tuning = tuning,
                manualRoi = requireNotNull(manualRoi),
                limitRegionWorking = limitRegionWorking,
                probabilityMatrix = probabilityMatrix
            )
        }

        DetectionMode.AdaptiveRoi -> {
            detectSpotsWithAdaptiveRoi(
                detector = detector,
                frame = frame,
                tuning = tuning,
                lastTriSpots = requireNotNull(lastTriSpots),
                lastTriSpotsImageSize = requireNotNull(lastTriSpotsImageSize),
                limitRegionWorking = limitRegionWorking,
                probabilityMatrix = probabilityMatrix,
                onRoiInfo = onRoiInfo
            )
        }

        DetectionMode.Global -> {
            onRoiInfo?.invoke(null)
            detectSpotsGlobally(
                detector = detector,
                bitmap = frame.workingBitmap,
                threshold = tuning.detectionThreshold,
                gridSize = tuning.gridSize,
                limitRegionWorking = limitRegionWorking,
                probabilityMatrix = probabilityMatrix
            )
        }
    }
}

private fun captureCalibrationBitmapIfNeeded(
    captureBitmapForCalibration: Boolean,
    rotatedBitmap: Bitmap,
    onBitmapCaptured: (Bitmap) -> Unit
) {
    if (!captureBitmapForCalibration) return
    onBitmapCaptured(
        rotatedBitmap.copy(
            rotatedBitmap.config ?: Bitmap.Config.ARGB_8888,
            false
        )
    )
}

private fun prepareFrame(
    bitmap: Bitmap,
    imageProxy: ImageProxy,
    tuning: AppTuningSettings
): PreparedFrame {
    val rotatedBitmap = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees.toFloat())
    val maxDim = tuning.workingMaxDimension.toFloat().coerceAtLeast(64f)
    val origW = rotatedBitmap.width
    val origH = rotatedBitmap.height
    val scaleDown = min(1f, maxDim / max(origW, origH))
    val workingBitmap = if (scaleDown < 1f) {
        rotatedBitmap.scale(
            (origW * scaleDown).toInt().coerceAtLeast(1),
            (origH * scaleDown).toInt().coerceAtLeast(1)
        )
    } else {
        rotatedBitmap
    }
    val workingScale = min(
        workingBitmap.width.toFloat() / rotatedBitmap.width,
        workingBitmap.height.toFloat() / rotatedBitmap.height
    )
    return PreparedFrame(rotatedBitmap, workingBitmap, scaleDown, workingScale)
}

private fun ManualRoiCoords.toWorkingRegion(
    workingScale: Float,
    bitmapWidth: Int,
    bitmapHeight: Int
): WorkingRegion {
    val left = (this.left * workingScale).toInt().coerceIn(0, bitmapWidth - 1)
    val top = (this.top * workingScale).toInt().coerceIn(0, bitmapHeight - 1)
    val right = (this.right * workingScale).toInt().coerceIn(left + 1, bitmapWidth - 1)
    val bottom = (this.bottom * workingScale).toInt().coerceIn(top + 1, bitmapHeight - 1)
    return WorkingRegion(left, top, right, bottom)
}

private fun detectSpotsInManualRoi(
    detector: BrightSpotDetector,
    frame: PreparedFrame,
    tuning: AppTuningSettings,
    manualRoi: ManualRoiCoords,
    limitRegionWorking: WorkingRegion?,
    probabilityMatrix: ProbabilityMatrix32x24?
): DetectionResult {
    val workingWidth = frame.workingBitmap.width
    val workingHeight = frame.workingBitmap.height
    var roiLeft = (manualRoi.left * frame.workingScale).toInt().coerceIn(0, workingWidth - 1)
    var roiTop = (manualRoi.top * frame.workingScale).toInt().coerceIn(0, workingHeight - 1)
    var roiRight = (manualRoi.right * frame.workingScale).toInt()
        .coerceAtLeast(roiLeft + 1)
        .coerceAtMost(workingWidth - 1)
    var roiBottom = (manualRoi.bottom * frame.workingScale).toInt()
        .coerceAtLeast(roiTop + 1)
        .coerceAtMost(workingHeight - 1)

    val intersected = intersectWithLimitRegion(
        roiLeft,
        roiTop,
        roiRight,
        roiBottom,
        limitRegionWorking
    )
    roiLeft = intersected.left
    roiTop = intersected.top
    roiRight = intersected.right
    roiBottom = intersected.bottom

    val spots = detector.detectBrightSpots(
        bitmap = frame.workingBitmap,
        threshold = tuning.detectionThreshold,
        gridSize = tuning.gridSize,
        roiLeft = roiLeft,
        roiTop = roiTop,
        roiRight = roiRight,
        roiBottom = roiBottom,
        probabilityMatrix = probabilityMatrix
    )
    return DetectionResult(spots = spots, foundInManualRoi = spots.size == 3)
}

private fun detectSpotsWithAdaptiveRoi(
    detector: BrightSpotDetector,
    frame: PreparedFrame,
    tuning: AppTuningSettings,
    lastTriSpots: Triple<Offset, Offset, Offset>,
    lastTriSpotsImageSize: Pair<Int, Int>,
    limitRegionWorking: WorkingRegion?,
    probabilityMatrix: ProbabilityMatrix32x24?,
    onRoiInfo: ((RoiVisualizationInfo?) -> Unit)?
): DetectionResult {
    val initialRoiSeed = buildInitialRoiSeed(
        lastTriSpots = lastTriSpots,
        lastTriSpotsImageSize = lastTriSpotsImageSize,
        currentWidth = frame.rotatedBitmap.width,
        currentHeight = frame.rotatedBitmap.height,
        workingScale = frame.workingScale,
        roiInitialRadiusMultiplier = tuning.roiInitialRadiusMultiplier
    )

    var currentRadius = initialRoiSeed.third
    var lastTriRadius: Float? = null
    val recentAttempts = mutableListOf<Boolean>()
    val maxRecentAttempts = tuning.roiRecentAttempts.coerceAtLeast(1)
    val failureThreshold = tuning.roiFailureThreshold.coerceIn(1, maxRecentAttempts)
    val workingWidth = frame.workingBitmap.width
    val workingHeight = frame.workingBitmap.height
    val maxRadiusLimit = limitRegionWorking?.let {
        maxOf(it.right - it.left, it.bottom - it.top) * 0.5f
    }
    val maxAllowedRadius = min(
        max(workingWidth, workingHeight) * tuning.roiMaxRadiusRatio.coerceIn(0.05f, 1f),
        maxRadiusLimit ?: Float.MAX_VALUE
    )

    var foundSpots: List<BrightSpot>? = null
    var finalRoiInfo: RoiVisualizationInfo? = null
    var expansionCount = 0
    val maxExpansions = tuning.roiMaxExpansions.coerceAtLeast(1)

    while (foundSpots == null && currentRadius <= maxAllowedRadius && expansionCount < maxExpansions) {
        val roiBounds = intersectWithLimitRegion(
            left = (initialRoiSeed.first - currentRadius).toInt().coerceAtLeast(0),
            top = (initialRoiSeed.second - currentRadius).toInt().coerceAtLeast(0),
            right = (initialRoiSeed.first + currentRadius).toInt().coerceAtMost(workingWidth - 1),
            bottom = (initialRoiSeed.second + currentRadius).toInt().coerceAtMost(workingHeight - 1),
            limitRegionWorking = limitRegionWorking
        )

        if (roiBounds.right <= roiBounds.left || roiBounds.bottom <= roiBounds.top) {
            currentRadius *= tuning.roiExpandFactor.coerceAtLeast(1.01f)
            expansionCount++
            continue
        }

        val roiSpots = detector.detectBrightSpots(
            bitmap = frame.workingBitmap,
            threshold = tuning.detectionThreshold,
            gridSize = tuning.gridSize,
            roiLeft = roiBounds.left,
            roiTop = roiBounds.top,
            roiRight = roiBounds.right,
            roiBottom = roiBounds.bottom,
            probabilityMatrix = probabilityMatrix
        )

        val isStable = if (roiSpots.size == 3 && lastTriRadius != null) {
            val currRadius = calculateCircumradius(
                roiSpots[0].position,
                roiSpots[1].position,
                roiSpots[2].position
            )
            val changeRatio = abs(currRadius - lastTriRadius) / lastTriRadius
            changeRatio <= tuning.roiStabilityTolerance.coerceAtLeast(0f)
        } else {
            roiSpots.size == 3
        }

        if (roiSpots.size == 3 && isStable) {
            foundSpots = roiSpots
            finalRoiInfo = RoiVisualizationInfo(
                left = roiBounds.left / frame.workingScale,
                top = roiBounds.top / frame.workingScale,
                right = roiBounds.right / frame.workingScale,
                bottom = roiBounds.bottom / frame.workingScale,
                expansionCount = expansionCount,
                found = true
            )
            lastTriRadius = calculateCircumradius(
                roiSpots[0].position,
                roiSpots[1].position,
                roiSpots[2].position
            )
            recentAttempts.add(true)
        } else {
            currentRadius *= tuning.roiExpandFactor.coerceAtLeast(1.01f)
            expansionCount++
            recentAttempts.add(false)
        }

        if (recentAttempts.size > maxRecentAttempts) {
            recentAttempts.removeAt(0)
        }
        if (recentAttempts.count { !it } >= failureThreshold) {
            foundSpots = null
            onRoiInfo?.invoke(null)
            break
        }
    }

    return if (foundSpots == null) {
        onRoiInfo?.invoke(null)
        detectSpotsGlobally(
            detector = detector,
            bitmap = frame.workingBitmap,
            threshold = tuning.detectionThreshold,
            gridSize = tuning.gridSize,
            limitRegionWorking = limitRegionWorking,
            probabilityMatrix = probabilityMatrix
        )
    } else {
        finalRoiInfo?.let { info -> onRoiInfo?.invoke(info) }
        DetectionResult(foundSpots, foundInManualRoi = false)
    }
}

private fun buildInitialRoiSeed(
    lastTriSpots: Triple<Offset, Offset, Offset>,
    lastTriSpotsImageSize: Pair<Int, Int>,
    currentWidth: Int,
    currentHeight: Int,
    workingScale: Float,
    roiInitialRadiusMultiplier: Float
): Triple<Float, Float, Float> {
    val (lastW, lastH) = lastTriSpotsImageSize
    val scaleX = currentWidth.toFloat() / lastW
    val scaleY = currentHeight.toFloat() / lastH
    val p1 = Offset(lastTriSpots.first.x * scaleX, lastTriSpots.first.y * scaleY)
    val p2 = Offset(lastTriSpots.second.x * scaleX, lastTriSpots.second.y * scaleY)
    val p3 = Offset(lastTriSpots.third.x * scaleX, lastTriSpots.third.y * scaleY)
    val centerX = (p1.x + p2.x + p3.x) / 3f
    val centerY = (p1.y + p2.y + p3.y) / 3f
    val radius = calculateCircumradius(p1, p2, p3) * roiInitialRadiusMultiplier
    return Triple(centerX * workingScale, centerY * workingScale, radius * workingScale)
}

private fun detectSpotsGlobally(
    detector: BrightSpotDetector,
    bitmap: Bitmap,
    threshold: Float,
    gridSize: Int,
    limitRegionWorking: WorkingRegion?,
    probabilityMatrix: ProbabilityMatrix32x24?
): DetectionResult {
    val spots = limitRegionWorking?.let {
        detector.detectBrightSpots(
            bitmap = bitmap,
            threshold = threshold,
            gridSize = gridSize,
            roiLeft = it.left,
            roiTop = it.top,
            roiRight = it.right,
            roiBottom = it.bottom,
            probabilityMatrix = probabilityMatrix
        )
    } ?: detector.detectBrightSpots(
        bitmap = bitmap,
        threshold = threshold,
        gridSize = gridSize,
        probabilityMatrix = probabilityMatrix
    )
    return DetectionResult(spots, foundInManualRoi = false)
}

private fun intersectWithLimitRegion(
    left: Int,
    top: Int,
    right: Int,
    bottom: Int,
    limitRegionWorking: WorkingRegion?
): WorkingRegion {
    if (limitRegionWorking == null) {
        return WorkingRegion(left, top, right, bottom)
    }
    val intersectedLeft = maxOf(left, limitRegionWorking.left)
    val intersectedTop = maxOf(top, limitRegionWorking.top)
    val intersectedRight = minOf(right, limitRegionWorking.right)
    val intersectedBottom = minOf(bottom, limitRegionWorking.bottom)
    return if (intersectedRight <= intersectedLeft || intersectedBottom <= intersectedTop) {
        limitRegionWorking
    } else {
        WorkingRegion(intersectedLeft, intersectedTop, intersectedRight, intersectedBottom)
    }
}

private fun mapSpotsBackToOriginal(
    spots: List<BrightSpot>,
    scaleDown: Float,
    limitRegion: ManualRoiCoords?
): List<BrightSpot> {
    val invScale = if (scaleDown < 1f) 1f / scaleDown else 1f
    val mappedSpots = spots.map { spot ->
        spot.copy(position = Offset(spot.position.x * invScale, spot.position.y * invScale))
    }
    return limitRegion?.let { region ->
        mappedSpots.filter { spot ->
            spot.position.x in region.left..region.right &&
                spot.position.y in region.top..region.bottom
        }
    } ?: mappedSpots
}

private fun recyclePreparedFrame(originalBitmap: Bitmap, frame: PreparedFrame) {
    if (frame.rotatedBitmap != originalBitmap) {
        frame.rotatedBitmap.recycle()
    }
    if (frame.workingBitmap != frame.rotatedBitmap) {
        frame.workingBitmap.recycle()
    }
    originalBitmap.recycle()
}

private fun refineSpotsOnOriginal(
    bitmap: Bitmap,
    spots: List<BrightSpot>,
    tuning: AppTuningSettings
): List<BrightSpot> {
    if (spots.isEmpty()) return spots
    val width = bitmap.width
    val height = bitmap.height
    val step = max(1, max(width / tuning.gridSize, height / tuning.gridSize))
    val radius = (step * tuning.refineRadiusMultiplier).toInt()
        .coerceAtLeast(tuning.refineMinRadiusPx)
        .coerceAtMost(min(width, height) / 6)

    fun weightFor(pixel: Int, prefer: LedColor, backgroundDarkness: Float = 0.5f): Float {
        val r = ((pixel shr 16) and 0xFF).toFloat()
        val g = ((pixel shr 8) and 0xFF).toFloat()
        val b = (pixel and 0xFF).toFloat()

        val colorScore = when (prefer) {
            LedColor.RED -> (r - (g + b) / 2f).coerceAtLeast(0f)
            LedColor.GREEN -> (g - (r + b) / 2f).coerceAtLeast(0f)
            LedColor.BLUE -> (b - (r + g) / 2f).coerceAtLeast(0f)
            else -> (r + g + b).coerceAtLeast(0f)
        }

        val v = (r + g + b) / (255f * 3f)
        val backgroundScore = (1f - v).pow(2) * backgroundDarkness
        return colorScore * (1f + backgroundScore)
    }

    fun refineOne(cx: Float, cy: Float, prefer: LedColor): Offset {
        var centerX = cx
        var centerY = cy
        repeat(tuning.refineIterations.coerceAtLeast(1)) {
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
            val radF = radius.toFloat()
            val sigmaFactor = tuning.refineGaussianSigmaFactor.coerceAtLeast(0.1f)
            val invTwoSigma2 = 1f / (2f * (radF * sigmaFactor) * (radF * sigmaFactor))
            for (yy in 0 until hRect) {
                val row = yy * wRect
                val yAbs = (y0 + yy).toFloat()
                for (xx in 0 until wRect) {
                    val xAbs = (x0 + xx).toFloat()
                    val w = weightFor(
                        buf[row + xx],
                        prefer,
                        backgroundDarkness = tuning.refineBackgroundDarkness
                    )
                    if (w <= 0f) continue
                    val dx = xAbs - centerX
                    val dy = yAbs - centerY
                    val radial = kotlin.math.exp(-(dx * dx + dy * dy) * invTwoSigma2)
                    val ww = w * radial
                    sumW += ww
                    sumX += xAbs * ww
                    sumY += yAbs * ww
                }
            }
            if (sumW > 1e-6f) {
                centerX = sumX / sumW
                centerY = sumY / sumW
            }
        }
        return Offset(centerX, centerY)
    }

    return spots.map { spot ->
        val prefer = when {
            spot.color != LedColor.UNKNOWN -> spot.color
            spot.redIntensity >= spot.greenIntensity && spot.redIntensity >= spot.blueIntensity -> LedColor.RED
            spot.greenIntensity >= spot.redIntensity && spot.greenIntensity >= spot.blueIntensity -> LedColor.GREEN
            else -> LedColor.BLUE
        }
        val refined = refineOne(spot.position.x, spot.position.y, prefer)
        spot.copy(position = refined)
    }
}

private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
    if (degrees == 0f) return bitmap
    val matrix = Matrix().apply { postRotate(degrees) }
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
