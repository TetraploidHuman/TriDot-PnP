package org.example.tridotpnp

internal fun BrightSpotDetector.applyTuning(tuning: AppTuningSettings) {
    minPixelBrightness = tuning.minPixelBrightness
    minTotalBrightness = tuning.minTotalBrightness
    dynamicThresholdMin = tuning.dynamicThresholdMin
    dynamicThresholdRatio = tuning.dynamicThresholdRatio
    minPixelCount = tuning.minPixelCount
    minRefineRadius = tuning.detectorMinRefineRadius
    maxBrightPixelRatio = tuning.maxBrightPixelRatio
    maxBrightPixelCountMin = tuning.maxBrightPixelCountMin
    hsvMinSaturation = tuning.hsvMinSaturation
    hsvMinValue = tuning.hsvMinValue
    hsvHueWeight = tuning.hsvHueWeight
    hsvSatWeight = tuning.hsvSatWeight
    hsvValWeight = tuning.hsvValWeight
    hsvSharpness = tuning.hsvSharpness
    triShapePenaltyWeight = tuning.triShapePenaltyWeight
    triGeometryPenaltyWeight = tuning.triGeometryPenaltyWeight
    triPurityBoostWeight = tuning.triPurityBoostWeight
    triCenterBlackWeight = tuning.triCenterBlackWeight
    triColorScoreWeight = tuning.triColorScoreWeight
    topNCandidatesPerColor = tuning.topNCandidatesPerColor
    minPairDistanceMultiplier = tuning.minPairDistanceMultiplier
    maxPairDistanceMultiplier = tuning.maxPairDistanceMultiplier
    darkBackgroundBoost = tuning.darkBackgroundBoost
    probabilityCandidateExponent = tuning.probabilityCandidateExponent
    probabilityGroupExponent = tuning.probabilityGroupExponent
    probabilityWeightFloor = tuning.probabilityWeightFloor
    calibrationSearchStep = tuning.calibrationSearchStep
    calibrationSearchRadius = tuning.calibrationSearchRadius
    sampleTopPercent = tuning.sampleTopPercent
}

/** 进程内共享的 PnP 时序跟踪器（一帧一次，跨帧保留状态）。 */
internal object AppTemporalTracker {
    val tracker = PnPTemporalTracker()
}

internal fun PnPTemporalTracker.applyTuning(tuning: AppTuningSettings) {
    alpha = tuning.pnpTemporalSmoothing
}

internal fun PnPDistanceCalculator.applyTuning(tuning: AppTuningSettings) {
    estimatedFovDegrees = tuning.estimatedFovDegrees
    depthInitFallbackMm = tuning.pnpDepthInitFallbackMm
    solverMaxIterations = tuning.pnpSolverMaxIterations
    solverConvergenceErrorMm = tuning.pnpSolverConvergenceErrorMm
    solverMinDepthMm = tuning.pnpSolverMinDepthMm
}

internal fun calculateRgbPnpResult(
    spots: List<BrightSpot>,
    imageSize: Pair<Int, Int>?,
    tuning: AppTuningSettings,
    pnpCalculator: PnPDistanceCalculator
): PnPDistanceCalculator.PnPResult? {
    val size = imageSize ?: return null
    if (spots.size !in 2..3) return null

    val red = spots.firstOrNull { it.color == LedColor.RED }
    val green = spots.firstOrNull { it.color == LedColor.GREEN }
    val blue = spots.firstOrNull { it.color == LedColor.BLUE }
    if (listOf(red, green, blue).count { it != null } != spots.size) return null

    val now = System.nanoTime() / 1_000_000_000.0

    if (tuning.temporalFilterEnabled) {
        // 三缺一：用上一帧位姿 + 速度外推这一帧的姿态，最多连续 maxCoastFrames 帧。
        // 这样检测漏掉一个点时 DISTANCE/AZIMUTH/6DOF 不会整帧消失；
        // 跟踪器还没锁定时直接返回 null，保持原来的行为。
        if (red == null || green == null || blue == null) {
            // 只做姿态外推、不把本帧观测写回滤波器（与 Python 侧 completer 的
            // 「状态只由全可见帧更新」一致）；连续丢点超过 maxCoastFrames 后返回 null。
            val completed = pnpCalculator.calculate3PointPnPCompleted(
                triangleEdgeLength = tuning.knownTriangleEdgeLengthMm,
                timestampSec = now,
                tracker = AppTemporalTracker.tracker
            )
            if (completed != null) return completed
            return null
        }

        val tracked = pnpCalculator.calculate3PointPnPTracked(
            redPoint = red.position,
            greenPoint = green.position,
            bluePoint = blue.position,
            triangleEdgeLength = tuning.knownTriangleEdgeLengthMm,
            imageWidth = size.first,
            imageHeight = size.second,
            timestampSec = now,
            tracker = AppTemporalTracker.tracker
        )
        if (tracked != null) return tracked
    }

    if (spots.size != 3) return null
    return pnpCalculator.calculate3PointPnP(
        redPoint = red!!.position,
        greenPoint = green!!.position,
        bluePoint = blue!!.position,
        triangleEdgeLength = tuning.knownTriangleEdgeLengthMm,
        focalLength = null,
        imageWidth = size.first,
        imageHeight = size.second
    )
}
