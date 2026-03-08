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

internal fun PnPDistanceCalculator.applyTuning(tuning: AppTuningSettings) {
    estimatedFovDegrees = tuning.estimatedFovDegrees
    depthInitFallbackMm = tuning.pnpDepthInitFallbackMm
    solverMaxIterations = tuning.pnpSolverMaxIterations
    solverConvergenceErrorMm = tuning.pnpSolverConvergenceErrorMm
    solverLearningRate = tuning.pnpSolverLearningRate
    solverMinDepthMm = tuning.pnpSolverMinDepthMm
}

internal fun calculateRgbPnpResult(
    spots: List<BrightSpot>,
    imageSize: Pair<Int, Int>?,
    tuning: AppTuningSettings,
    pnpCalculator: PnPDistanceCalculator
): PnPDistanceCalculator.PnPResult? {
    val size = imageSize ?: return null
    if (spots.size != 3) return null

    val red = spots.firstOrNull { it.color == LedColor.RED } ?: return null
    val green = spots.firstOrNull { it.color == LedColor.GREEN } ?: return null
    val blue = spots.firstOrNull { it.color == LedColor.BLUE } ?: return null

    return pnpCalculator.calculate3PointPnP(
        redPoint = red.position,
        greenPoint = green.position,
        bluePoint = blue.position,
        triangleEdgeLength = tuning.knownTriangleEdgeLengthMm,
        focalLength = null,
        imageWidth = size.first,
        imageHeight = size.second
    )
}
