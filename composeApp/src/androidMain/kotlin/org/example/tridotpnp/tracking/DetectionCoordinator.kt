package org.example.tridotpnp.tracking

import org.example.tridotpnp.config.AppTuningSettings
import org.example.tridotpnp.vision.BrightSpot
import org.example.tridotpnp.vision.BrightSpotDetector
import org.example.tridotpnp.vision.ColorCandidatePools
import org.example.tridotpnp.vision.LedColor
data class FrameTrackingSnapshot(
    val trackedNodes: List<TrackedNodeState>,
    val visionObservation: VisionObservation?,
    val pnpResult: PnPDistanceCalculator.PnPResult?
)

/**
 * Heavy tracking / PnP work intended for the camera analyzer thread.
 * Must keep the same math as the previous main-thread path.
 */
fun computeFrameTracking(
    nodeTracker: NodeTrackerManager,
    pnpCalculator: PnPDistanceCalculator,
    spots: List<BrightSpot>,
    imageSize: Pair<Int, Int>,
    tuning: AppTuningSettings,
    candidates: ColorCandidatePools = ColorCandidatePools()
): FrameTrackingSnapshot {
    val extras = if (tuning.fusion.enableMultiNodeTracking) {
        extractExtraTriangles(
            candidates = candidates,
            primarySpots = spots,
            maxNodes = tuning.fusion.maxTrackedNodes
        )
    } else {
        emptyList()
    }
    val nodes = nodeTracker.updateFromPrimaryTriple(
        spots = spots,
        imageSize = imageSize,
        tuning = tuning,
        extraTriples = extras
    )
    val primary = selectPrimaryObservation(nodes, spots)
        ?: buildVisionObservation(
            spots = spots,
            imageSize = imageSize,
            tuning = tuning,
            pnpCalculator = pnpCalculator
        )
    val pnp = if (primary == null || !primary.accepted) {
        if (!tuning.fusion.enableVisionConfidenceGate) {
            calculateRgbPnpResult(
                spots = spots,
                imageSize = imageSize,
                tuning = tuning,
                pnpCalculator = pnpCalculator
            )
        } else {
            null
        }
    } else {
        primary.pnp
    }
    return FrameTrackingSnapshot(
        trackedNodes = nodes,
        visionObservation = primary,
        pnpResult = pnp
    )
}

internal fun BrightSpotDetector.applyTuning(tuning: AppTuningSettings) {
    minPixelBrightness = tuning.detect.minPixelBrightness
    minTotalBrightness = tuning.detect.minTotalBrightness
    dynamicThresholdMin = tuning.detect.dynamicThresholdMin
    dynamicThresholdRatio = tuning.detect.dynamicThresholdRatio
    minPixelCount = tuning.detect.minPixelCount
    minRefineRadius = tuning.detect.detectorMinRefineRadius
    maxBrightPixelRatio = tuning.detect.maxBrightPixelRatio
    maxBrightPixelCountMin = tuning.detect.maxBrightPixelCountMin
    hsvMinSaturation = tuning.detect.hsvMinSaturation
    hsvMinValue = tuning.detect.hsvMinValue
    hsvHueWeight = tuning.detect.hsvHueWeight
    hsvSatWeight = tuning.detect.hsvSatWeight
    hsvValWeight = tuning.detect.hsvValWeight
    hsvSharpness = tuning.detect.hsvSharpness
    triShapePenaltyWeight = tuning.detect.triShapePenaltyWeight
    triGeometryPenaltyWeight = tuning.detect.triGeometryPenaltyWeight
    triPurityBoostWeight = tuning.detect.triPurityBoostWeight
    triCenterBlackWeight = tuning.detect.triCenterBlackWeight
    triColorScoreWeight = tuning.detect.triColorScoreWeight
    topNCandidatesPerColor = tuning.detect.topNCandidatesPerColor
    minPairDistanceMultiplier = tuning.detect.minPairDistanceMultiplier
    maxPairDistanceMultiplier = tuning.detect.maxPairDistanceMultiplier
    darkBackgroundBoost = tuning.detect.darkBackgroundBoost
    probabilityCandidateExponent = tuning.detect.probabilityCandidateExponent
    probabilityGroupExponent = tuning.detect.probabilityGroupExponent
    probabilityWeightFloor = tuning.detect.probabilityWeightFloor
    calibrationSearchStep = tuning.detect.calibrationSearchStep
    calibrationSearchRadius = tuning.detect.calibrationSearchRadius
    sampleTopPercent = tuning.detect.sampleTopPercent
    trackSwitchMargin = tuning.trackLock.trackSwitchMargin
    trackLockMaxDistPx = tuning.trackLock.trackLockMaxDistPx
    trackLockConfidenceBoost = tuning.trackLock.trackLockConfidenceBoost
    gmmColorModel.enabled = tuning.detect.enableGmmSoftSegment
    gmmColorModel.blend = tuning.detect.gmmBlend
}

internal fun PnPDistanceCalculator.applyTuning(tuning: AppTuningSettings) {
    estimatedFovDegrees = tuning.pnp.estimatedFovDegrees
    depthInitFallbackMm = tuning.pnp.pnpDepthInitFallbackMm
    solverMaxIterations = tuning.pnp.pnpSolverMaxIterations
    solverConvergenceErrorMm = tuning.pnp.pnpSolverConvergenceErrorMm
    solverLearningRate = tuning.pnp.pnpSolverLearningRate
    solverMinDepthMm = tuning.pnp.pnpSolverMinDepthMm
}

internal fun NodeTrackerManager.applyTuning(tuning: AppTuningSettings) {
    configure(tuning)
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
        triangleEdgeLength = tuning.pnp.knownTriangleEdgeLengthMm,
        focalLength = null,
        imageWidth = size.first,
        imageHeight = size.second
    )
}

internal fun buildVisionObservation(
    spots: List<BrightSpot>,
    imageSize: Pair<Int, Int>?,
    tuning: AppTuningSettings,
    pnpCalculator: PnPDistanceCalculator,
    identityConfirmCount: Int = 0,
    nodeId: Int? = null
): VisionObservation? {
    val pnp = calculateRgbPnpResult(spots, imageSize, tuning, pnpCalculator) ?: return null
    if (!tuning.fusion.enableVisionConfidenceGate) {
        return VisionObservation(
            nodeId = nodeId,
            spots = spots,
            pnp = pnp,
            confidence = 1f,
            geometryScore = 1f,
            reprojectionErrorMm = 0f,
            identityConfirmCount = identityConfirmCount,
            timestampMs = System.currentTimeMillis(),
            accepted = true
        )
    }
    return VisionConfidenceGate.evaluate(
        spots = spots,
        pnp = pnp,
        edgeLengthMm = tuning.pnp.knownTriangleEdgeLengthMm,
        identityConfirmCount = identityConfirmCount,
        nodeId = nodeId,
        config = VisionGateConfig(
            minConfidence = tuning.fusion.visionMinConfidence,
            maxReprojectionErrorMm = tuning.fusion.visionMaxReprojectionErrorMm
        )
    )
}

internal fun selectPrimaryObservation(
    nodes: List<TrackedNodeState>,
    primarySpots: List<BrightSpot>
): VisionObservation? {
    if (nodes.isEmpty()) return null
    if (primarySpots.size == 3) {
        val cx = (primarySpots[0].position.x + primarySpots[1].position.x + primarySpots[2].position.x) / 3f
        val cy = (primarySpots[0].position.y + primarySpots[1].position.y + primarySpots[2].position.y) / 3f
        val nearest = nodes.minByOrNull { n ->
            val p = n.observation?.pnp?.centerCoords
            val sx = n.spots.getOrNull(0)?.position?.x
            val sy = n.spots.getOrNull(0)?.position?.y
            // Prefer image-space center of tracked spots when available.
            val nx = if (n.spots.size == 3) {
                (n.spots[0].position.x + n.spots[1].position.x + n.spots[2].position.x) / 3f
            } else {
                sx ?: (p?.x ?: Float.MAX_VALUE)
            }
            val ny = if (n.spots.size == 3) {
                (n.spots[0].position.y + n.spots[1].position.y + n.spots[2].position.y) / 3f
            } else {
                sy ?: (p?.y ?: Float.MAX_VALUE)
            }
            val dx = nx - cx
            val dy = ny - cy
            dx * dx + dy * dy
        }
        nearest?.observation?.let { return it }
    }
    return nodes.maxByOrNull { it.observation?.confidence ?: -1f }?.observation
}

internal fun extractExtraTriangles(
    candidates: ColorCandidatePools,
    primarySpots: List<BrightSpot>,
    maxNodes: Int
): List<DetectedTriangle> {
    if (maxNodes <= 1 || candidates.isEmpty()) return emptyList()
    val all = MultiTriangleExtractor.extract(
        reds = candidates.reds,
        greens = candidates.greens,
        blues = candidates.blues,
        maxNodes = maxNodes
    )
    if (primarySpots.size != 3) return all
    val primaryCenter = androidx.compose.ui.geometry.Offset(
        (primarySpots[0].position.x + primarySpots[1].position.x + primarySpots[2].position.x) / 3f,
        (primarySpots[0].position.y + primarySpots[1].position.y + primarySpots[2].position.y) / 3f
    )
    return all.filter { tri ->
        val dx = tri.center.x - primaryCenter.x
        val dy = tri.center.y - primaryCenter.y
        dx * dx + dy * dy > 40f * 40f
    }
}
