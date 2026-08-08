package org.example.tridotpnp.config

/** Camera capture / working-resolution / ROI sampling. */
data class CameraPipelineTuning(
    val gridSize: Int = 256,
    val exposureCompensation: Int = 0,
    val enableRoiOptimization: Boolean = true,
    val detectionThreshold: Float = 80f,
    val workingMaxDimension: Int = 1280,
    /** ROI 内按「目标步长 + 格子预算」自适应密度；关闭则 ROI 与全图同用 gridSize。 */
    val enableRoiBudgetGrid: Boolean = true,
    /** ROI 内目标采样步长（工作图像素）。越小越密；大 ROI 会被预算自动放宽。 */
    val roiIdealStepPx: Int = 4,
    /** 单次 ROI 检测最多扫描的格子数（约），防止大圈时算力爆炸。 */
    val roiMaxCells: Int = 4096,
    val roiInitialRadiusMultiplier: Float = 2f,
    val roiExpandFactor: Float = 1.3f,
    val roiMaxRadiusRatio: Float = 0.4f,
    val roiMaxExpansions: Int = 10,
    val roiStabilityTolerance: Float = 0.3f,
    val roiRecentAttempts: Int = 7,
    val roiFailureThreshold: Int = 4,
    val refineIterations: Int = 2,
    val refineRadiusMultiplier: Float = 2f,
    val refineMinRadiusPx: Int = 6,
    val refineBackgroundDarkness: Float = 0.6f,
    val refineGaussianSigmaFactor: Float = 0.6f
)

/** Bright-spot detector scoring / HSV / triple selection. */
data class DetectTuning(
    val minPixelBrightness: Float = 10f,
    val minTotalBrightness: Float = 20f,
    val dynamicThresholdMin: Float = 30f,
    val dynamicThresholdRatio: Float = 0.7f,
    val minPixelCount: Int = 0,
    val detectorMinRefineRadius: Int = 2,
    val maxBrightPixelRatio: Float = 0.18f,
    val maxBrightPixelCountMin: Int = 12,
    val hsvMinSaturation: Float = 0.15f,
    val hsvMinValue: Float = 0.10f,
    val hsvHueWeight: Float = 1.0f,
    val hsvSatWeight: Float = 0.6f,
    val hsvValWeight: Float = 0.3f,
    val hsvSharpness: Float = 8.0f,
    val triShapePenaltyWeight: Float = 0.2f,
    val triGeometryPenaltyWeight: Float = 0.6f,
    val triPurityBoostWeight: Float = 0.25f,
    val triCenterBlackWeight: Float = 0.7f,
    val triColorScoreWeight: Float = 1.0f,
    val topNCandidatesPerColor: Int = 16,
    val minPairDistanceMultiplier: Float = 2f,
    val maxPairDistanceMultiplier: Float = 20f,
    val darkBackgroundBoost: Float = 0.5f,
    val probabilityCandidateExponent: Float = 1.0f,
    val probabilityGroupExponent: Float = 1.0f,
    val probabilityWeightFloor: Float = 0.01f,
    val calibrationSearchStep: Int = 10,
    val calibrationSearchRadius: Int = 15,
    val sampleTopPercent: Float = 0.2f,
    val enableGmmSoftSegment: Boolean = true,
    val gmmBlend: Float = 0.65f
)

/**
 * Multi-frame RGB lock stickiness (detector hysteresis + analyzer confidence).
 */
data class TrackLockTuning(
    /**
     * 已锁定 RGB 三点的基础粘滞裕度：新三角形得分需高出该比例才允许切换。
     * 实际裕度还会随锁定置信度放大。0 = 关闭粘滞。
     */
    val trackSwitchMargin: Float = 0.20f,
    /** 判定仍是「同一组锁定点」的单色最大像素距离（工作图坐标）。 */
    val trackLockMaxDistPx: Float = 72f,
    /** 每连续稳定跟踪一帧，锁定置信度增加多少（0..1）。 */
    val trackLockConfidenceRise: Float = 0.08f,
    /** 丢检时每帧锁定置信度衰减多少。 */
    val trackLockConfidenceDecay: Float = 0.06f,
    /**
     * 置信度对粘滞裕度的放大：有效裕度 = 基础 × (1 + boost × confidence)。
     * 例如 boost=2、confidence=1 时，裕度为基础的 3 倍。
     */
    val trackLockConfidenceBoost: Float = 2.0f
)

data class PnPTuning(
    /** 已知 RGB 三角形边长（mm），直接影响距离解算。 */
    val knownTriangleEdgeLengthMm: Float = 18f,
    val estimatedFovDegrees: Float = 60f,
    val pnpDepthInitFallbackMm: Float = 1000f,
    val pnpSolverMaxIterations: Int = 50,
    val pnpSolverConvergenceErrorMm: Float = 0.1f,
    val pnpSolverLearningRate: Float = 0.5f,
    val pnpSolverMinDepthMm: Float = 100f
)

/** Multi-node / identity / ESKF / IK fusion gates. */
data class FusionTuning(
    val enableVisionConfidenceGate: Boolean = true,
    val visionMinConfidence: Float = 0.45f,
    val visionMaxReprojectionErrorMm: Float = 25f,
    val enableIdentityDecode: Boolean = true,
    val identityConfirmNeeded: Int = 3,
    val enableMultiNodeTracking: Boolean = true,
    val maxTrackedNodes: Int = 6,
    val enableTopologyPrior: Boolean = true,
    val enableEskfFusion: Boolean = true,
    val enableIkPostProcess: Boolean = true
)

data class BleTuning(
    val enableBleImu: Boolean = false
)

/**
 * App-wide tuning root. Domain code should prefer the nested groups
 * ([camera], [detect], [trackLock], [pnp], [fusion], [ble]).
 */
data class AppTuningSettings(
    val camera: CameraPipelineTuning = CameraPipelineTuning(),
    val detect: DetectTuning = DetectTuning(),
    val trackLock: TrackLockTuning = TrackLockTuning(),
    val pnp: PnPTuning = PnPTuning(),
    val fusion: FusionTuning = FusionTuning(),
    val ble: BleTuning = BleTuning()
) {
    // --- Flat read aliases (UI / call sites that only read) ---
    val gridSize get() = camera.gridSize
    val exposureCompensation get() = camera.exposureCompensation
    val enableRoiOptimization get() = camera.enableRoiOptimization
    val detectionThreshold get() = camera.detectionThreshold
    val workingMaxDimension get() = camera.workingMaxDimension
    val enableRoiBudgetGrid get() = camera.enableRoiBudgetGrid
    val roiIdealStepPx get() = camera.roiIdealStepPx
    val roiMaxCells get() = camera.roiMaxCells
    val roiInitialRadiusMultiplier get() = camera.roiInitialRadiusMultiplier
    val roiExpandFactor get() = camera.roiExpandFactor
    val roiMaxRadiusRatio get() = camera.roiMaxRadiusRatio
    val roiMaxExpansions get() = camera.roiMaxExpansions
    val roiStabilityTolerance get() = camera.roiStabilityTolerance
    val roiRecentAttempts get() = camera.roiRecentAttempts
    val roiFailureThreshold get() = camera.roiFailureThreshold
    val refineIterations get() = camera.refineIterations
    val refineRadiusMultiplier get() = camera.refineRadiusMultiplier
    val refineMinRadiusPx get() = camera.refineMinRadiusPx
    val refineBackgroundDarkness get() = camera.refineBackgroundDarkness
    val refineGaussianSigmaFactor get() = camera.refineGaussianSigmaFactor

    val minPixelBrightness get() = detect.minPixelBrightness
    val minTotalBrightness get() = detect.minTotalBrightness
    val dynamicThresholdMin get() = detect.dynamicThresholdMin
    val dynamicThresholdRatio get() = detect.dynamicThresholdRatio
    val minPixelCount get() = detect.minPixelCount
    val detectorMinRefineRadius get() = detect.detectorMinRefineRadius
    val maxBrightPixelRatio get() = detect.maxBrightPixelRatio
    val maxBrightPixelCountMin get() = detect.maxBrightPixelCountMin
    val hsvMinSaturation get() = detect.hsvMinSaturation
    val hsvMinValue get() = detect.hsvMinValue
    val hsvHueWeight get() = detect.hsvHueWeight
    val hsvSatWeight get() = detect.hsvSatWeight
    val hsvValWeight get() = detect.hsvValWeight
    val hsvSharpness get() = detect.hsvSharpness
    val triShapePenaltyWeight get() = detect.triShapePenaltyWeight
    val triGeometryPenaltyWeight get() = detect.triGeometryPenaltyWeight
    val triPurityBoostWeight get() = detect.triPurityBoostWeight
    val triCenterBlackWeight get() = detect.triCenterBlackWeight
    val triColorScoreWeight get() = detect.triColorScoreWeight
    val topNCandidatesPerColor get() = detect.topNCandidatesPerColor
    val minPairDistanceMultiplier get() = detect.minPairDistanceMultiplier
    val maxPairDistanceMultiplier get() = detect.maxPairDistanceMultiplier
    val darkBackgroundBoost get() = detect.darkBackgroundBoost
    val probabilityCandidateExponent get() = detect.probabilityCandidateExponent
    val probabilityGroupExponent get() = detect.probabilityGroupExponent
    val probabilityWeightFloor get() = detect.probabilityWeightFloor
    val calibrationSearchStep get() = detect.calibrationSearchStep
    val calibrationSearchRadius get() = detect.calibrationSearchRadius
    val sampleTopPercent get() = detect.sampleTopPercent
    val enableGmmSoftSegment get() = detect.enableGmmSoftSegment
    val gmmBlend get() = detect.gmmBlend

    val trackSwitchMargin get() = trackLock.trackSwitchMargin
    val trackLockMaxDistPx get() = trackLock.trackLockMaxDistPx
    val trackLockConfidenceRise get() = trackLock.trackLockConfidenceRise
    val trackLockConfidenceDecay get() = trackLock.trackLockConfidenceDecay
    val trackLockConfidenceBoost get() = trackLock.trackLockConfidenceBoost

    val knownTriangleEdgeLengthMm get() = pnp.knownTriangleEdgeLengthMm
    val estimatedFovDegrees get() = pnp.estimatedFovDegrees
    val pnpDepthInitFallbackMm get() = pnp.pnpDepthInitFallbackMm
    val pnpSolverMaxIterations get() = pnp.pnpSolverMaxIterations
    val pnpSolverConvergenceErrorMm get() = pnp.pnpSolverConvergenceErrorMm
    val pnpSolverLearningRate get() = pnp.pnpSolverLearningRate
    val pnpSolverMinDepthMm get() = pnp.pnpSolverMinDepthMm

    val enableVisionConfidenceGate get() = fusion.enableVisionConfidenceGate
    val visionMinConfidence get() = fusion.visionMinConfidence
    val visionMaxReprojectionErrorMm get() = fusion.visionMaxReprojectionErrorMm
    val enableIdentityDecode get() = fusion.enableIdentityDecode
    val identityConfirmNeeded get() = fusion.identityConfirmNeeded
    val enableMultiNodeTracking get() = fusion.enableMultiNodeTracking
    val maxTrackedNodes get() = fusion.maxTrackedNodes
    val enableTopologyPrior get() = fusion.enableTopologyPrior
    val enableEskfFusion get() = fusion.enableEskfFusion
    val enableIkPostProcess get() = fusion.enableIkPostProcess

    val enableBleImu get() = ble.enableBleImu
}

inline fun AppTuningSettings.updateCamera(
    block: CameraPipelineTuning.() -> CameraPipelineTuning
): AppTuningSettings = copy(camera = camera.block())

inline fun AppTuningSettings.updateDetect(
    block: DetectTuning.() -> DetectTuning
): AppTuningSettings = copy(detect = detect.block())

inline fun AppTuningSettings.updateTrackLock(
    block: TrackLockTuning.() -> TrackLockTuning
): AppTuningSettings = copy(trackLock = trackLock.block())

inline fun AppTuningSettings.updatePnp(
    block: PnPTuning.() -> PnPTuning
): AppTuningSettings = copy(pnp = pnp.block())

inline fun AppTuningSettings.updateFusion(
    block: FusionTuning.() -> FusionTuning
): AppTuningSettings = copy(fusion = fusion.block())

inline fun AppTuningSettings.updateBle(
    block: BleTuning.() -> BleTuning
): AppTuningSettings = copy(ble = ble.block())
