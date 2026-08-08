package org.example.tridotpnp

data class AppTuningSettings(
    val gridSize: Int = 256,
    val exposureCompensation: Int = 0,
    val knownTriangleEdgeLengthMm: Float = 18f,
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
    val trackLockConfidenceBoost: Float = 2.0f,
    val refineIterations: Int = 2,
    val refineRadiusMultiplier: Float = 2f,
    val refineMinRadiusPx: Int = 6,
    val refineBackgroundDarkness: Float = 0.6f,
    val refineGaussianSigmaFactor: Float = 0.6f,
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
    val estimatedFovDegrees: Float = 60f,
    val pnpDepthInitFallbackMm: Float = 1000f,
    val pnpSolverMaxIterations: Int = 50,
    val pnpSolverConvergenceErrorMm: Float = 0.1f,
    val pnpSolverLearningRate: Float = 0.5f,
    val pnpSolverMinDepthMm: Float = 100f,
    // --- Vision-inertial / multi-node extensions ---
    val enableGmmSoftSegment: Boolean = true,
    val gmmBlend: Float = 0.65f,
    val enableVisionConfidenceGate: Boolean = true,
    val visionMinConfidence: Float = 0.45f,
    val visionMaxReprojectionErrorMm: Float = 25f,
    val enableIdentityDecode: Boolean = true,
    val identityConfirmNeeded: Int = 3,
    val enableMultiNodeTracking: Boolean = true,
    val maxTrackedNodes: Int = 6,
    val enableTopologyPrior: Boolean = true,
    val enableEskfFusion: Boolean = true,
    val enableIkPostProcess: Boolean = true,
    val enableBleImu: Boolean = false
)
