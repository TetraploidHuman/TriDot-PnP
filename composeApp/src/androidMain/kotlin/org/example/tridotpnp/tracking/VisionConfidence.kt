package org.example.tridotpnp.tracking


import org.example.tridotpnp.config.AppTuningSettings
import org.example.tridotpnp.vision.BrightSpot
import org.example.tridotpnp.vision.BrightSpotDetector
import org.example.tridotpnp.vision.ColorCalibration
import org.example.tridotpnp.vision.LedColor
import org.example.tridotpnp.vision.GmmColorModel
import androidx.compose.ui.geometry.Offset
import kotlin.math.abs
import kotlin.math.sqrt

data class VisionObservation(
    val nodeId: Int?,
    val spots: List<BrightSpot>,
    val pnp: PnPDistanceCalculator.PnPResult,
    val confidence: Float,
    val geometryScore: Float,
    val reprojectionErrorMm: Float,
    val identityConfirmCount: Int,
    val timestampMs: Long,
    val accepted: Boolean
)

data class VisionGateConfig(
    val minConfidence: Float = 0.45f,
    val maxReprojectionErrorMm: Float = 25f,
    val minGeometryScore: Float = 0.35f,
    val geometryWeight: Float = 0.40f,
    val reprojectionWeight: Float = 0.35f,
    val purityWeight: Float = 0.20f,
    val identityWeight: Float = 0.05f
)

object VisionConfidenceGate {

    fun evaluate(
        spots: List<BrightSpot>,
        pnp: PnPDistanceCalculator.PnPResult,
        edgeLengthMm: Float,
        identityConfirmCount: Int = 0,
        nodeId: Int? = null,
        config: VisionGateConfig = VisionGateConfig(),
        timestampMs: Long = System.currentTimeMillis()
    ): VisionObservation {
        val geometry = geometryScore(spots)
        val purity = purityScore(spots)
        val reproj = estimateEdgeResidualMm(pnp, edgeLengthMm)
        val reprojScore = (1f - (reproj / config.maxReprojectionErrorMm.coerceAtLeast(1f)))
            .coerceIn(0f, 1f)
        // Identity is optional; missing decode should not tank confidence.
        val identityScore = if (identityConfirmCount <= 0) {
            1f
        } else {
            (identityConfirmCount / 3f).coerceIn(0f, 1f)
        }

        val confidence = (
            config.geometryWeight * geometry +
                config.reprojectionWeight * reprojScore +
                config.purityWeight * purity +
                config.identityWeight * identityScore
            ).coerceIn(0f, 1f)

        val accepted = confidence >= config.minConfidence &&
            geometry >= config.minGeometryScore &&
            reproj <= config.maxReprojectionErrorMm

        return VisionObservation(
            nodeId = nodeId,
            spots = spots,
            pnp = pnp,
            confidence = confidence,
            geometryScore = geometry,
            reprojectionErrorMm = reproj,
            identityConfirmCount = identityConfirmCount,
            timestampMs = timestampMs,
            accepted = accepted
        )
    }

    private fun geometryScore(spots: List<BrightSpot>): Float {
        if (spots.size != 3) return 0f
        val a = spots[0].position
        val b = spots[1].position
        val c = spots[2].position
        val d1 = dist(a, b)
        val d2 = dist(b, c)
        val d3 = dist(c, a)
        val mean = (d1 + d2 + d3) / 3f
        if (mean < 1e-3f) return 0f
        val shapePenalty = (abs(d1 - mean) + abs(d2 - mean) + abs(d3 - mean)) / mean
        return (1f - shapePenalty / 0.45f).coerceIn(0f, 1f)
    }

    private fun purityScore(spots: List<BrightSpot>): Float {
        if (spots.isEmpty()) return 0f
        val mean = spots.map { spot ->
            val total = spot.redIntensity + spot.greenIntensity + spot.blueIntensity + 1e-3f
            val own = when (spot.color) {
                LedColor.RED -> spot.redIntensity
                LedColor.GREEN -> spot.greenIntensity
                LedColor.BLUE -> spot.blueIntensity
                else -> spot.brightness
            }
            (own / total).coerceIn(0f, 1f)
        }.average().toFloat()
        return mean
    }

    /**
     * Mean absolute residual of recovered 3D triangle edges vs known rigid edge length.
     * Does NOT mix range-vs-depth (those differ off-axis and caused false rejects).
     */
    private fun estimateEdgeResidualMm(
        pnp: PnPDistanceCalculator.PnPResult,
        edgeLengthMm: Float
    ): Float {
        val p1 = pnp.point1Coords
        val p2 = pnp.point2Coords
        val p3 = pnp.point3Coords ?: return abs(dist3(p1, p2) - edgeLengthMm)
        val e12 = abs(dist3(p1, p2) - edgeLengthMm)
        val e23 = abs(dist3(p2, p3) - edgeLengthMm)
        val e31 = abs(dist3(p3, p1) - edgeLengthMm)
        return (e12 + e23 + e31) / 3f
    }

    private fun dist(a: Offset, b: Offset): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun dist3(a: PnPDistanceCalculator.Point3D, b: PnPDistanceCalculator.Point3D): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        val dz = a.z - b.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
}
