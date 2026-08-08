package org.example.tridotpnp

import androidx.compose.ui.geometry.Offset
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random

data class DetectedTriangle(
    val spots: List<BrightSpot>,
    val score: Float,
    val center: Offset
)

/**
 * Multi-triangle extraction with RANSAC-style sampling over RGB candidate pools.
 * Falls back to greedy non-overlapping selection.
 */
object MultiTriangleExtractor {

    fun extract(
        reds: List<BrightSpot>,
        greens: List<BrightSpot>,
        blues: List<BrightSpot>,
        maxNodes: Int,
        edgeTolerance: Float = 0.15f,
        iterations: Int = 48,
        rng: Random = Random.Default
    ): List<DetectedTriangle> {
        if (reds.isEmpty() || greens.isEmpty() || blues.isEmpty() || maxNodes <= 0) return emptyList()

        val candidates = mutableListOf<DetectedTriangle>()
        val rPool = reds
        val gPool = greens
        val bPool = blues

        repeat(iterations.coerceAtLeast(1)) {
            val r = rPool[rng.nextInt(rPool.size)]
            val g = gPool[rng.nextInt(gPool.size)]
            val b = bPool[rng.nextInt(bPool.size)]
            scoreTriple(r, g, b, edgeTolerance)?.let { candidates += it }
        }

        // Also evaluate a few deterministic top pairings for stability.
        for (r in rPool.take(4)) {
            for (g in gPool.take(4)) {
                for (b in bPool.take(4)) {
                    scoreTriple(r, g, b, edgeTolerance)?.let { candidates += it }
                }
            }
        }

        candidates.sortByDescending { it.score }
        val selected = mutableListOf<DetectedTriangle>()
        for (c in candidates) {
            if (selected.size >= maxNodes) break
            val overlaps = selected.any { other ->
                centersClose(other.center, c.center, minDist = 24f) || sharesSpot(other, c)
            }
            if (!overlaps) selected += c
        }
        return selected
    }

    private fun scoreTriple(
        r: BrightSpot,
        g: BrightSpot,
        b: BrightSpot,
        edgeTolerance: Float
    ): DetectedTriangle? {
        val dRG = dist(r.position, g.position)
        val dGB = dist(g.position, b.position)
        val dBR = dist(b.position, r.position)
        val mean = (dRG + dGB + dBR) / 3f
        if (mean < 4f) return null
        val ratioOk =
            abs(dRG - mean) / mean <= edgeTolerance * 2f &&
                abs(dGB - mean) / mean <= edgeTolerance * 2f &&
                abs(dBR - mean) / mean <= edgeTolerance * 2f
        if (!ratioOk) return null

        // Circum/inradius proxy: for equilateral, R/r = 2.
        val s = (dRG + dGB + dBR) / 2f
        val area2 = s * (s - dRG) * (s - dGB) * (s - dBR)
        if (area2 <= 1e-3f) return null
        val area = sqrt(area2)
        val inR = area / s
        val circR = dRG * dGB * dBR / (4f * area)
        val rr = if (inR > 1e-3f) circR / inR else 10f
        val geo = (1f - abs(rr - 2f) / 2f).coerceIn(0f, 1f)
        val color = (r.redIntensity + g.greenIntensity + b.blueIntensity)
        val score = color * (0.4f + 0.6f * geo)
        val center = Offset(
            (r.position.x + g.position.x + b.position.x) / 3f,
            (r.position.y + g.position.y + b.position.y) / 3f
        )
        return DetectedTriangle(
            spots = listOf(
                r.copy(color = LedColor.RED),
                g.copy(color = LedColor.GREEN),
                b.copy(color = LedColor.BLUE)
            ),
            score = score,
            center = center
        )
    }

    private fun sharesSpot(a: DetectedTriangle, b: DetectedTriangle): Boolean {
        for (sa in a.spots) {
            for (sb in b.spots) {
                if (dist(sa.position, sb.position) < 8f) return true
            }
        }
        return false
    }

    private fun centersClose(a: Offset, b: Offset, minDist: Float): Boolean =
        dist(a, b) < minDist

    private fun dist(a: Offset, b: Offset): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }
}
