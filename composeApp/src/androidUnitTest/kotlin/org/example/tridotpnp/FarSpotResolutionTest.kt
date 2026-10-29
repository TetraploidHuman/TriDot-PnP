package org.example.tridotpnp

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class FarSpotResolutionTest {
    private fun pixels(width: Int, height: Int, radius: Int): IntArray {
        val px = IntArray(width * height) { 0xFF000000.toInt() }
        val cols = intArrayOf(0xFFFF0000.toInt(), 0xFF00FF00.toInt(), 0xFF0000FF.toInt())
        val pts = arrayOf(
            intArrayOf(width / 3, height / 2),
            intArrayOf(width * 2 / 3, height / 2),
            intArrayOf(width / 2, height / 2 - width / 12)
        )
        for (i in 0 until 3) {
            for (dy in -radius..radius) for (dx in -radius..radius) {
                if (dx * dx + dy * dy > radius * radius) continue
                val x = pts[i][0] + dx
                val y = pts[i][1] + dy
                if (x !in 0 until width || y !in 0 until height) continue
                px[y * width + x] = cols[i]
            }
        }
        return px
    }

    @Test
    fun detectsTinyDotsAtBothAnalysisSizes() {
        val w = 1280
        val h = 720
        val px = pixels(w, h, radius = 3)
        val expectedEdge = 300f
        for (maxDim in intArrayOf(640, 512, 448)) {
            val det = FarSpotDetector(maxDim = maxDim)
            val spots = det.detectRgb(px, w, h, expectedEdgePx = expectedEdge)
            assertNotNull(spots, "maxDim=$maxDim 应检出三点")
            assertTrue(spots!!.size == 3)
            val meanEdge = (0 until 3).sumOf { i ->
                val a = spots[i]
                val b = spots[(i + 1) % 3]
                hypot((a.u - b.u).toDouble(), (a.v - b.v).toDouble())
            } / 3.0
            assertTrue(
                abs(meanEdge - expectedEdge) / expectedEdge < 0.35,
                "maxDim=$maxDim 平均边长 $meanEdge 应接近 $expectedEdge"
            )
        }
    }
}
