package org.example.tridotpnp

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * 主线三点自洽性校验：远距/泛光下 BrightSpotDetector 可能给出三个高分但几何荒谬的点，
 * 这层校验决定是否让远距辅助接管。这里直接测几何规则（不依赖 Android 图形栈）。
 */
class TripleConsistencyTest {
    private fun pts(a: FloatArray, b: FloatArray, c: FloatArray) = arrayOf(a, b, c)

    /** 边长约 100px 的正三角形。 */
    private val good = pts(
        floatArrayOf(100f, 100f),
        floatArrayOf(200f, 100f),
        floatArrayOf(150f, 186.6f)
    )

    @Test
    fun equilateralTripleIsPlausible() {
        assertTrue(TripleConsistency.isPlausible(good, 100f))
        assertEquals(100f, TripleConsistency.meanEdge(good), 0.5f)
    }

    @Test
    fun collinearTripleIsRejected() {
        val bad = pts(floatArrayOf(10f, 10f), floatArrayOf(110f, 10f), floatArrayOf(210f, 10f))
        assertFalse(TripleConsistency.isPlausible(bad, 0f))
    }

    @Test
    fun scaleMismatchIsRejectedWhenPriorGiven() {
        // 形状像等边，但视边长只有先验的一半 → 多半是把近处干扰块当成了目标
        assertFalse(TripleConsistency.isPlausible(good, 200f))
        // 无先验时只做形状校验，应通过
        assertTrue(TripleConsistency.isPlausible(good, 0f))
    }

    @Test
    fun tinyTriangleIsRejected() {
        val tiny = pts(floatArrayOf(10f, 10f), floatArrayOf(14f, 10f), floatArrayOf(12f, 13.5f))
        assertFalse(TripleConsistency.isPlausible(tiny, 0f))
    }

    @Test
    fun squashedTripleIsRejected() {
        // 一条边 100、一条 100、一条 400：主轴严重拉伸，不应通过形状校验
        val squashed = pts(floatArrayOf(0f, 0f), floatArrayOf(100f, 0f), floatArrayOf(500f, 0f))
        assertFalse(TripleConsistency.isPlausible(squashed, 0f))
    }

    @Test
    fun twoPointsAreRejected() {
        assertFalse(TripleConsistency.isPlausible(arrayOf(floatArrayOf(0f, 0f), floatArrayOf(10f, 0f)), 0f))
        assertEquals(0f, TripleConsistency.meanEdge(emptyArray()))
    }
}
