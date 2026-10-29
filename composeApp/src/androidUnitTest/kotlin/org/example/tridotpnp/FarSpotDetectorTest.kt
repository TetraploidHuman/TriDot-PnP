package org.example.tridotpnp

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * JVM 单测：远距前端在「1-2px 小光斑 + 背景泛光 + 噪声」下的行为。
 *
 * 这里直接用 IntArray 而不是 Bitmap，避免 JVM 单测里依赖 Android 图形栈。
 * 颜色约定与 App 内一致：像素为 ARGB。
 */
class FarSpotDetectorTest {
    private val detector = FarSpotDetector()

    // ===== 测试 =====

    @Test
    fun detectsSmallDotsOnDarkBackground() {
        val image = Image(320, 240)
        image.dot(60f, 70f, 1.2f, 0xFFE61919.toInt())
        image.dot(150f, 78f, 1.2f, 0xFF19E619.toInt())
        image.dot(96f, 168f, 1.2f, 0xFF1919E6.toInt())

        val spots = detector.detectRgb(image.pixels, image.width, image.height)

        assertNotNull(spots)
        assertEquals(3, spots.size)
        assertNear(spots, 60f, 70f)
        assertNear(spots, 150f, 78f)
        assertNear(spots, 96f, 168f)
    }

    @Test
    fun detectsDotsUnderSensorNoise() {
        val image = Image(320, 240)
        for (y in 0 until image.height) {
            for (x in 0 until image.width) {
                val n = image.random.nextInt(-12, 13)
                image.pixels[y * image.width + x] = rgb(16 + n, 16 + n, 16 + n)
            }
        }
        image.dot(80f, 60f, 1.4f, 0xFFE61919.toInt())
        image.dot(170f, 72f, 1.4f, 0xFF19E619.toInt())
        image.dot(120f, 150f, 1.4f, 0xFF1919E6.toInt())

        val spots = detector.detectRgb(image.pixels, image.width, image.height)

        assertNotNull(spots)
        assertEquals(3, spots.size)
        assertNear(spots, 80f, 60f)
        assertNear(spots, 170f, 72f)
        assertNear(spots, 120f, 150f)
    }

    @Test
    fun ignoresLargeWhiteBlob() {
        val image = Image(320, 240)
        // 白墙/桌面反光：40x40 大块亮区
        image.rect(180, 100, 40, 40, 0xFFF0F0F0.toInt())
        image.dot(60f, 70f, 1.2f, 0xFFE61919.toInt())
        image.dot(150f, 78f, 1.2f, 0xFF19E619.toInt())
        image.dot(96f, 168f, 1.2f, 0xFF1919E6.toInt())

        val spots = detector.detectRgb(image.pixels, image.width, image.height)

        assertNotNull(spots)
        assertEquals(3, spots.size)
        assertNear(spots, 96f, 168f)
    }

    @Test
    fun returnsNullWhenColorMissing() {
        val image = Image(320, 240)
        image.dot(60f, 70f, 1.2f, 0xFFE61919.toInt())
        image.dot(150f, 78f, 1.2f, 0xFF19E619.toInt())
        // 没有蓝点

        assertNull(detector.detectRgb(image.pixels, image.width, image.height))
    }

    @Test
    fun returnsNullForNonEquilateralLayout() {
        val image = Image(320, 240)
        // 三点几乎共线：不满足等边几何约束
        image.dot(60f, 120f, 1.2f, 0xFFE61919.toInt())
        image.dot(160f, 124f, 1.2f, 0xFF19E619.toInt())
        image.dot(260f, 128f, 1.2f, 0xFF1919E6.toInt())

        assertNull(detector.detectRgb(image.pixels, image.width, image.height))
    }

    @Test
    fun adaptiveKernelStillFindsTinyDots() {
        val image = Image(320, 240)
        image.dot(90f, 90f, 1.0f, 0xFFE61919.toInt())
        image.dot(190f, 100f, 1.0f, 0xFF19E619.toInt())
        image.dot(140f, 180f, 1.0f, 0xFF1919E6.toInt())

        // expectedEdgePx≈实际视边长（≈97px），让尺度项选中小核；同时验证评分器不被 31 核丢掉 1px 点
        val spots = detector.detectRgb(image.pixels, image.width, image.height, expectedEdgePx = 97f)

        assertNotNull(spots)
        assertEquals(3, spots.size)
        assertNear(spots, 90f, 90f)
    }

    @Test
    fun chromaStrengthSeparatesColorfulFromDarkFrames() {
        val dark = Image(160, 120)
        val colorful = Image(160, 120)
        colorful.rect(20, 20, 20, 20, 0xFFE61919.toInt())

        val darkStrength = chromaStrength(dark.pixels)
        val colorStrength = chromaStrength(colorful.pixels)

        assertTrue(colorStrength > darkStrength + 2f, "彩色帧的色度强度应明显更高")
    }

    @Test
    fun recoversChromaDegradedDotFromLuma() {
        // 真机 YUV 4:2:0 会把 1-2px 彩点的色度平均掉，只剩灰白亮度。
        val image = Image(320, 240)
        image.dot(60f, 70f, 1.2f, 0xFFE61919.toInt())
        image.dot(150f, 78f, 1.2f, 0xFF19E619.toInt())
        image.dot(96f, 168f, 1.6f, 0xFFDCDCDC.toInt())

        val spots = detector.detectRgb(image.pixels, image.width, image.height)

        assertNotNull(spots)
        assertEquals(3, spots.size)
        val blue = spots.first { it.color == LedColor.BLUE }
        val dist = hypot(blue.u - 96f, blue.v - 168f)
        assertTrue(dist <= 1.5f, "色度丢失的蓝点应在 (96,168) 附近，实际 (%.1f, %.1f)".format(blue.u, blue.v))
    }

    @Test
    fun staysFastEnoughForPerFrameFallback() {
        // 640x360（端侧 maxDim=640 的分析分辨率），多尺度核 3/7/15/31 走一遍；
        // 这条路径只在「常规检测凑不齐三点」的帧触发，单帧预算按 250ms 兜。
        val image = Image(640, 360)
        for (y in 10 until image.height step 23) {
            for (x in 10 until image.width step 31) {
                val c = when ((x + y) % 3) {
                    0 -> 0xFFE61919.toInt()
                    1 -> 0xFF19E619.toInt()
                    else -> 0xFF1919E6.toInt()
                }
                image.dot(x.toFloat(), y.toFloat(), 1.3f, c)
            }
        }
        val t0 = System.nanoTime()
        repeat(3) { detector.detectRgb(image.pixels, image.width, image.height) }
        val ms = (System.nanoTime() - t0) / 3.0 / 1_000_000.0
        println("[FarSpotDetectorTest] 640x360 多尺度回退平均耗时 = %.1f ms".format(ms))
        assertTrue(ms < 250.0, "远距回退平均耗时 %.1fms".format(ms))
    }

    // ===== 辅助 =====

    private fun assertNear(spots: List<FarSpot>, u: Float, v: Float, tol: Float = 1.5f) {
        val hit = spots.minByOrNull { hypot(it.u - u, it.v - v) }
        assertNotNull(hit)
        val dist = hypot(hit.u - u, hit.v - v)
        assertTrue(dist <= tol, "未命中 (%.1f, %.1f)：最近点 (%.1f, %.1f) 距离 %.2fpx".format(u, v, hit.u, hit.v, dist))
    }

    private class Image(val width: Int, val height: Int) {
        val pixels = IntArray(width * height) { 0xFF101010.toInt() }
        val random = Random(42)

        fun dot(cx: Float, cy: Float, radius: Float, color: Int) {
            val r2 = radius * radius
            val x0 = (cx - radius).toInt().coerceAtLeast(0)
            val x1 = (cx + radius).toInt().coerceAtMost(width - 1)
            val y0 = (cy - radius).toInt().coerceAtLeast(0)
            val y1 = (cy + radius).toInt().coerceAtMost(height - 1)
            for (y in y0..y1) {
                for (x in x0..x1) {
                    val dx = x - cx
                    val dy = y - cy
                    if (dx * dx + dy * dy <= r2) pixels[y * width + x] = color
                }
            }
        }

        fun rect(x0: Int, y0: Int, w: Int, h: Int, color: Int) {
            for (y in y0 until (y0 + h).coerceAtMost(height)) {
                for (x in x0 until (x0 + w).coerceAtMost(width)) {
                    pixels[y * width + x] = color
                }
            }
        }
    }

    private fun rgb(r: Int, g: Int, b: Int): Int {
        val rr = r.coerceIn(0, 255)
        val gg = g.coerceIn(0, 255)
        val bb = b.coerceIn(0, 255)
        return (0xFF shl 24) or (rr shl 16) or (gg shl 8) or bb
    }
}

/**
 * C：小光斑的抛物线亚像素细化。
 *
 * 直接构造响应图（不经过颜色图），验证 1px 光斑的坐标误差被压到 <0.35px，
 * 以及平坦/过冲输入不会把坐标推出 ±0.5px。
 */
class FarSpotSubpixelTest {
    private fun gaussian(
        w: Int, h: Int, cx: Float, cy: Float, sigma: Float = 0.75f, peak: Float = 200f
    ): FloatArray {
        val out = FloatArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val dx = x - cx
            val dy = y - cy
            out[y * w + x] = peak * kotlin.math.exp(-(dx * dx + dy * dy) / (2f * sigma * sigma))
        }
        return out
    }

    private fun refine(map: FloatArray, w: Int, h: Int, x: Float, y: Float): FloatArray {
        val m = FarSpotDetector::class.java.getDeclaredMethod(
            "refineSubpixel", FloatArray::class.java, Int::class.javaPrimitiveType,
            Int::class.javaPrimitiveType, Float::class.javaPrimitiveType,
            Float::class.javaPrimitiveType
        )
        m.isAccessible = true
        @Suppress("UNCHECKED_CAST")
        return m.invoke(dummy, map, w, h, x, y) as FloatArray
    }

    private val dummy = FarSpotDetector()

    @Test
    fun parabolaRecoversFractionalCentre() {
        val w = 32
        val h = 24
        val map = gaussian(w, h, 15.35f, 9.7f)
        val refined = refine(map, w, h, 15f, 10f)
        assertTrue(abs(refined[0] - 15.35f) < 0.35f, "dx=${refined[0]}")
        assertTrue(abs(refined[1] - 9.7f) < 0.35f, "dy=${refined[1]}")
    }

    @Test
    fun flatMapKeepsCoordinate() {
        val map = FloatArray(16 * 16) { 42f }
        val refined = refine(map, 16, 16, 7f, 8f)
        assertEquals(7f, refined[0])
        assertEquals(8f, refined[1])
    }

    @Test
    fun offsetIsClamped() {
        // 左右差值极大但中心极小的病态图案：位移必须被夹在 ±0.5px
        val w = 9
        val h = 9
        val map = FloatArray(w * h) { 0f }
        map[4 * w + 3] = 100f
        map[4 * w + 4] = 1f
        map[4 * w + 5] = 0f
        val refined = refine(map, w, h, 4f, 4f)
        assertTrue(refined[0] in 3.5f..4.5f, "dx=${refined[0]}")
    }
}
