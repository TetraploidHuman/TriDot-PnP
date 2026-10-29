package org.example.tridotpnp

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 小检测器的端侧实现：15x15 颜色差图 patch → 「是不是真 LED 圆点」概率。
 *
 * 权重由 `research/scripts/export_dot_cls.py` 从 PyTorch checkpoint 导出
 * （BatchNorm 已折进卷积），以 little-endian float32 依次排列：
 * c1w(16,1,3,3) c1b(16) c2w(32,16,3,3) c2b(32) c3w(64,32,3,3) c3b(64) fcw(64) fcb(1)。
 * 端侧没有 torch / ONNX Runtime 依赖，3 层卷积 + GAP + 单线性层直接手写前向。
 */
internal class DotPatchWeights private constructor(private val tensors: Map<String, FloatArray>) {

    fun tensor(name: String): FloatArray = tensors.getValue(name)

    companion object {
        const val PATCH = 15
        const val IN_CHANNELS = 1
        const val INPUT_SCALE = 1f / 255f

        private val SHAPES: List<Pair<String, IntArray>> = listOf(
            "c1w" to intArrayOf(16, 1, 3, 3), "c1b" to intArrayOf(16),
            "c2w" to intArrayOf(32, 16, 3, 3), "c2b" to intArrayOf(32),
            "c3w" to intArrayOf(64, 32, 3, 3), "c3b" to intArrayOf(64),
            "fcw" to intArrayOf(64), "fcb" to intArrayOf(1)
        )

        private fun size(shape: IntArray): Int = shape.fold(1) { acc, v -> acc * v }

        fun fromBytes(bytes: ByteArray): DotPatchWeights {
            require(bytes.size % 4 == 0) { "weights size must be a multiple of 4" }
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val map = LinkedHashMap<String, FloatArray>()
            for ((name, shape) in SHAPES) {
                val count = size(shape)
                require(buf.remaining() >= count * 4) { "weights truncated at $name" }
                val arr = FloatArray(count)
                for (i in arr.indices) arr[i] = buf.float
                map[name] = arr
            }
            return DotPatchWeights(map)
        }

        fun fromStream(stream: InputStream): DotPatchWeights =
            stream.use { fromBytes(it.readBytes()) }
    }
}

internal class DotPatchClassifier(private val w: DotPatchWeights) {

    private val c1w = w.tensor("c1w"); private val c1b = w.tensor("c1b")
    private val c2w = w.tensor("c2w"); private val c2b = w.tensor("c2b")
    private val c3w = w.tensor("c3w"); private val c3b = w.tensor("c3b")
    private val fcw = w.tensor("fcw"); private val fcb = w.tensor("fcb")

    /** [patch] is a row-major 15x15 map of the colour-difference channel (0..255). */
    fun score(patch: FloatArray): Float {
        require(patch.size == DotPatchWeights.PATCH * DotPatchWeights.PATCH)
        val n = DotPatchWeights.PATCH
        val input = FloatArray(patch.size) { patch[it] * DotPatchWeights.INPUT_SCALE }
        var x = conv3x3(input, 1, n, n, 16, c1w, c1b)
        x = relu(x)
        x = maxPool2(x, 16, n, n)
        val h2 = n / 2
        x = conv3x3(x, 16, h2, h2, 32, c2w, c2b)
        x = relu(x)
        x = maxPool2(x, 32, h2, h2)
        val h3 = h2 / 2
        x = conv3x3(x, 32, h3, h3, 64, c3w, c3b)
        x = relu(x)
        val gap = FloatArray(64)
        for (c in 0 until 64) {
            var s = 0f
            for (i in 0 until h3 * h3) s += x[c * h3 * h3 + i]
            gap[c] = s / (h3 * h3)
        }
        var logit = fcb[0]
        for (c in 0 until 64) logit += fcw[c] * gap[c]
        return 1f / (1f + kotlin.math.exp(-logit))
    }

    private fun conv3x3(src: FloatArray, inCh: Int, h: Int, w2: Int, outCh: Int,
                        weights: FloatArray, bias: FloatArray): FloatArray {
        val out = FloatArray(outCh * h * w2)
        val plane = h * w2
        for (oc in 0 until outCh) {
            val outBase = oc * plane
            val b = bias[oc]
            for (y in 0 until h) {
                for (xx in 0 until w2) {
                    var acc = b
                    for (ic in 0 until inCh) {
                        val srcBase = ic * plane
                        val wBase = ((oc * inCh) + ic) * 9
                        for (ky in -1..1) {
                            val sy = y + ky
                            if (sy < 0 || sy >= h) continue
                            for (kx in -1..1) {
                                val sx = xx + kx
                                if (sx < 0 || sx >= w2) continue
                                acc += src[srcBase + sy * w2 + sx] * weights[wBase + (ky + 1) * 3 + (kx + 1)]
                            }
                        }
                    }
                    out[outBase + y * w2 + xx] = acc
                }
            }
        }
        return out
    }

    private fun relu(x: FloatArray): FloatArray = FloatArray(x.size) { maxOf(0f, x[it]) }

    private fun maxPool2(src: FloatArray, ch: Int, h: Int, w2: Int): FloatArray {
        val oh = h / 2
        val ow = w2 / 2
        val out = FloatArray(ch * oh * ow)
        for (c in 0 until ch) {
            val sb = c * h * w2
            val ob = c * oh * ow
            for (y in 0 until oh) {
                for (xx in 0 until ow) {
                    val y0 = y * 2
                    val x0 = xx * 2
                    var m = src[sb + y0 * w2 + x0]
                    m = maxOf(m, src[sb + y0 * w2 + x0 + 1])
                    m = maxOf(m, src[sb + (y0 + 1) * w2 + x0])
                    m = maxOf(m, src[sb + (y0 + 1) * w2 + x0 + 1])
                    out[ob + y * ow + xx] = m
                }
            }
        }
        return out
    }
}

/**
 * 15x15 patch 提取，与 `research/tridot_temporal/dot_cls.patch_from_map_scaled`
 * 保持一致：窗口 = 15·clip(r_exp/3, 0.45, 2.5)，下采样用面积平均，放大用双线性。
 */
internal object DotPatchExtractor {
    const val PATCH = 15

    fun scaled(map: FloatArray, width: Int, height: Int, u: Float, v: Float,
               expectedRadiusPx: Float): FloatArray {
        val scale = if (expectedRadiusPx <= 1e-3f) 1f
        else (expectedRadiusPx / 3f).coerceIn(0.45f, 2.5f)
        val win = maxOf(5, Math.round(PATCH * scale))
        val half = win / 2
        val ui = Math.round(u)
        val vi = Math.round(v)
        val x0 = ui - half
        val y0 = vi - half
        val out = FloatArray(PATCH * PATCH)
        if (win == PATCH) {
            for (y in 0 until PATCH) {
                for (x in 0 until PATCH) {
                    val sx = x0 + x
                    val sy = y0 + y
                    out[y * PATCH + x] = if (sx in 0 until width && sy in 0 until height)
                        map[sy * width + sx] else 0f
                }
            }
            return out
        }
        if (win > PATCH) {
            // area average of the source window for each output pixel
            val step = win.toFloat() / PATCH
            for (y in 0 until PATCH) {
                val ya = (y0 + y * step).toInt()
                val yb = (y0 + (y + 1) * step).toInt().coerceAtMost(y0 + win)
                for (x in 0 until PATCH) {
                    val xa = (x0 + x * step).toInt()
                    val xb = (x0 + (x + 1) * step).toInt().coerceAtMost(x0 + win)
                    var sum = 0f
                    var cnt = 0
                    for (sy in ya until yb) {
                        if (sy < 0 || sy >= height) continue
                        for (sx in xa until xb) {
                            if (sx < 0 || sx >= width) continue
                            sum += map[sy * width + sx]
                            cnt++
                        }
                    }
                    out[y * PATCH + x] = if (cnt > 0) sum / cnt else 0f
                }
            }
            return out
        }
        // upscale: bilinear sample at output pixel centres
        val step = win.toFloat() / PATCH
        for (y in 0 until PATCH) {
            val fy = y0 + (y + 0.5f) * step - 0.5f
            for (x in 0 until PATCH) {
                val fx = x0 + (x + 0.5f) * step - 0.5f
                out[y * PATCH + x] = bilinear(map, width, height, fx, fy)
            }
        }
        return out
    }

    private fun bilinear(map: FloatArray, width: Int, height: Int, x: Float, y: Float): Float {
        val x0 = kotlin.math.floor(x).toInt()
        val y0 = kotlin.math.floor(y).toInt()
        val fx = x - x0
        val fy = y - y0
        fun at(sx: Int, sy: Int): Float =
            if (sx < 0 || sy < 0 || sx >= width || sy >= height) 0f else map[sy * width + sx]
        val a = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx
        val b = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx
        return a * (1 - fy) + b * fy
    }
}

/**
 * 从 Android assets 懒加载端侧小检测器（`assets/dot_cls/dot_cls.bin`）。
 * 加载失败（资源缺失/版本不匹配）时静默退化为 null，检测行为回到经典路径。
 */
internal object DotClassifierAssets {
    @Volatile private var state = 0  // 0 = not tried, 1 = loaded/none, 2 = failed
    @Volatile private var instance: DotPatchClassifier? = null

    fun init(context: android.content.Context) {
        if (state != 0) return
        synchronized(this) {
            if (state != 0) return
            state = 1
            instance = runCatching {
                context.assets.open("dot_cls/dot_cls.bin").use { stream ->
                    DotPatchClassifier(DotPatchWeights.fromStream(stream))
                }
            }.onFailure { it.printStackTrace() }.getOrNull()
        }
    }

    fun get(): DotPatchClassifier? = instance

    /** 单元测试用：注入预先加载好的分类器。 */
    internal fun setForTest(classifier: DotPatchClassifier?) {
        state = 1
        instance = classifier
    }
}
