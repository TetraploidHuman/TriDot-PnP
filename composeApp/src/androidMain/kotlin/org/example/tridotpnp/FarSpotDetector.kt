package org.example.tridotpnp

import android.graphics.Bitmap
import androidx.compose.ui.geometry.Offset
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 远距/小光斑检测（研究侧 classic_v3 的端侧精简版）。
 *
 * 与 [BrightSpotDetector] 的绝对亮度阈值不同，这里：
 *  1. 用「颜色差图」 r−max(g,b) 等得到对白色饱和核心不敏感的色度响应；
 *  2. 用形态学顶帽（盘状核，核大小由目标视边长/上一帧位姿给出的光斑半径决定）
 *     去掉桌面/墙面这类缓慢变化的背景泛光；
 *  3. 阈值由高频 MAD 噪声 + 画面亮尾分位数自适应决定，没有绝对亮度门；
 *  4. 8 连通域 + 响应加权质心（1–2px 小点也能出亚像素）。
 *
 * 只在常规检测凑不齐三点时才被调用（见 CameraFrameProcessing），
 * 并且输出还要通过「等边三角形 + 边长下限」几何校验，避免在无目标场景里报假点。
 */
data class FarSpot(
    val u: Float,
    val v: Float,
    val score: Float,
    val color: LedColor,
    val peak: Float,
    /** 连通域像素数；0 表示未知（外部构造的候选）。用于无先验时的尺度估计。 */
    val area: Int = 0
)

/** 候选三元组校验器：返回代价（越小越好），null 表示拒绝。 */
fun interface TripleValidator {
    fun validate(tri: List<FarSpot>): Float?
}

class FarSpotDetector(
    /** 分析分辨率上限（长边像素）。小点检测不需要全 720p，640 足够且快 4 倍。 */
    var maxDim: Int = 640,
    /** 每色保留的候选数（用于跨色配对前的排序）。 */
    val topKPerColor: Int = 12,
    val minArea: Int = 1,
    val maxAreaFrac: Float = 0.04f,
    val maxAspect: Float = 6f,
    /** 阈值下限（颜色差单位 0..255）。 */
    val floor: Float = 12f,
    /** 噪声倍数，σ 用高频 MAD 估计。 */
    val kSigma: Float = 3.5f,
    /** 相对画面亮尾的比例下限，抑制中亮度杂波。 */
    val relFrac: Float = 0.18f,
    /** 光斑半径 ≈ dotRadiusRatio × 视边长（与合成集实测 0.087 对齐）。 */
    val dotRadiusRatio: Float = 0.09f,
) {
    /** 便捷入口：从 Bitmap 检测，返回三点（RED/GREEN/BLUE）或 null。 */
    /**
     * 可选的小检测器（15x15 patch → 圆点概率）。端侧权重来自
     * `research/scripts/export_dot_cls.py`，由 [DotClassifierAssets] 懒加载；
     * 单元测试直接注入。为 null 时行为与之前完全一致。
     */
    internal var classifier: DotPatchClassifier? = null

    /** 分类器概率与经典分数的混合权重：score = (1-w)·classic + w·P(dot)。 */
    internal var classifierWeight: Float = 0.5f

    fun detect(
        bitmap: Bitmap,
        expectedEdgePx: Float = 0f,
        predictedSpots: List<Offset>? = null,
        validator: TripleValidator? = null
    ): List<BrightSpot>? {
        val w = bitmap.width
        val h = bitmap.height
        if (w < 8 || h < 8) return null
        val scale = min(1f, maxDim.toFloat() / max(w, h))
        val sw = max(1, (w * scale).roundToInt())
        val sh = max(1, (h * scale).roundToInt())
        val scaled = if (scale < 1f) Bitmap.createScaledBitmap(bitmap, sw, sh, true) else bitmap
        val pixels = IntArray(sw * sh)
        scaled.getPixels(pixels, 0, sw, 0, 0, sw, sh)
        if (scaled !== bitmap) scaled.recycle()
        val predicted = predictedSpots?.let { spots ->
            FloatArray(spots.size * 2) { idx ->
                val spot = spots[idx / 2]
                if (idx % 2 == 0) (spot.x * scale) else (spot.y * scale)
            }
        }
        return detectRgb(pixels, sw, sh, expectedEdgePx * scale, predicted, validator)
            ?.map { spot ->
                BrightSpot(
                    position = Offset(spot.u / scale, spot.v / scale),
                    brightness = spot.score,
                    color = spot.color,
                    redIntensity = if (spot.color == LedColor.RED) spot.peak else 0f,
                    greenIntensity = if (spot.color == LedColor.GREEN) spot.peak else 0f,
                    blueIntensity = if (spot.color == LedColor.BLUE) spot.peak else 0f
                )
            }
    }

    /** 纯数组入口（可在 JVM 单测里直接用，不依赖 Bitmap）。 */
    fun detectRgb(
        pixels: IntArray,
        width: Int,
        height: Int,
        expectedEdgePx: Float = 0f,
        predicted: FloatArray? = null,
        validator: TripleValidator? = null
    ): List<FarSpot>? {
        if (width < 8 || height < 8 || pixels.size < width * height) return null
        val n = width * height
        val red = FloatArray(n)
        val green = FloatArray(n)
        val blue = FloatArray(n)
        val luma = FloatArray(n)
        for (i in 0 until n) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF).toFloat()
            val g = ((p shr 8) and 0xFF).toFloat()
            val b = (p and 0xFF).toFloat()
            red[i] = (r - max(g, b)).coerceAtLeast(0f)
            green[i] = (g - max(r, b)).coerceAtLeast(0f)
            blue[i] = (b - max(r, g)).coerceAtLeast(0f)
            luma[i] = max(r, max(g, b))
        }
        val ks = kernelSizes(expectedEdgePx, width, height)
        val maps = listOf(LedColor.RED to red, LedColor.GREEN to green, LedColor.BLUE to blue)
        val candidates = ArrayList<FarSpot>(3 * topKPerColor)
        val pending = maps.mapTo(ArrayList()) { it.first }
        // 尺度由小到大：小光斑在小核上就能出候选，凑齐三色后不再跑更大的核（常见帧只花一次顶帽）。
        for (k in ks) {
            for ((color, map) in maps) {
                if (color !in pending) continue
                val response = topHatResponse(map, width, height, intArrayOf(k))
                val found = extract(color, response, width, height, peakWindow = max(3, k / 2))
                // union：对比度候选之外再并入「原始颜色差图 + 绝对阈值」候选，
                // 饱和点/强泛光下顶帽会削掉光斑核心，原始阈值能补回来（研究侧 classic_v3 union 证据）。
                val rawFound = extract(color, map, width, height, absolute = true, peakWindow = max(3, k / 2))
                val merged = (found + rawFound).distinctBy { (it.u / 12f).toInt() * 10000 + (it.v / 12f).toInt() }
                    .sortedByDescending { it.score }
                    .take(topKPerColor)
                if (merged.isNotEmpty()) {
                    candidates += merged
                    pending.remove(color)
                }
            }
            if (pending.isEmpty()) break
        }
        if (pending.isNotEmpty()) {
            // D：色度被 4:2:0 / 降噪吃掉时，彩点会退化成灰白点。
            // 用亮度顶帽找候选，再按「该处没有别的颜色的强证据」给缺失颜色补位。
            val lumaResponse = topHatResponse(luma, width, height, ks)
            val grey = extract(LedColor.UNKNOWN, lumaResponse, width, height, peakWindow = max(3, ks.last() / 2))
            for (spot in grey) {
                val idx = spot.v.toInt().coerceIn(0, height - 1) * width + spot.u.toInt().coerceIn(0, width - 1)
                for (missing in pending) {
                    val own = when (missing) {
                        LedColor.RED -> red[idx]
                        LedColor.GREEN -> green[idx]
                        LedColor.BLUE -> blue[idx]
                        else -> 0f
                    }
                    val others = max(
                        max(
                            if (missing == LedColor.RED) 0f else red[idx],
                            if (missing == LedColor.GREEN) 0f else green[idx]
                        ),
                        if (missing == LedColor.BLUE) 0f else blue[idx]
                    )
                    if (others >= 40f && others >= 1.6f * own) continue  // 明显是别的颜色，别抢
                    candidates.add(spot.copy(color = missing))
                    break
                }
            }
        }
        applyClassifier(candidates, red, green, blue, width, height, expectedEdgePx)
        return pickTriangle(candidates, expectedEdgePx, predicted, validator)
    }

    /**
     * 候选级重打分：按尺度归一化的 15x15 patch 估「圆点概率」，与经典分数加权混合。
     * 与 Python 侧 `ClassicV3Detector(dot_cls_path=..., dot_cls_scale=True)` 对齐。
     */
    private fun applyClassifier(
        candidates: MutableList<FarSpot>,
        red: FloatArray,
        green: FloatArray,
        blue: FloatArray,
        width: Int,
        height: Int,
        expectedEdgePx: Float
    ) {
        val clf = classifier ?: return
        if (candidates.isEmpty()) return
        val maps = mapOf(LedColor.RED to red, LedColor.GREEN to green, LedColor.BLUE to blue)
        // 有上一帧三点时用「视边长 × 点半径比」；没有先验时用本帧候选的中位等效半径，
        // 这样第一帧（或跟踪丢失后）也能做尺度归一化，而不是退化成固定 15px 窗口。
        val rExp = if (expectedEdgePx > 1f) {
            dotRadiusRatio * expectedEdgePx
        } else {
            val radii = candidates.mapNotNull { c ->
                if (c.area > 0) kotlin.math.sqrt(c.area.toFloat() / kotlin.math.PI.toFloat()) else null
            }.sorted()
            if (radii.isEmpty()) 0f else radii[radii.size / 2].coerceIn(0.8f, 12f)
        }
        val rescored = ArrayList<FarSpot>(candidates.size)
        for (spot in candidates) {
            val m = maps[spot.color] ?: continue
            val patch = DotPatchExtractor.scaled(m, width, height, spot.u, spot.v, rExp)
            val p = clf.score(patch)
            rescored.add(spot.copy(score = (1f - classifierWeight) * spot.score + classifierWeight * p))
        }
        if (rescored.isEmpty()) return
        val byColor = rescored.groupBy { it.color }
        candidates.clear()
        for ((_, list) in byColor) {
            candidates.addAll(list.sortedByDescending { it.score }.take(topKPerColor))
        }
        candidates.sortByDescending { it.score }
    }

    // ===== 内部实现 =====

    private fun kernelSizes(expectedEdgePx: Float, width: Int, height: Int): IntArray {
        val diag = min(width, height).toFloat()
        if (expectedEdgePx > 1f) {
            val radius = (dotRadiusRatio * expectedEdgePx).coerceIn(0.6f, diag / 6f)
            val k = oddKernel(2 * (2f * radius).roundToInt() + 1)
            return intArrayOf(k)
        }
        // 不知道距离：多尺度取最大响应（近处大点靠大核，远处小点靠小核）
        return intArrayOf(3, 7, 15, 31)
    }

    private fun oddKernel(k: Int): Int {
        val even = k % 2 == 0
        return (k + if (even) 1 else 0).coerceIn(3, 31)
    }

    /** 顶帽响应 = 原图 − 开运算（先腐蚀再膨胀），多尺度取逐像素最大。 */
    private fun topHatResponse(map: FloatArray, width: Int, height: Int, ks: IntArray): FloatArray {
        var best: FloatArray? = null
        for (k in ks) {
            val erode = minFilter(map, width, height, k)
            val opened = maxFilter(erode, width, height, k)
            val resp = FloatArray(map.size) { (map[it] - opened[it]).coerceAtLeast(0f) }
            if (best == null) {
                best = resp
            } else {
                val cur = best
                for (i in resp.indices) if (resp[i] > cur[i]) cur[i] = resp[i]
            }
            if (ks.size == 1) break
        }
        return best ?: FloatArray(map.size)
    }

    /**
     * 可分离方形最小值滤波（边缘复制），用单调队列做到 O(1)/像素 —— 核可以到 31 而不变慢。
     */
    private fun minFilter(src: FloatArray, width: Int, height: Int, k: Int): FloatArray =
        separableFilter(src, width, height, k, isMax = false)

    /** 可分离方形最大值滤波（边缘复制）。 */
    private fun maxFilter(src: FloatArray, width: Int, height: Int, k: Int): FloatArray =
        separableFilter(src, width, height, k, isMax = true)

    private fun separableFilter(
        src: FloatArray,
        width: Int,
        height: Int,
        k: Int,
        isMax: Boolean
    ): FloatArray {
        val r = k / 2
        val horizontal = FloatArray(src.size)
        val out = FloatArray(src.size)
        val deque = IntArray(max(width, height))
        // 水平
        for (y in 0 until height) {
            val row = y * width
            var head = 0
            var tail = 0
            for (x in 0 until width + r) {
                if (x < width) {
                    val v = src[row + x]
                    while (tail > head) {
                        val back = src[row + deque[tail - 1]]
                        if (if (isMax) back <= v else back >= v) tail-- else break
                    }
                    deque[tail++] = x
                }
                val outX = x - r
                if (outX >= 0) {
                    while (deque[head] < outX - r) head++
                    horizontal[row + outX] = src[row + deque[head]]
                }
            }
        }
        // 垂直
        for (x in 0 until width) {
            var head = 0
            var tail = 0
            for (y in 0 until height + r) {
                if (y < height) {
                    val v = horizontal[y * width + x]
                    while (tail > head) {
                        val back = horizontal[deque[tail - 1] * width + x]
                        if (if (isMax) back <= v else back >= v) tail-- else break
                    }
                    deque[tail++] = y
                }
                val outY = y - r
                if (outY >= 0) {
                    while (deque[head] < outY - r) head++
                    out[outY * width + x] = horizontal[deque[head] * width + x]
                }
            }
        }
        return out
    }

    private fun extract(
        color: LedColor,
        response: FloatArray,
        width: Int,
        height: Int,
        absolute: Boolean = false,
        peakWindow: Int = 3
    ): List<FarSpot> {
        val step = 4
        val samples = ArrayList<Float>(max(16, response.size / (step * step)))
        var i = 0
        while (i < response.size) {
            samples.add(response[i])
            i += step
        }
        samples.sort()
        val median = samples[samples.size / 2]
        val devs = FloatArray(samples.size) { abs(samples[it] - median) }
        devs.sort()
        val sigma = 1.4826f * devs[devs.size / 2]
        val p995 = samples[((samples.size - 1) * 0.995f).roundToInt().coerceIn(0, samples.size - 1)]
        val thr = if (absolute) max(floor, 0.22f * p995) else max(floor, max(kSigma * sigma, relFrac * p995))
        // 稀疏场景（整幅画面只有几个小点）p99.5 可能为 0，不能用它提前退出；
        // 只有「全图最大响应都低于阈值」才说明确实没有候选。
        if (response.max() < thr) return emptyList()

        // 只在局部极大处取候选：大块杂波里也能把每个小光点拆出来，避免整块连通域被面积过滤掉。
        val window = max(3, peakWindow or 1)
        val localMax = separableFilter(response, width, height, window, isMax = true)
        val mask = BooleanArray(response.size)
        for (j in response.indices) mask[j] = response[j] >= thr && response[j] >= localMax[j]

        val visited = BooleanArray(response.size)
        val queue = IntArray(response.size)
        val spots = ArrayList<FarSpot>()
        val maxArea = (maxAreaFrac * width * height).toInt().coerceAtLeast(4)
        for (start in response.indices) {
            if (!mask[start] || visited[start]) continue
            var head = 0
            var tail = 0
            queue[tail++] = start
            visited[start] = true
            var sumW = 0f
            var sumX = 0f
            var sumY = 0f
            var peak = 0f
            var minX = width
            var maxX = 0
            var minY = height
            var maxY = 0
            var area = 0
            while (head < tail) {
                val idx = queue[head++]
                val x = idx % width
                val y = idx / width
                val wgt = response[idx]
                sumW += wgt
                sumX += wgt * x
                sumY += wgt * y
                if (wgt > peak) peak = wgt
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
                area++
                for (dy in -1..1) {
                    val ny = y + dy
                    if (ny < 0 || ny >= height) continue
                    for (dx in -1..1) {
                        val nx = x + dx
                        if (nx < 0 || nx >= width) continue
                        val nIdx = ny * width + nx
                        if (mask[nIdx] && !visited[nIdx]) {
                            visited[nIdx] = true
                            queue[tail++] = nIdx
                        }
                    }
                }
            }
            if (area < minArea || area > maxArea) continue
            val bw = (maxX - minX + 1).toFloat()
            val bh = (maxY - minY + 1).toFloat()
            val aspect = max(bw, bh) / max(1f, min(bw, bh))
            if (aspect > maxAspect) continue
            if (sumW <= 0f) continue
            var u = sumX / sumW
            var v = sumY / sumW
            // C：小光斑（<=3px 外接框）做亚像素级细化。1–2px 的远距点整像素质心误差
            // 可达 0.5px，抛物线插值把它压到 0.1–0.3px（与 Python 侧 4x 上采样质心等价但更便宜）。
            if (max(bw, bh) <= 3.5f) {
                val refined = refineSubpixel(response, width, height, u, v)
                u = refined[0]
                v = refined[1]
            }
            // 圆度用「面积 / 外接框面积」近似（比周长公式稳，1–2px 点也不炸）
            val fill = (area / (bw * bh)).coerceIn(0f, 1f)
            val score = 0.55f * (peak / 255f) + 0.45f * fill
            spots.add(FarSpot(u, v, score, color, peak, area))
        }
        spots.sortByDescending { it.score }
        // 同色 20px 内去重
        val kept = ArrayList<FarSpot>()
        for (s in spots) {
            if (kept.size >= topKPerColor) break
            if (kept.none { abs(it.u - s.u) < 10f && abs(it.v - s.v) < 10f }) kept.add(s)
        }
        return kept
    }

    /**
     * 抛物线亚像素细化：对响应图在 (u,v) 的 3x3 邻域做逐轴二次插值，返回细化后的坐标。
     *
     * 只对小光斑启用（1–2px 点）；大光斑的峰值平台会让分母趋零，因此分母过小时直接跳过。
     * 位移夹在 ±0.5px，避免噪声帧上抛物线过冲。
     */
    private fun refineSubpixel(
        response: FloatArray,
        width: Int,
        height: Int,
        u: Float,
        v: Float
    ): FloatArray {
        val cx = u.roundToInt().coerceIn(1, width - 2)
        val cy = v.roundToInt().coerceIn(1, height - 2)
        val c = response[cy * width + cx]
        val left = response[cy * width + cx - 1]
        val right = response[cy * width + cx + 1]
        val up = response[(cy - 1) * width + cx]
        val down = response[(cy + 1) * width + cx]
        fun delta(a: Float, b: Float, center: Float): Float {
            val den = a - 2f * center + b
            if (abs(den) < 1e-4f) return 0f
            return (0.5f * (a - b) / den).coerceIn(-0.5f, 0.5f)
        }
        val dx = delta(left, right, c)
        val dy = delta(up, down, c)
        return floatArrayOf(u + dx, v + dy)
    }

    /**
     * 枚举三色候选的所有三元组，用「形状 + 亮度 + 视边长一致 + 上一帧预测」打分取最优。
     *
     * 只取每色 top1 的写法在杂波/泛光场景会被高分杂波带偏（真实帧实测选错 100px 以上）；
     * 打分沿用研究侧 classic_v2/classic_v3 的 `_score_triple`（去掉面积项，端侧不带 area 统计）：
     *   cost = shape + 0.5·|log(meanEdge/expectedEdge)| − 2.4·phot + 0.8·时序偏离
     * `predicted` 存在时（上一帧三点）用 soft 项，不硬拒绝——硬门限在真实帧上会直接无解。
     */
    private fun pickTriangle(
        candidates: List<FarSpot>,
        expectedEdgePx: Float,
        predicted: FloatArray?,
        validator: TripleValidator?
    ): List<FarSpot>? {
        val hasScale = expectedEdgePx > 1f
        val gate = if (hasScale) max(12f, 0.45f * expectedEdgePx) else 20f
        // 先用「上一帧预测」做候选关联，剪不掉再退回全量搜索：
        // 剪枝本身没有证据支持（Python A/B：只剪枝在 all_on 配置下 0.478→0.367），
        // 但退回语义保证它不会造成硬漏检，因此这里只当「加速/去噪的第一遍」。
        val gated = scanTriples(candidates, expectedEdgePx, predicted, validator, gate,
                                applyGate = predicted != null && predicted.size >= 6)
        if (gated != null && gated.second <= 2.2f) return gated.first
        val full = scanTriples(candidates, expectedEdgePx, predicted, validator, gate,
                               applyGate = false)
        return if (full != null && full.second <= 2.2f) full.first else null
    }

    /** Exhaustive triple search. Returns (triangle, cost) of the best candidate. */
    private fun scanTriples(
        candidates: List<FarSpot>,
        expectedEdgePx: Float,
        predicted: FloatArray?,
        validator: TripleValidator?,
        gate: Float,
        applyGate: Boolean
    ): Pair<List<FarSpot>, Float>? {
        val byColor = candidates.groupBy { it.color }
        val reds = byColor[LedColor.RED] ?: return null
        val greens = byColor[LedColor.GREEN] ?: return null
        val blues = byColor[LedColor.BLUE] ?: return null
        val hasScale = expectedEdgePx > 1f
        var bestCost = Float.MAX_VALUE
        var best: List<FarSpot>? = null
        for (r in reds) for (g in greens) for (b in blues) {
            if (applyGate && predicted != null && predicted.size >= 6) {
                if (dist(r, predicted[0], predicted[1]) > gate) continue
                if (dist(g, predicted[2], predicted[3]) > gate) continue
                if (dist(b, predicted[4], predicted[5]) > gate) continue
            }
            val e1 = dist(r, g)
            val e2 = dist(g, b)
            val e3 = dist(b, r)
            val mean = (e1 + e2 + e3) / 3f
            if (mean < 6f) continue
            val shape = (max(e1, max(e2, e3)) - min(e1, min(e2, e3))) / mean
            val area = abs((g.u - r.u) * (b.v - r.v) - (g.v - r.v) * (b.u - r.u)) / 2f
            if (area < 0.05f * mean * mean) continue
            val tri = listOf(r, g, b)
            var cost = 0.5f * shape
            if (hasScale) cost += 0.5f * abs(ln(mean / expectedEdgePx))
            if (predicted != null && predicted.size >= 6) {
                val temporal =
                    min(3f, dist(r, predicted[0], predicted[1]) / gate) +
                        min(3f, dist(g, predicted[2], predicted[3]) / gate) +
                        min(3f, dist(b, predicted[4], predicted[5]) / gate)
                cost += 0.6f * temporal / 3f
            }
            if (validator != null) {
                val poseCost = validator.validate(tri) ?: continue
                cost += 0.5f * poseCost
            }
            if (cost < bestCost) {
                bestCost = cost
                best = tri
            }
        }
        return if (best != null) best to bestCost else null
    }

    private fun dist(spot: FarSpot, u: Float, v: Float): Float {
        val dx = spot.u - u
        val dy = spot.v - v
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    private fun dist(a: FarSpot, b: FarSpot): Float {
        val dx = a.u - b.u
        val dy = a.v - b.v
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }
}

/** 颜色差图的平均强度（用于判断当前画面是否值得跑远距辅助）。 */
internal fun chromaStrength(pixels: IntArray): Float {
    if (pixels.isEmpty()) return 0f
    var sum = 0f
    var count = 0
    var i = 0
    while (i < pixels.size) {
        val p = pixels[i]
        val r = ((p shr 16) and 0xFF).toFloat()
        val g = ((p shr 8) and 0xFF).toFloat()
        val b = (p and 0xFF).toFloat()
        sum += max(0f, max(r - max(g, b), max(g - max(r, b), b - max(r, g))))
        count++
        i += 97
    }
    return if (count == 0) 0f else sum / count
}
