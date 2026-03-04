package org.example.tridotpnp

import android.graphics.Bitmap
import androidx.compose.ui.geometry.Offset
import androidx.core.graphics.get

enum class LedColor {
    RED, GREEN, BLUE, UNKNOWN
}


data class BrightSpot(
    val position: Offset,
    val brightness: Float,  // 颜色强度
    val color: LedColor = LedColor.UNKNOWN,  // LED颜色
    val redIntensity: Float = 0f,    // 红色强度
    val greenIntensity: Float = 0f,  // 绿色强度
    val blueIntensity: Float = 0f    // 蓝色强度
)

/**
 * 颜色校准数据
 */

data class ColorCalibration(
    val redColor: Triple<Float, Float, Float>,    // 红色LED的实际RGB值
    val greenColor: Triple<Float, Float, Float>,  // 绿色LED的实际RGB值
    val blueColor: Triple<Float, Float, Float>    // 蓝色LED的实际RGB值
)

class BrightSpotDetector {
    private var pixelBuffer: IntArray? = null
    private var bufferW = 0
    private var bufferH = 0

    // 颜色校准数据（可选）
    private var calibration: ColorCalibration? = null

    // 检测参数 - 可调整以检测微小LED
    var minPixelBrightness: Float = 50f  // 最小像素亮度（原100f）
    var minTotalBrightness: Float = 60f  // 最小总亮度（原120f）
    var dynamicThresholdMin: Float = 50f  // 动态阈值最小值（原100f）
    var minPixelCount: Int = 1  // 最小像素数（原3）
    var minRefineRadius: Int = 2  // 最小细化半径（原6）

    private fun obtainPixelBuffer(w: Int, h: Int): IntArray {
        val neededSize = w * h
        // 如果当前缓存为空或缓存大小不匹配，则重新分配
        if (pixelBuffer == null || pixelBuffer!!.size < neededSize || bufferW != w || bufferH != h) {
            bufferW = w
            bufferH = h
            pixelBuffer = IntArray(neededSize)
        }
        return pixelBuffer!!
    }


    /**
     * 设置颜色校准数据
     */
    fun setCalibration(calibrationData: ColorCalibration?) {
        calibration = calibrationData
    }

    /**
     * 检测图像中的彩色LED（红、绿、蓝）
     * @param bitmap 要分析的图像
     * @param threshold 颜色强度阈值 (0-255)
     * @param gridSize 网格大小，用于将图像分成小块进行分析
     * @param maxSpots 最多返回的亮点数量，null表示返回所有亮点
     * @param detectColoredLeds 是否检测彩色LED（true=RGB三色，false=仅绿色）
     * @param roiLeft ROI左边界（可选，用于限制搜索范围）
     * @param roiTop ROI上边界（可选，用于限制搜索范围）
     * @param roiRight ROI右边界（可选，用于限制搜索范围）
     * @param roiBottom ROI下边界（可选，用于限制搜索范围）
     * @return 检测到的彩色LED列表（按强度降序排列）
     */
    fun detectBrightSpots(
        bitmap: Bitmap,
        threshold: Float = 80f,
        gridSize: Int = 20,
        maxSpots: Int? = null,
        detectColoredLeds: Boolean = false,
        roiLeft: Int? = null,
        roiTop: Int? = null,
        roiRight: Int? = null,
        roiBottom: Int? = null
    ): List<BrightSpot> {

        val width = bitmap.width
        val height = bitmap.height
        if (width <= 1 || height <= 1) return emptyList()

        // ROI范围（或全图）
        val searchLeft = roiLeft?.coerceAtLeast(0) ?: 0
        val searchTop = roiTop?.coerceAtLeast(0) ?: 0
        val searchRight = (roiRight?.coerceAtMost(width - 1) ?: (width - 1)).coerceAtLeast(searchLeft)
        val searchBottom = (roiBottom?.coerceAtMost(height - 1) ?: (height - 1)).coerceAtLeast(searchTop)

        // 一次性读像素
        val pixels = obtainPixelBuffer(width, height)
        try {
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        } catch (_: Exception) {
            return emptyList()
        }

        // step 防止为0（必须）
        // ceil(width / gridSize)，避免右侧/底部漏扫
        val stepX = ((width + gridSize - 1) / gridSize).coerceAtLeast(1)
        val stepY = ((height + gridSize - 1) / gridSize).coerceAtLeast(1)

        // 计算网格范围（只遍历 ROI 覆盖到的格子）
        val startI = (searchLeft / stepX).coerceAtLeast(0)
        val endI = ((searchRight + stepX - 1) / stepX).coerceAtMost(gridSize - 1)
        val startJ = (searchTop / stepY).coerceAtLeast(0)
        val endJ = ((searchBottom + stepY - 1) / stepY).coerceAtMost(gridSize - 1)

        data class Candidate(
            val x: Int,
            val y: Int,
            val score: Float,   // 用于 Top-N/Top-K 的排序分数（允许与 raw 强度不同）
            val r: Float = 0f,  // raw 红强度
            val g: Float = 0f,  // raw 绿强度
            val b: Float = 0f   // raw 蓝强度
        )

        // 维护 Top-K（K不大时，这种 O(K) 更新非常快，避免全量 refine）
        fun pushTopK(list: MutableList<Candidate>, cand: Candidate, k: Int) {
            if (k <= 0) return
            if (list.size < k) {
                list.add(cand)
                return
            }
            var minIdx = 0
            var minVal = list[0].score
            for (i in 1 until list.size) {
                val v = list[i].score
                if (v < minVal) {
                    minVal = v
                    minIdx = i
                }
            }
            if (cand.score > minVal) list[minIdx] = cand
        }

        // refine 半径（你原逻辑）
        fun calcRefineRadius(): Int {
            val baseRadius = kotlin.math.max(stepX, stepY)
            return (baseRadius * 1.5f).toInt()
                .coerceAtLeast(minRefineRadius)
                .coerceAtMost(kotlin.math.min(width, height) / 4)
        }

        // -----------------------------
        // A) RGB三色 + maxSpots==3：Top-N 候选 + 三点几何“软惩罚” + 纯度“加分” 只 refine 3 次
        // -----------------------------
        if (detectColoredLeds && maxSpots == 3) {

            // ===== 可调参数（先用默认值）=====
            val topN = 25                              // 比 25 更稳，减少“刚好被挤出 TopN”的闪烁
            val base = kotlin.math.max(stepX, stepY).toFloat()

            // 几何范围（不再硬过滤，而是软惩罚）
            val minPairDist = base * 2f              // 允许更近一些，减少移动时掉线
            val maxPairDist = base * 20.0f             // 允许更远一些，减少移动时掉线
            val minPairDist2 = minPairDist * minPairDist
            val maxPairDist2 = maxPairDist * maxPairDist

            // 形状/几何惩罚权重
            val shapePenaltyW = 0.12f                  // 等边偏好（软）
            val geoPenaltyW = 0.6f                    // 超出距离范围的惩罚（软）

            // 纯度加分强度：越大越强调“颜色优势”
            val purityBoostW = 0.25f

            fun dist2(ax: Int, ay: Int, bx: Int, by: Int): Float {
                val dx = (ax - bx).toFloat()
                val dy = (ay - by).toFloat()
                return dx * dx + dy * dy
            }

            // 距离范围惩罚：在范围内=0；范围外按相对超出比例增加惩罚
            fun rangePenalty(d2: Float, min2: Float, max2: Float): Float {
                return when {
                    d2 < min2 -> (min2 - d2) / (min2 + 1e-6f)
                    d2 > max2 -> (d2 - max2) / (max2 + 1e-6f)
                    else -> 0f
                }
            }

            // Top-N（按 score）维护
            fun pushTopN(list: MutableList<Candidate>, cand: Candidate, n: Int) {
                if (n <= 0) return
                if (list.size < n) {
                    list.add(cand)
                    return
                }
                var minIdx = 0
                var minVal = list[0].score
                for (i in 1 until list.size) {
                    val v = list[i].score
                    if (v < minVal) {
                        minVal = v
                        minIdx = i
                    }
                }
                if (cand.score > minVal) list[minIdx] = cand
            }

            // 纯度加分：用 raw 强度计算颜色优势，输出一个“用于排序”的分数
            // 说明：不再用 purityRatio 硬门槛，否则一抖就掉候选 -> 鬼畜
            fun redScore(rI: Float, gI: Float, bI: Float): Float {
                val denom = (gI + bI + 1e-3f)
                val purity = rI / denom          // 越大越“红”
                return rI * (1f + purityBoostW * kotlin.math.ln(1f + purity))
            }

            fun greenScore(rI: Float, gI: Float, bI: Float): Float {
                val denom = (rI + bI + 1e-3f)
                val purity = gI / denom
                return gI * (1f + purityBoostW * kotlin.math.ln(1f + purity))
            }

            fun blueScore(rI: Float, gI: Float, bI: Float): Float {
                val denom = (rI + gI + 1e-3f)
                val purity = bI / denom
                return bI * (1f + purityBoostW * kotlin.math.ln(1f + purity))
            }

            val reds = mutableListOf<Candidate>()
            val greens = mutableListOf<Candidate>()
            val blues = mutableListOf<Candidate>()

            for (i in startI..endI) {
                val cx = (i * stepX + stepX / 2).coerceIn(0, width - 1)
                for (j in startJ..endJ) {
                    val cy = (j * stepY + stepY / 2).coerceIn(0, height - 1)

                    val intensities = calculateRGBIntensitiesFast(
                        pixels = pixels,
                        bmpWidth = width,
                        bmpHeight = height,
                        centerX = cx,
                        centerY = cy,
                        radiusX = stepX / 2,
                        radiusY = stepY / 2,
                        calibration = calibration
                    ) ?: continue

                    val rI = intensities.first
                    val gI = intensities.second
                    val bI = intensities.third

                    // 基础阈值：至少要有一个通道足够强
                    val best = maxOf(rI, gI, bI)
                    if (best <= threshold) continue

                    // 不做“纯度硬门槛”，而是三路都允许进入 Top-N，
                    // 由 score（含纯度加分）决定谁更像“真正的红/绿/蓝点”
                    pushTopN(reds, Candidate(cx, cy, score = redScore(rI, gI, bI), r = rI, g = gI, b = bI), topN)
                    pushTopN(greens, Candidate(cx, cy, score = greenScore(rI, gI, bI), r = rI, g = gI, b = bI), topN)
                    pushTopN(blues, Candidate(cx, cy, score = blueScore(rI, gI, bI), r = rI, g = gI, b = bI), topN)
                }
            }

            if (reds.isEmpty() || greens.isEmpty() || blues.isEmpty()) return emptyList()

            var bestTripleR: Candidate? = null
            var bestTripleG: Candidate? = null
            var bestTripleB: Candidate? = null
            var bestGroupScore = Float.NEGATIVE_INFINITY

            for (rC in reds) {
                for (gC in greens) {
                    val dRG2 = dist2(rC.x, rC.y, gC.x, gC.y)

                    for (bC in blues) {
                        val dRB2 = dist2(rC.x, rC.y, bC.x, bC.y)
                        val dGB2 = dist2(gC.x, gC.y, bC.x, bC.y)

                        // === A 改进：几何范围不硬过滤，而是软惩罚 ===
                        val geoPenalty =
                            rangePenalty(dRG2, minPairDist2, maxPairDist2) +
                                    rangePenalty(dRB2, minPairDist2, maxPairDist2) +
                                    rangePenalty(dGB2, minPairDist2, maxPairDist2)

                        // 形状惩罚：鼓励三边长度相近（仍是软）
                        val dRG = kotlin.math.sqrt(dRG2)
                        val dRB = kotlin.math.sqrt(dRB2)
                        val dGB = kotlin.math.sqrt(dGB2)
                        val mean = (dRG + dRB + dGB) / 3f
                        val shapePenalty = if (mean > 1e-3f) {
                            (kotlin.math.abs(dRG - mean) + kotlin.math.abs(dRB - mean) + kotlin.math.abs(dGB - mean)) / mean
                        } else 1f

                        // 颜色分数：用 raw 强度（更符合真实亮度），不要用带纯度加分的 score
                        val colorScore = rC.r + gC.g + bC.b

                        val groupScore =
                            colorScore -
                                    shapePenaltyW * colorScore * shapePenalty -
                                    geoPenaltyW * colorScore * geoPenalty

                        if (groupScore > bestGroupScore) {
                            bestGroupScore = groupScore
                            bestTripleR = rC
                            bestTripleG = gC
                            bestTripleB = bC
                        }
                    }
                }
            }

            if (bestTripleR == null || bestTripleG == null || bestTripleB == null) return emptyList()

            // 只 refine 这 3 个
            val refineRadius = calcRefineRadius()
            val result = ArrayList<BrightSpot>(3)

            fun addRefined(c: Candidate?, color: LedColor) {
                if (c == null) return
                val refined = refineSpotCenterIterativeFast(
                    pixels = pixels,
                    bmpWidth = width,
                    bmpHeight = height,
                    centerX = c.x,
                    centerY = c.y,
                    radius = refineRadius,
                    preferChannel = color
                )
                val brightness = when (color) {
                    LedColor.RED -> c.r
                    LedColor.GREEN -> c.g
                    LedColor.BLUE -> c.b
                    else -> c.score
                }
                result.add(
                    BrightSpot(
                        position = refined,
                        brightness = brightness,
                        color = color,
                        redIntensity = c.r,
                        greenIntensity = c.g,
                        blueIntensity = c.b
                    )
                )
            }

            addRefined(bestTripleR, LedColor.RED)
            addRefined(bestTripleG, LedColor.GREEN)
            addRefined(bestTripleB, LedColor.BLUE)
            return result
        }

        // -----------------------------
        // B) 其它情况：先粗扫候选（阈值 or Top-K），再只对候选 refine
        // -----------------------------
        val candidates = mutableListOf<Candidate>()

        // Top-K 的 K 值：maxSpots 越小，K 给稍大，保证稳定；上限别太大
        val k = when {
            maxSpots == null -> 0                      // 阈值模式，不用Top-K
            maxSpots <= 0 -> 0
            maxSpots <= 3 -> 48
            else -> (maxSpots * 8).coerceAtMost(256)
        }

        for (i in startI..endI) {
            val cx = (i * stepX + stepX / 2).coerceIn(0, width - 1)
            for (j in startJ..endJ) {
                val cy = (j * stepY + stepY / 2).coerceIn(0, height - 1)

                if (detectColoredLeds) {
                    val intensities = calculateRGBIntensitiesFast(
                        pixels = pixels,
                        bmpWidth = width,
                        bmpHeight = height,
                        centerX = cx,
                        centerY = cy,
                        radiusX = stepX / 2,
                        radiusY = stepY / 2,
                        calibration = calibration
                    ) ?: continue

                    val rI = intensities.first
                    val gI = intensities.second
                    val bI = intensities.third
                    val score = maxOf(rI, gI, bI)

                    if (maxSpots == null) {
                        // 阈值模式：只有超过阈值才 refine
                        if (score > threshold) {
                            candidates.add(Candidate(cx, cy, score, rI, gI, bI))
                        }
                    } else {
                        // Top-K：保留最强 K 个
                        pushTopK(candidates, Candidate(cx, cy, score, rI, gI, bI), k)
                    }
                } else {
                    val greenIntensity = calculateRegionGreenIntensityFast(
                        pixels = pixels,
                        bmpWidth = width,
                        bmpHeight = height,
                        centerX = cx,
                        centerY = cy,
                        radiusX = stepX / 2,
                        radiusY = stepY / 2
                    )

                    if (maxSpots == null) {
                        if (greenIntensity > threshold) {
                            candidates.add(Candidate(cx, cy, greenIntensity))
                        }
                    } else {
                        pushTopK(candidates, Candidate(cx, cy, greenIntensity), k)
                    }
                }
            }
        }

        if (candidates.isEmpty()) return emptyList()

        // 只 refine 候选
        val refineRadius = calcRefineRadius()
        val refinedSpots = ArrayList<BrightSpot>(candidates.size)

        for (c in candidates) {
            if (detectColoredLeds) {
                val prefer = when {
                    c.r >= c.g && c.r >= c.b -> LedColor.RED
                    c.g >= c.r && c.g >= c.b -> LedColor.GREEN
                    else -> LedColor.BLUE
                }
                val refined = refineSpotCenterIterativeFast(
                    pixels = pixels,
                    bmpWidth = width,
                    bmpHeight = height,
                    centerX = c.x,
                    centerY = c.y,
                    radius = refineRadius,
                    preferChannel = prefer
                )
                refinedSpots.add(
                    BrightSpot(
                        position = refined,
                        brightness = c.score,
                        color = LedColor.UNKNOWN,
                        redIntensity = c.r,
                        greenIntensity = c.g,
                        blueIntensity = c.b
                    )
                )
            } else {
                val refined = refineSpotCenterIterativeFast(
                    pixels = pixels,
                    bmpWidth = width,
                    bmpHeight = height,
                    centerX = c.x,
                    centerY = c.y,
                    radius = refineRadius,
                    preferChannel = LedColor.GREEN
                )
                refinedSpots.add(
                    BrightSpot(
                        position = refined,
                        brightness = c.score,
                        color = LedColor.GREEN
                    )
                )
            }
        }

        // 小数量直接取 top（避免 merge 的 O(n^2)）
        if (!detectColoredLeds && maxSpots != null && maxSpots in 1..3) {
            return refinedSpots.sortedByDescending { it.brightness }.take(maxSpots)
        }

        // 常规 merge（你原逻辑）
        val minDistance = (stepX.coerceAtLeast(stepY) * 1.5f)
        val merged = mergeNearbySpots(refinedSpots, minDistance)
        val sorted = merged.sortedByDescending { it.brightness }
        return if (maxSpots != null && maxSpots > 0) sorted.take(maxSpots) else sorted
    }

    private fun calculateRegionGreenIntensityFast(
        pixels: IntArray,
        bmpWidth: Int,
        bmpHeight: Int,
        centerX: Int,
        centerY: Int,
        radiusX: Int,
        radiusY: Int
    ): Float {
        var totalGreenIntensity = 0f
        var pixelCount = 0
        val startX = (centerX - radiusX).coerceAtLeast(0)
        val endX = (centerX + radiusX).coerceAtMost(bmpWidth - 1)
        val startY = (centerY - radiusY).coerceAtLeast(0)
        val endY = (centerY + radiusY).coerceAtMost(bmpHeight - 1)
        for (y in startY..endY) {
            val rowBase = y * bmpWidth
            for (x in startX..endX) {
                val pixel = pixels[rowBase + x]
                val greenIntensity = calculatePixelGreenIntensity(pixel)
                totalGreenIntensity += greenIntensity
                pixelCount++
            }
        }
        return if (pixelCount > 0) totalGreenIntensity / pixelCount else 0f
    }

    /**
     * 计算单个像素的绿色强度
     * 方法：绿色值要明显高于红色和蓝色的平均值
     * 绿色强度 = Green - (Red + Blue) / 2
     * 这样可以突出显示真正的绿色，而不是白色（RGB都高）或其他颜色
     */
    private fun calculatePixelGreenIntensity(pixel: Int): Float {
        val red = (pixel shr 16) and 0xFF
        val green = (pixel shr 8) and 0xFF
        val blue = pixel and 0xFF

        // 计算绿色的相对强度
        // 绿色值必须明显高于红色和蓝色的平均值
        val otherAverage = (red + blue) / 2f
        val greenIntensity = green - otherAverage

        // 同时要求绿色值本身要足够高（至少50）
        // 避免检测到暗绿色或噪点
        return if (green >= 50 && greenIntensity > 0) {
            greenIntensity
        } else {
            0f
        }
    }

    /**
     * 计算区域的RGB三种颜色强度
     * @return Triple(红色强度, 绿色强度, 蓝色强度) 或 null（太暗）
     */
    private fun calculateRGBIntensities(
        bitmap: Bitmap,
        centerX: Int,
        centerY: Int,
        radiusX: Int,
        radiusY: Int
    ): Triple<Float, Float, Float>? {
        val startX = (centerX - radiusX).coerceAtLeast(0)
        val endX = (centerX + radiusX).coerceAtMost(bitmap.width - 1)
        val startY = (centerY - radiusY).coerceAtLeast(0)
        val endY = (centerY + radiusY).coerceAtMost(bitmap.height - 1)

        // 收集所有像素，只保留足够亮的像素（LED特征）
        val brightPixels = mutableListOf<Triple<Float, Float, Float>>()

        for (x in startX..endX) {
            for (y in startY..endY) {
                val pixel = bitmap[x, y]
                val r = ((pixel shr 16) and 0xFF).toFloat()
                val g = ((pixel shr 8) and 0xFF).toFloat()
                val b = (pixel and 0xFF).toFloat()

                // 只收集足够亮的像素，过滤背景
                val brightness = r + g + b
                if (brightness > minPixelBrightness) {
                    brightPixels.add(Triple(r, g, b))
                }
            }
        }

        // 如果亮像素太少，说明这个区域没有LED
        if (brightPixels.size < minPixelCount) return null

        // 取最亮的前30%像素的平均值
        // 这样可以排除边缘的暗像素，只关注LED中心
        val sortedByBrightness = brightPixels.sortedByDescending { it.first + it.second + it.third }
        val topCount = maxOf(3, (sortedByBrightness.size * 0.3).toInt())
        val topPixels = sortedByBrightness.take(topCount)

        var totalRed = 0f
        var totalGreen = 0f
        var totalBlue = 0f

        topPixels.forEach { (r, g, b) ->
            totalRed += r
            totalGreen += g
            totalBlue += b
        }

        val avgRed = totalRed / topPixels.size
        val avgGreen = totalGreen / topPixels.size
        val avgBlue = totalBlue / topPixels.size

        // 计算总亮度
        val totalBrightness = avgRed + avgGreen + avgBlue
        if (totalBrightness < minTotalBrightness) {  // 亮度要求，确保是LED
            return null
        }

        // 计算三种颜色的相对强度（归一化色度空间）
        val sum = avgRed + avgGreen + avgBlue
        val r = avgRed / sum
        val g = avgGreen / sum
        val b = avgBlue / sum

        // 计算每种颜色的纯度分数
        // 红色强度 = 红色占比 × 绝对亮度
        val redIntensity = r * avgRed

        // 绿色强度 = 绿色占比 × 绝对亮度
        val greenIntensity = g * avgGreen

        // 蓝色强度 = 蓝色占比 × 绝对亮度
        val blueIntensity = b * avgBlue

        return Triple(redIntensity, greenIntensity, blueIntensity)
    }

    private fun calculateRGBIntensitiesFast(
        pixels: IntArray,
        bmpWidth: Int,
        bmpHeight: Int,
        centerX: Int,
        centerY: Int,
        radiusX: Int,
        radiusY: Int,
        calibration: ColorCalibration?
    ): Triple<Float, Float, Float>? {
        val startX = (centerX - radiusX).coerceAtLeast(0)
        val endX = (centerX + radiusX).coerceAtMost(bmpWidth - 1)
        val startY = (centerY - radiusY).coerceAtLeast(0)
        val endY = (centerY + radiusY).coerceAtMost(bmpHeight - 1)
        var maxBrightness = 0f
        for (y in startY..endY) {
            val base = y * bmpWidth
            for (x in startX..endX) {
                val p = pixels[base + x]
                val r = ((p shr 16) and 0xFF).toFloat()
                val g = ((p shr 8) and 0xFF).toFloat()
                val b = (p and 0xFF).toFloat()
                val br = r + g + b
                if (br > maxBrightness) maxBrightness = br
            }
        }
        if (maxBrightness <= 0f) return null
        val dynThreshold = kotlin.math.max(dynamicThresholdMin, maxBrightness * 0.7f)
        var totalRed = 0f
        var totalGreen = 0f
        var totalBlue = 0f
        var count = 0
        for (y in startY..endY) {
            val base = y * bmpWidth
            for (x in startX..endX) {
                val p = pixels[base + x]
                val r = ((p shr 16) and 0xFF).toFloat()
                val g = ((p shr 8) and 0xFF).toFloat()
                val b = (p and 0xFF).toFloat()
                val br = r + g + b
                if (br >= dynThreshold) {
                    totalRed += r
                    totalGreen += g
                    totalBlue += b
                    count++
                }
            }
        }
        if (count < minPixelCount) return null
        val avgRed = totalRed / count
        val avgGreen = totalGreen / count
        val avgBlue = totalBlue / count
        val totalBrightness = avgRed + avgGreen + avgBlue
        if (totalBrightness < minTotalBrightness) return null
        if (calibration == null) {
            val sum = avgRed + avgGreen + avgBlue
            val r = avgRed / sum
            val g = avgGreen / sum
            val b = avgBlue / sum
            val redIntensity = r * avgRed
            val greenIntensity = g * avgGreen
            val blueIntensity = b * avgBlue
            return Triple(redIntensity, greenIntensity, blueIntensity)
        } else {
            val norm = avgRed + avgGreen + avgBlue
            val rNorm = avgRed / norm
            val gNorm = avgGreen / norm
            val bNorm = avgBlue / norm
            val (calR, calG, calB) = calibration.redColor
            val calRedNorm = calR + calG + calB
            val redDistance = kotlin.math.sqrt(
                (rNorm - calR / calRedNorm) * (rNorm - calR / calRedNorm) +
                        (gNorm - calG / calRedNorm) * (gNorm - calG / calRedNorm) +
                        (bNorm - calB / calRedNorm) * (bNorm - calB / calRedNorm)
            )
            val redSimilarity = kotlin.math.exp(-redDistance * 8f)
            val (calGR, calGG, calGB) = calibration.greenColor
            val calGreenNorm = calGR + calGG + calGB
            val greenDistance = kotlin.math.sqrt(
                (rNorm - calGR / calGreenNorm) * (rNorm - calGR / calGreenNorm) +
                        (gNorm - calGG / calGreenNorm) * (gNorm - calGG / calGreenNorm) +
                        (bNorm - calGB / calGreenNorm) * (bNorm - calGB / calGreenNorm)
            )
            val greenSimilarity = kotlin.math.exp(-greenDistance * 8f)
            val (calBR, calBG, calBB) = calibration.blueColor
            val calBlueNorm = calBR + calBG + calBB
            val blueDistance = kotlin.math.sqrt(
                (rNorm - calBR / calBlueNorm) * (rNorm - calBR / calBlueNorm) +
                        (gNorm - calBG / calBlueNorm) * (gNorm - calBG / calBlueNorm) +
                        (bNorm - calBB / calBlueNorm) * (bNorm - calBB / calBlueNorm)
            )
            val blueSimilarity = kotlin.math.exp(-blueDistance * 8f)
            return Triple(
                if (redSimilarity > 0.3f) redSimilarity * totalBrightness else 0f,
                if (greenSimilarity > 0.3f) greenSimilarity * totalBrightness else 0f,
                if (blueSimilarity > 0.3f) blueSimilarity * totalBrightness else 0f
            )
        }
    }

    /**
     * 使用强度加权质心细化亮点中心，支持按指定颜色通道加权。
     * 权重选择：
     *  - GREEN: g - (r + b)/2
     *  - RED:   r - (g + b)/2
     *  - BLUE:  b - (r + g)/2
     * 否则使用总亮度 r+g+b。权重小于0的像素被忽略。
     */
    private fun refineSpotCenterFast(
        pixels: IntArray,
        bmpWidth: Int,
        bmpHeight: Int,
        centerX: Int,
        centerY: Int,
        radius: Int,
        preferChannel: LedColor
    ): Offset {
        val startX = (centerX - radius).coerceAtLeast(0)
        val endX = (centerX + radius).coerceAtMost(bmpWidth - 1)
        val startY = (centerY - radius).coerceAtLeast(0)
        val endY = (centerY + radius).coerceAtMost(bmpHeight - 1)
        var sumW = 0f
        var sumX = 0f
        var sumY = 0f
        for (y in startY..endY) {
            val base = y * bmpWidth
            for (x in startX..endX) {
                val p = pixels[base + x]
                val r = ((p shr 16) and 0xFF).toFloat()
                val g = ((p shr 8) and 0xFF).toFloat()
                val b = (p and 0xFF).toFloat()
                val w = when (preferChannel) {
                    LedColor.GREEN -> (g - (r + b) / 2f)
                    LedColor.RED -> (r - (g + b) / 2f)
                    LedColor.BLUE -> (b - (r + g) / 2f)
                    else -> (r + g + b)
                }.coerceAtLeast(0f)
                if (w > 0f) {
                    sumW += w
                    sumX += w * x
                    sumY += w * y
                }
            }
        }
        return if (sumW > 0f) Offset(sumX / sumW, sumY / sumW) else Offset(centerX.toFloat(), centerY.toFloat())
    }

    /**
     * 两次迭代细化：先在初始中心做一次强度质心，再以新中心重复一次，
     * 对于亮斑半径大于区块尺寸时更稳健。
     */
    private fun refineSpotCenterIterativeFast(
        pixels: IntArray,
        bmpWidth: Int,
        bmpHeight: Int,
        centerX: Int,
        centerY: Int,
        radius: Int,
        preferChannel: LedColor
    ): Offset {
        val first = refineSpotCenterFast(pixels, bmpWidth, bmpHeight, centerX, centerY, radius, preferChannel)
        val cx = first.x.toInt().coerceIn(0, bmpWidth - 1)
        val cy = first.y.toInt().coerceIn(0, bmpHeight - 1)
        val second = refineSpotCenterFast(pixels, bmpWidth, bmpHeight, cx, cy, radius, preferChannel)
        return second
    }

    /**
     * 从所有候选点中选择最红、最绿、最蓝的三个点
     * 遍历整个画面，找到红色强度最大、绿色强度最大、蓝色强度最大的三个位置
     */
    private fun selectBestRGBPoints(spots: List<BrightSpot>): List<BrightSpot> {
        if (spots.isEmpty()) {
            return emptyList()
        }

        // 找出红色强度最大的点
        val redSpot = spots.maxByOrNull { it.redIntensity }
        val redPoint = redSpot?.copy(
            color = LedColor.RED,
            brightness = redSpot.redIntensity
        )

        // 找出绿色强度最大的点
        val greenSpot = spots.maxByOrNull { it.greenIntensity }
        val greenPoint = greenSpot?.copy(
            color = LedColor.GREEN,
            brightness = greenSpot.greenIntensity
        )

        // 找出蓝色强度最大的点
        val blueSpot = spots.maxByOrNull { it.blueIntensity }
        val bluePoint = blueSpot?.copy(
            color = LedColor.BLUE,
            brightness = blueSpot.blueIntensity
        )

        // 返回三个点（按红、绿、蓝顺序）
        val result = mutableListOf<BrightSpot>()
        if (redPoint != null) result.add(redPoint)
        if (greenPoint != null) result.add(greenPoint)
        if (bluePoint != null) result.add(bluePoint)

        return result
    }


    /**
     * 在圆形区域内搜索最红、最绿、最蓝的点作为校准颜色
     * 使用与检测算法完全相同的逻辑，确保一致性
     */
    fun findRGBColorsInCircle(
        bitmap: Bitmap,
        centerX: Int,
        centerY: Int,
        radius: Int
    ): Triple<Triple<Float, Float, Float>, Triple<Float, Float, Float>, Triple<Float, Float, Float>>? {

        if (bitmap.isRecycled) {
            android.util.Log.e("ColorSearch", "Bitmap已被回收，无法执行颜色搜索")
            return null
        }

        val w = bitmap.width
        val h = bitmap.height
        if (w <= 1 || h <= 1) return null

        // 一次性读像素（关键）
        val pixels = IntArray(w * h)
        try {
            bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        } catch (e: Exception) {
            android.util.Log.e("ColorSearch", "getPixels失败: ${e.message}")
            return null
        }

        var bestR = 0f
        var bestG = 0f
        var bestB = 0f
        var bestRPos: Pair<Int, Int>? = null
        var bestGPos: Pair<Int, Int>? = null
        var bestBPos: Pair<Int, Int>? = null

        val sampleStep = 10
        val sampleRadius = 15

        // 扫描圆内采样点
        val yStart = (centerY - radius).coerceAtLeast(sampleRadius)
        val yEnd = (centerY + radius).coerceAtMost(h - 1 - sampleRadius)
        val xStart0 = (centerX - radius).coerceAtLeast(sampleRadius)
        val xEnd0 = (centerX + radius).coerceAtMost(w - 1 - sampleRadius)

        val radiusSq = radius.toFloat() * radius.toFloat()

        for (y in yStart..yEnd step sampleStep) {
            if (bitmap.isRecycled) return null

            val dy = (y - centerY).toFloat()
            for (x in xStart0..xEnd0 step sampleStep) {
                val dx = (x - centerX).toFloat()
                if (dx * dx + dy * dy > radiusSq) continue

                val intensities = calculateRGBIntensitiesFast(
                    pixels = pixels,
                    bmpWidth = w,
                    bmpHeight = h,
                    centerX = x,
                    centerY = y,
                    radiusX = sampleRadius,
                    radiusY = sampleRadius,
                    calibration = null // 校准阶段先用原始判别更稳；你也可以传 calibration
                ) ?: continue

                val rI = intensities.first
                val gI = intensities.second
                val bI = intensities.third

                if (rI > bestR) {
                    bestR = rI
                    bestRPos = x to y
                }
                if (gI > bestG) {
                    bestG = gI
                    bestGPos = x to y
                }
                if (bI > bestB) {
                    bestB = bI
                    bestBPos = x to y
                }
            }
        }

        android.util.Log.d("ColorSearch", "搜索结果 - 红:$bestR 绿:$bestG 蓝:$bestB")

        if (bestRPos == null || bestGPos == null || bestBPos == null) {
            android.util.Log.e(
                "ColorSearch",
                "未找到完整RGB！红:${bestRPos != null} 绿:${bestGPos != null} 蓝:${bestBPos != null}"
            )
            return null
        }

        val redColor = sampleColorAt(bitmap, bestRPos.first, bestRPos.second, radius = 20)
        val greenColor = sampleColorAt(bitmap, bestGPos.first, bestGPos.second, radius = 20)
        val blueColor = sampleColorAt(bitmap, bestBPos.first, bestBPos.second, radius = 20)

        android.util.Log.d("ColorSearch", "找到的位置 - 红:$bestRPos 绿:$bestGPos 蓝:$bestBPos")
        return Triple(redColor, greenColor, blueColor)
    }

    /**
     * 采样指定位置的颜色值（用于校准）
     * 关键改进：只采样区域内最亮的像素，避免被背景色稀释
     */
    fun sampleColorAt(
        bitmap: Bitmap,
        x: Int,
        y: Int,
        radius: Int = 20
    ): Triple<Float, Float, Float> {
        if (bitmap.isRecycled) return Triple(0f, 0f, 0f)

        val w = bitmap.width
        val h = bitmap.height
        if (w <= 1 || h <= 1) return Triple(0f, 0f, 0f)

        val startX = (x - radius).coerceAtLeast(0)
        val endX = (x + radius).coerceAtMost(w - 1)
        val startY = (y - radius).coerceAtLeast(0)
        val endY = (y + radius).coerceAtMost(h - 1)

        // 先找最大亮度（只考虑亮像素）
        var maxBr = 0f
        for (py in startY..endY) {
            for (px in startX..endX) {
                val p = bitmap[px, py]
                val r = ((p ushr 16) and 0xFF).toFloat()
                val g = ((p ushr 8) and 0xFF).toFloat()
                val b = (p and 0xFF).toFloat()
                val br = r + g + b
                if (br > minPixelBrightness && br > maxBr) maxBr = br
            }
        }
        if (maxBr <= 0f) return Triple(0f, 0f, 0f)

        // 取“最亮 topPercent”的阈值（和你原来 top20% 的意图一致）
        val topPercent = 0.2f
        val brightGate = kotlin.math.max(minPixelBrightness, maxBr * (1f - topPercent))

        var sumR = 0f
        var sumG = 0f
        var sumB = 0f
        var count = 0

        for (py in startY..endY) {
            for (px in startX..endX) {
                val p = bitmap[px, py]
                val r = ((p ushr 16) and 0xFF).toFloat()
                val g = ((p ushr 8) and 0xFF).toFloat()
                val b = (p and 0xFF).toFloat()
                val br = r + g + b
                if (br >= brightGate) {
                    sumR += r
                    sumG += g
                    sumB += b
                    count++
                }
            }
        }

        return if (count > 0) {
            Triple(sumR / count, sumG / count, sumB / count)
        } else {
            Triple(0f, 0f, 0f)
        }
    }

    /**
     * 合并距离较近的绿色亮点
     */
    private fun mergeNearbySpots(
        spots: List<BrightSpot>,
        minDistance: Float = 50f
    ): List<BrightSpot> {
        if (spots.isEmpty() || spots.size == 1) return spots
        if (minDistance <= 0f) return spots

        // 用 minDistance 作为 cellSize，保证：两点距离 < minDistance => cell 坐标差 <= 1
        val cellSize = minDistance
        val invCell = 1f / cellSize
        val minDist2 = minDistance * minDistance

        // 轻量 IntList，避免 MutableList<Int> 产生大量装箱和对象
        class IntList(initialCapacity: Int = 8) {
            private var a = IntArray(initialCapacity)
            var size: Int = 0
                private set

            fun add(v: Int) {
                if (size == a.size) a = a.copyOf(a.size * 2)
                a[size++] = v
            }

            operator fun get(i: Int): Int = a[i]
        }

        fun key(cx: Int, cy: Int): Long {
            // pack two Int into one Long
            return (cx.toLong() shl 32) or (cy.toLong() and 0xffffffffL)
        }

        // 1) 建桶：cell -> indices
        val buckets = HashMap<Long, IntList>(spots.size * 2)
        for (i in spots.indices) {
            val p = spots[i].position
            // p.x/p.y 均为正坐标时 toInt 等价 floor
            val cx = (p.x * invCell).toInt()
            val cy = (p.y * invCell).toInt()
            val k = key(cx, cy)
            val list = buckets[k]
            if (list == null) {
                val nl = IntList()
                nl.add(i)
                buckets[k] = nl
            } else {
                list.add(i)
            }
        }

        // 2) 合并：只检查 seed 周围 3x3 cell，避免 O(n^2)
        val used = BooleanArray(spots.size)
        val merged = ArrayList<BrightSpot>(spots.size)

        for (i in spots.indices) {
            if (used[i]) continue

            val seed = spots[i]
            val sp = seed.position
            val seedCx = (sp.x * invCell).toInt()
            val seedCy = (sp.y * invCell).toInt()

            used[i] = true

            var sumX = sp.x
            var sumY = sp.y
            var count = 1

            // 用簇内“最亮”的 spot 当代表，保留它的 color / RGB 强度字段
            var bestIdx = i
            var bestBrightness = seed.brightness

            // 只扫周围 9 个格子
            for (cy in (seedCy - 1)..(seedCy + 1)) {
                for (cx in (seedCx - 1)..(seedCx + 1)) {
                    val list = buckets[key(cx, cy)] ?: continue
                    for (t in 0 until list.size) {
                        val j = list[t]
                        if (used[j]) continue

                        val p = spots[j].position
                        val dx = p.x - sp.x
                        val dy = p.y - sp.y
                        if (dx * dx + dy * dy <= minDist2) {
                            used[j] = true
                            sumX += p.x
                            sumY += p.y
                            count++

                            val br = spots[j].brightness
                            if (br > bestBrightness) {
                                bestBrightness = br
                                bestIdx = j
                            }
                        }
                    }
                }
            }

            val avgX = sumX / count
            val avgY = sumY / count

            val best = spots[bestIdx]

            // 保留 best 的颜色/强度，只更新位置和 brightness
            merged.add(best.copy(position = Offset(avgX, avgY), brightness = bestBrightness))
        }

        return merged
    }
    /**
     * 计算两点之间的距离
     */
    private fun calculateDistance(p1: Offset, p2: Offset): Float {
        val dx = p1.x - p2.x
        val dy = p1.y - p2.y
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }
}

