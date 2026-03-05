package org.example.tridotpnp

import android.graphics.Bitmap
import android.graphics.Color
import androidx.compose.ui.geometry.Offset
import androidx.core.graphics.get
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

enum class LedColor {
    RED, GREEN, BLUE, UNKNOWN
}

data class BrightSpot(
    val position: Offset,
    val brightness: Float,  // 颜色强度（这里指“与目标颜色相似度 × 亮度”）
    val color: LedColor = LedColor.UNKNOWN,  // LED颜色
    val redIntensity: Float = 0f,    // “目标1”(RED) 强度
    val greenIntensity: Float = 0f,  // “目标2”(GREEN) 强度
    val blueIntensity: Float = 0f    // “目标3”(BLUE) 强度
)

/**
 * 颜色校准数据（仍用 RGB 存储；内部会转换到 HSV 做相似度）
 * 注意：这里 red/green/blue 只是三个目标颜色的槽位名，不要求是三原色。
 */
data class ColorCalibration(
    val redColor: Triple<Float, Float, Float>,    // 目标1 的实际RGB值
    val greenColor: Triple<Float, Float, Float>,  // 目标2 的实际RGB值
    val blueColor: Triple<Float, Float, Float>    // 目标3 的实际RGB值
)

class BrightSpotDetector {
    private var pixelBuffer: IntArray? = null
    private var bufferW = 0
    private var bufferH = 0

    // 颜色校准数据（可选）
    private var calibration: ColorCalibration? = null

    // ===== HSV 目标缓存（避免每次都转）=====
    private val targetRedHSV = FloatArray(3)
    private val targetGreenHSV = FloatArray(3)
    private val targetBlueHSV = FloatArray(3)
    private var targetsPrepared = false

    // 检测参数 - 可调整以检测微小LED
    var minPixelBrightness: Float = 50f   // 最小像素亮度（用于 sampleColorAt 等处）
    var minTotalBrightness: Float = 60f   // 最小总亮度（区域平均RGB之和）
    var dynamicThresholdMin: Float = 50f  // 动态阈值最小值
    var minPixelCount: Int = 1            // 最小像素数
    var minRefineRadius: Int = 2          // 最小细化半径

    // 面积过滤：在区域里，超过 dynThreshold 的像素太多 => 大块亮色，过滤掉
    var maxBrightPixelRatio: Float = 0.18f   // 亮像素最多占区域面积的比例（0.10~0.30）
    var maxBrightPixelCountMin: Int = 12     // 小窗口时兜底的最大亮像素数上限（6~30）

    // ===== HSV 相似度参数（可调）=====
    var hsvMinSaturation: Float = 0.15f   // 低饱和（偏灰/白）直接弱化/过滤
    var hsvMinValue: Float = 0.10f        // 太暗直接弱化/过滤（HSV V）
    var hsvHueWeight: Float = 1.0f
    var hsvSatWeight: Float = 0.6f
    var hsvValWeight: Float = 0.3f
    var hsvSharpness: Float = 8.0f        // 越大越“挑剔”（相似度衰减更快）

    private fun obtainPixelBuffer(w: Int, h: Int): IntArray {
        val neededSize = w * h
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
        targetsPrepared = false
    }

    // ===== HSV 工具 =====

    private fun clamp255(v: Float): Int = v.toInt().coerceIn(0, 255)

    private fun rgbTripleToHSV(rgb: Triple<Float, Float, Float>, out: FloatArray) {
        Color.RGBToHSV(clamp255(rgb.first), clamp255(rgb.second), clamp255(rgb.third), out)
    }

    private fun ensureTargetsPrepared() {
        if (targetsPrepared) return

        val cal = calibration
        if (cal != null) {
            rgbTripleToHSV(cal.redColor, targetRedHSV)
            rgbTripleToHSV(cal.greenColor, targetGreenHSV)
            rgbTripleToHSV(cal.blueColor, targetBlueHSV)
        } else {
            // 没有校准时：给三组默认 HSV（只是兜底）
            // 注意：你的目标颜色不一定是三原色，所以强烈建议传 calibration
            targetRedHSV[0] = 0f;   targetRedHSV[1] = 1f; targetRedHSV[2] = 1f
            targetGreenHSV[0] = 120f; targetGreenHSV[1] = 1f; targetGreenHSV[2] = 1f
            targetBlueHSV[0] = 240f;  targetBlueHSV[1] = 1f; targetBlueHSV[2] = 1f
        }
        targetsPrepared = true
    }

    /**
     * HSV 相似度：返回 0..1
     * - Hue 采用圆周距离（0..180）归一化到 0..1
     * - S/V 直接差值（0..1）
     * - 通过 exp(-sharpness * weightedDistance) 得到平滑相似度
     */
    private fun hsvSimilarity(hsv: FloatArray, target: FloatArray): Float {
        val h1 = hsv[0]
        val s1 = hsv[1]
        val v1 = hsv[2]

        // 对灰/暗做快速抑制（避免白色反光/暗噪点）
        if (s1 < hsvMinSaturation || v1 < hsvMinValue) return 0f

        val h2 = target[0]
        val s2 = target[1]
        val v2 = target[2]

        var dh = abs(h1 - h2)
        if (dh > 180f) dh = 360f - dh
        val dhN = dh / 180f // 0..1
        val ds = abs(s1 - s2) // 0..1
        val dv = abs(v1 - v2) // 0..1

        val d = hsvHueWeight * (dhN * dhN) + hsvSatWeight * (ds * ds) + hsvValWeight * (dv * dv)
        return exp(-hsvSharpness * d).toFloat()
    }

    private fun getTargetHSV(color: LedColor): FloatArray {
        ensureTargetsPrepared()
        return when (color) {
            LedColor.RED -> targetRedHSV
            LedColor.GREEN -> targetGreenHSV
            LedColor.BLUE -> targetBlueHSV
            else -> targetGreenHSV
        }
    }

    // ===== 主检测 =====

    /**
     * 检测图像中的彩色LED（3个目标色：RED/GREEN/BLUE 槽位）
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

        // 准备 HSV 目标
        ensureTargetsPrepared()

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
            val score: Float,   // Top-N/Top-K 排序分数（允许与 raw 强度不同）
            val r: Float = 0f,  // 目标1 强度
            val g: Float = 0f,  // 目标2 强度
            val b: Float = 0f   // 目标3 强度
        )

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

        fun calcRefineRadius(): Int {
            val baseRadius = max(stepX, stepY)
            return (baseRadius * 1.5f).toInt()
                .coerceAtLeast(minRefineRadius)
                .coerceAtMost(min(width, height) / 4)
        }

        // -----------------------------
        // A) 三色 + maxSpots==3：Top-N 候选 + 三点几何软惩罚 + “纯度加分” 只 refine 3 次
        // -----------------------------
        if (detectColoredLeds && maxSpots == 3) {

            val topN = 25
            val base = max(stepX, stepY).toFloat()

            val minPairDist = base * 2f
            val maxPairDist = base * 20.0f
            val minPairDist2 = minPairDist * minPairDist
            val maxPairDist2 = maxPairDist * maxPairDist

            // 形状/几何惩罚权重
            val shapePenaltyW = 0.2f                  // 等边偏好（软）
            val geoPenaltyW = 0.6f                    // 超出距离范围的惩罚（软）

            // “纯度加分”：强调某个目标强度要明显胜过另外两个（不硬阈值，避免抖动掉候选）
            val purityBoostW = 0.25f

            fun dist2(ax: Int, ay: Int, bx: Int, by: Int): Float {
                val dx = (ax - bx).toFloat()
                val dy = (ay - by).toFloat()
                return dx * dx + dy * dy
            }

            fun rangePenalty(d2: Float, min2: Float, max2: Float): Float {
                return when {
                    d2 < min2 -> (min2 - d2) / (min2 + 1e-6f)
                    d2 > max2 -> (d2 - max2) / (max2 + 1e-6f)
                    else -> 0f
                }
            }

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

            fun redScore(rI: Float, gI: Float, bI: Float): Float {
                val denom = (gI + bI + 1e-3f)
                val purity = rI / denom
                return rI * (1f + purityBoostW * ln(1f + purity))
            }

            fun greenScore(rI: Float, gI: Float, bI: Float): Float {
                val denom = (rI + bI + 1e-3f)
                val purity = gI / denom
                return gI * (1f + purityBoostW * ln(1f + purity))
            }

            fun blueScore(rI: Float, gI: Float, bI: Float): Float {
                val denom = (rI + gI + 1e-3f)
                val purity = bI / denom
                return bI * (1f + purityBoostW * ln(1f + purity))
            }

            val reds = mutableListOf<Candidate>()
            val greens = mutableListOf<Candidate>()
            val blues = mutableListOf<Candidate>()

            for (i in startI..endI) {
                val cx = (i * stepX + stepX / 2).coerceIn(0, width - 1)
                for (j in startJ..endJ) {
                    val cy = (j * stepY + stepY / 2).coerceIn(0, height - 1)

                    val intensities = calculateHSVIntensitiesFast(
                        pixels = pixels,
                        bmpWidth = width,
                        bmpHeight = height,
                        centerX = cx,
                        centerY = cy,
                        radiusX = stepX / 2,
                        radiusY = stepY / 2
                    ) ?: continue

                    val rI = intensities.first
                    val gI = intensities.second
                    val bI = intensities.third

                    val best = maxOf(rI, gI, bI)
                    if (best <= threshold) continue

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

                        val geoPenalty =
                            rangePenalty(dRG2, minPairDist2, maxPairDist2) +
                                    rangePenalty(dRB2, minPairDist2, maxPairDist2) +
                                    rangePenalty(dGB2, minPairDist2, maxPairDist2)

                        val dRG = sqrt(dRG2)
                        val dRB = sqrt(dRB2)
                        val dGB = sqrt(dGB2)
                        val mean = (dRG + dRB + dGB) / 3f
                        val shapePenalty = if (mean > 1e-3f) {
                            (abs(dRG - mean) + abs(dRB - mean) + abs(dGB - mean)) / mean
                        } else 1f

                        // 颜色分数：用 raw 强度（相似度×亮度）
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
                val refined = refineSpotCenterIterativeFastHSV(
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

        val k = when {
            maxSpots == null -> 0
            maxSpots <= 0 -> 0
            maxSpots <= 3 -> 48
            else -> (maxSpots * 8).coerceAtMost(256)
        }

        for (i in startI..endI) {
            val cx = (i * stepX + stepX / 2).coerceIn(0, width - 1)
            for (j in startJ..endJ) {
                val cy = (j * stepY + stepY / 2).coerceIn(0, height - 1)

                if (detectColoredLeds) {
                    val intensities = calculateHSVIntensitiesFast(
                        pixels = pixels,
                        bmpWidth = width,
                        bmpHeight = height,
                        centerX = cx,
                        centerY = cy,
                        radiusX = stepX / 2,
                        radiusY = stepY / 2
                    ) ?: continue

                    val rI = intensities.first
                    val gI = intensities.second
                    val bI = intensities.third
                    val score = maxOf(rI, gI, bI)

                    if (maxSpots == null) {
                        if (score > threshold) {
                            candidates.add(Candidate(cx, cy, score, rI, gI, bI))
                        }
                    } else {
                        pushTopK(candidates, Candidate(cx, cy, score, rI, gI, bI), k)
                    }
                } else {
                    // 绿色模式仍保留你原来的“绿相对强度”逻辑
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

        val refineRadius = calcRefineRadius()
        val refinedSpots = ArrayList<BrightSpot>(candidates.size)

        for (c in candidates) {
            if (detectColoredLeds) {
                val prefer = when {
                    c.r >= c.g && c.r >= c.b -> LedColor.RED
                    c.g >= c.r && c.g >= c.b -> LedColor.GREEN
                    else -> LedColor.BLUE
                }
                val refined = refineSpotCenterIterativeFastHSV(
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
                val refined = refineSpotCenterIterativeFastRGB(
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

        if (!detectColoredLeds && maxSpots != null && maxSpots in 1..3) {
            return refinedSpots.sortedByDescending { it.brightness }.take(maxSpots)
        }

        val minDistance = (stepX.coerceAtLeast(stepY) * 1.5f)
        val merged = mergeNearbySpots(refinedSpots, minDistance)
        val sorted = merged.sortedByDescending { it.brightness }
        return if (maxSpots != null && maxSpots > 0) sorted.take(maxSpots) else sorted
    }

    // ===== HSV 版区域强度：返回 (目标1, 目标2, 目标3) 强度 =====
    private fun calculateHSVIntensitiesFast(
        pixels: IntArray,
        bmpWidth: Int,
        bmpHeight: Int,
        centerX: Int,
        centerY: Int,
        radiusX: Int,
        radiusY: Int
    ): Triple<Float, Float, Float>? {
        val startX = (centerX - radiusX).coerceAtLeast(0)
        val endX = (centerX + radiusX).coerceAtMost(bmpWidth - 1)
        val startY = (centerY - radiusY).coerceAtLeast(0)
        val endY = (centerY + radiusY).coerceAtMost(bmpHeight - 1)

        // 1) 找区域内最大亮度（RGB和）
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

        // 2) 动态阈值：只取最亮的一撮像素
        val dynThreshold = max(dynamicThresholdMin, maxBrightness * 0.7f)

        var totalR = 0f
        var totalG = 0f
        var totalB = 0f
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
                    totalR += r
                    totalG += g
                    totalB += b
                    count++
                }
            }
        }

        if (count < minPixelCount) return null

        // 3) 面积过滤：亮像素太多 => 大块亮色
        val regionW = (endX - startX + 1)
        val regionH = (endY - startY + 1)
        val regionArea = regionW * regionH
        val maxAllowed = max(
            maxBrightPixelCountMin,
            (regionArea * maxBrightPixelRatio).toInt()
        )
        if (count > maxAllowed) return null

        // 4) 区域平均颜色 + 总亮度门槛
        val avgR = totalR / count
        val avgG = totalG / count
        val avgB = totalB / count
        val totalBrightness = avgR + avgG + avgB
        if (totalBrightness < minTotalBrightness) return null

        // 5) 转 HSV，算到三个目标颜色的相似度，再乘亮度得到“强度”
        val hsv = FloatArray(3)
        Color.RGBToHSV(clamp255(avgR), clamp255(avgG), clamp255(avgB), hsv)

        val simR = hsvSimilarity(hsv, targetRedHSV)
        val simG = hsvSimilarity(hsv, targetGreenHSV)
        val simB = hsvSimilarity(hsv, targetBlueHSV)

        return Triple(simR * totalBrightness, simG * totalBrightness, simB * totalBrightness)
    }

    // ===== 绿色模式保留（你的原逻辑）=====
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

    private fun calculatePixelGreenIntensity(pixel: Int): Float {
        val red = (pixel shr 16) and 0xFF
        val green = (pixel shr 8) and 0xFF
        val blue = pixel and 0xFF
        val otherAverage = (red + blue) / 2f
        val greenIntensity = green - otherAverage
        return if (green >= 50 && greenIntensity > 0) greenIntensity else 0f
    }

    // ===== refine：HSV 版（按目标颜色相似度做权重质心）=====
    private fun refineSpotCenterFastHSV(
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

        val target = if (preferChannel == LedColor.UNKNOWN) null else getTargetHSV(preferChannel)
        val hsv = FloatArray(3)

        var sumW = 0f
        var sumX = 0f
        var sumY = 0f

        for (y in startY..endY) {
            val base = y * bmpWidth
            for (x in startX..endX) {
                val p = pixels[base + x]
                val r = ((p shr 16) and 0xFF)
                val g = ((p shr 8) and 0xFF)
                val b = (p and 0xFF)

                Color.RGBToHSV(r, g, b, hsv)

                // 权重：相似度 × 亮度（HSV V）
                val w = if (target != null) {
                    val sim = hsvSimilarity(hsv, target)
                    sim * hsv[2] // v in 0..1
                } else {
                    // UNKNOWN：用亮度权重（避免全丢）
                    hsv[2]
                }

                if (w > 0f) {
                    sumW += w
                    sumX += w * x
                    sumY += w * y
                }
            }
        }

        return if (sumW > 0f) Offset(sumX / sumW, sumY / sumW) else Offset(centerX.toFloat(), centerY.toFloat())
    }

    private fun refineSpotCenterIterativeFastHSV(
        pixels: IntArray,
        bmpWidth: Int,
        bmpHeight: Int,
        centerX: Int,
        centerY: Int,
        radius: Int,
        preferChannel: LedColor
    ): Offset {
        val first = refineSpotCenterFastHSV(pixels, bmpWidth, bmpHeight, centerX, centerY, radius, preferChannel)
        val cx = first.x.toInt().coerceIn(0, bmpWidth - 1)
        val cy = first.y.toInt().coerceIn(0, bmpHeight - 1)
        return refineSpotCenterFastHSV(pixels, bmpWidth, bmpHeight, cx, cy, radius, preferChannel)
    }

    // ===== refine：RGB 版（仅给绿色模式沿用）=====
    private fun refineSpotCenterFastRGB(
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

    private fun refineSpotCenterIterativeFastRGB(
        pixels: IntArray,
        bmpWidth: Int,
        bmpHeight: Int,
        centerX: Int,
        centerY: Int,
        radius: Int,
        preferChannel: LedColor
    ): Offset {
        val first = refineSpotCenterFastRGB(pixels, bmpWidth, bmpHeight, centerX, centerY, radius, preferChannel)
        val cx = first.x.toInt().coerceIn(0, bmpWidth - 1)
        val cy = first.y.toInt().coerceIn(0, bmpHeight - 1)
        return refineSpotCenterFastRGB(pixels, bmpWidth, bmpHeight, cx, cy, radius, preferChannel)
    }

    // =====（可选）旧的 selectBestRGBPoints 保留：字段名仍叫 red/green/blue，但语义是“目标1/2/3”=====
    private fun selectBestRGBPoints(spots: List<BrightSpot>): List<BrightSpot> {
        if (spots.isEmpty()) return emptyList()

        val redSpot = spots.maxByOrNull { it.redIntensity }
        val redPoint = redSpot?.copy(color = LedColor.RED, brightness = redSpot.redIntensity)

        val greenSpot = spots.maxByOrNull { it.greenIntensity }
        val greenPoint = greenSpot?.copy(color = LedColor.GREEN, brightness = greenSpot.greenIntensity)

        val blueSpot = spots.maxByOrNull { it.blueIntensity }
        val bluePoint = blueSpot?.copy(color = LedColor.BLUE, brightness = blueSpot.blueIntensity)

        val result = mutableListOf<BrightSpot>()
        if (redPoint != null) result.add(redPoint)
        if (greenPoint != null) result.add(greenPoint)
        if (bluePoint != null) result.add(bluePoint)
        return result
    }

    /**
     * 在圆形区域内搜索三个目标颜色的最强点（用于校准或辅助）
     * - 若 calibration 已设置：按 HSV 相似度直接找三个目标
     * - 若 calibration 未设置：退化为默认三目标（近似 RGB 三原色，仅兜底）
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

        ensureTargetsPrepared()

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

                val intensities = calculateHSVIntensitiesFast(
                    pixels = pixels,
                    bmpWidth = w,
                    bmpHeight = h,
                    centerX = x,
                    centerY = y,
                    radiusX = sampleRadius,
                    radiusY = sampleRadius
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

        android.util.Log.d("ColorSearch", "搜索结果(HSV) - 目标1:$bestR 目标2:$bestG 目标3:$bestB")

        if (bestRPos == null || bestGPos == null || bestBPos == null) {
            android.util.Log.e(
                "ColorSearch",
                "未找到完整三色！1:${bestRPos != null} 2:${bestGPos != null} 3:${bestBPos != null}"
            )
            return null
        }

        val c1 = sampleColorAt(bitmap, bestRPos.first, bestRPos.second, radius = 20)
        val c2 = sampleColorAt(bitmap, bestGPos.first, bestGPos.second, radius = 20)
        val c3 = sampleColorAt(bitmap, bestBPos.first, bestBPos.second, radius = 20)

        android.util.Log.d("ColorSearch", "找到的位置 - 1:$bestRPos 2:$bestGPos 3:$bestBPos")
        return Triple(c1, c2, c3)
    }

    /**
     * 采样指定位置的颜色值（用于校准）
     * 只采样区域内最亮的像素，避免被背景色稀释
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

        val topPercent = 0.2f
        val brightGate = max(minPixelBrightness, maxBr * (1f - topPercent))

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

        return if (count > 0) Triple(sumR / count, sumG / count, sumB / count) else Triple(0f, 0f, 0f)
    }

    /**
     * 合并距离较近的亮点（保持你原逻辑）
     */
    private fun mergeNearbySpots(
        spots: List<BrightSpot>,
        minDistance: Float = 50f
    ): List<BrightSpot> {
        if (spots.isEmpty() || spots.size == 1) return spots
        if (minDistance <= 0f) return spots

        val cellSize = minDistance
        val invCell = 1f / cellSize
        val minDist2 = minDistance * minDistance

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
            return (cx.toLong() shl 32) or (cy.toLong() and 0xffffffffL)
        }

        val buckets = HashMap<Long, IntList>(spots.size * 2)
        for (i in spots.indices) {
            val p = spots[i].position
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

            var bestIdx = i
            var bestBrightness = seed.brightness

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
            merged.add(best.copy(position = Offset(avgX, avgY), brightness = bestBrightness))
        }

        return merged
    }
}