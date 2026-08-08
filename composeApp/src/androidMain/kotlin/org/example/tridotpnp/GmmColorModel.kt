package org.example.tridotpnp

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp

/**
 * Per-color 2-component GMM soft segmentation in HSV.
 * Used as a probability prior over the existing HSV similarity path.
 */
data class GaussianHsv(
    val meanH: Float,
    val meanS: Float,
    val meanV: Float,
    val sigmaH: Float,
    val sigmaS: Float,
    val sigmaV: Float,
    val weight: Float
) {
    fun logPdf(h: Float, s: Float, v: Float): Float {
        var dh = abs(h - meanH)
        if (dh > 180f) dh = 360f - dh
        val ds = s - meanS
        val dv = v - meanV
        val sh = sigmaH.coerceAtLeast(1f)
        val ss = sigmaS.coerceAtLeast(0.02f)
        val sv = sigmaV.coerceAtLeast(0.02f)
        val quad = (dh * dh) / (sh * sh) + (ds * ds) / (ss * ss) + (dv * dv) / (sv * sv)
        val norm = ln2Pi + ln(sh) + ln(ss) + ln(sv)
        return -0.5f * quad - norm
    }

    private fun ln(x: Float): Float = kotlin.math.ln(x.toDouble()).toFloat()

    companion object {
        private val ln2Pi = kotlin.math.ln(2.0 * PI).toFloat()
    }
}

data class ColorGmm(
    val components: List<GaussianHsv>
) {
    fun posterior(h: Float, s: Float, v: Float): Float {
        if (components.isEmpty()) return 0f
        var maxLog = Float.NEGATIVE_INFINITY
        val logs = FloatArray(components.size)
        for (i in components.indices) {
            val c = components[i]
            val log = kotlin.math.ln(c.weight.coerceAtLeast(1e-6f).toDouble()).toFloat() +
                c.logPdf(h, s, v)
            logs[i] = log
            if (log > maxLog) maxLog = log
        }
        var sum = 0f
        for (log in logs) {
            sum += exp((log - maxLog).toDouble()).toFloat()
        }
        // Softmax mass of all components relative to a uniform fallback.
        val evidence = exp(maxLog.toDouble()).toFloat() * sum
        return (evidence / (evidence + 1e-3f)).coerceIn(0f, 1f)
    }
}

class GmmColorModel {
    var red: ColorGmm = defaultGmm(0f)
    var green: ColorGmm = defaultGmm(120f)
    var blue: ColorGmm = defaultGmm(240f)
    var enabled: Boolean = true
    var blend: Float = 0.65f

    fun fitFromCalibration(calibration: ColorCalibration?) {
        if (calibration == null) {
            red = defaultGmm(0f)
            green = defaultGmm(120f)
            blue = defaultGmm(240f)
            return
        }
        red = fitSlot(calibration.redColor, fallbackHue = 0f)
        green = fitSlot(calibration.greenColor, fallbackHue = 120f)
        blue = fitSlot(calibration.blueColor, fallbackHue = 240f)
    }

    fun posterior(color: LedColor, h: Float, s: Float, v: Float): Float {
        val gmm = when (color) {
            LedColor.RED -> red
            LedColor.GREEN -> green
            LedColor.BLUE -> blue
            LedColor.UNKNOWN -> return 0f
        }
        return gmm.posterior(h, s, v)
    }

    /**
     * Blend classical HSV similarity with GMM posterior.
     */
    fun blendSimilarity(baseSim: Float, color: LedColor, h: Float, s: Float, v: Float): Float {
        if (!enabled) return baseSim
        val p = posterior(color, h, s, v)
        val a = blend.coerceIn(0f, 1f)
        return ((1f - a) * baseSim + a * baseSim * (0.35f + 0.65f * p)).coerceIn(0f, 1f)
    }

    private fun fitSlot(rgb: Triple<Float, Float, Float>, fallbackHue: Float): ColorGmm {
        val hsv = FloatArray(3)
        android.graphics.Color.RGBToHSV(
            rgb.first.toInt().coerceIn(0, 255),
            rgb.second.toInt().coerceIn(0, 255),
            rgb.third.toInt().coerceIn(0, 255),
            hsv
        )
        val primary = GaussianHsv(
            meanH = hsv[0],
            meanS = hsv[1].coerceAtLeast(0.2f),
            meanV = hsv[2].coerceAtLeast(0.2f),
            sigmaH = 18f,
            sigmaS = 0.18f,
            sigmaV = 0.22f,
            weight = 0.75f
        )
        val secondary = GaussianHsv(
            meanH = hsv[0],
            meanS = (hsv[1] * 0.85f).coerceIn(0.15f, 1f),
            meanV = (hsv[2] * 0.9f).coerceIn(0.15f, 1f),
            sigmaH = 35f,
            sigmaS = 0.28f,
            sigmaV = 0.35f,
            weight = 0.25f
        )
        // Keep a weak fallback around nominal hue for lighting drift.
        val tertiary = GaussianHsv(
            meanH = fallbackHue,
            meanS = 0.7f,
            meanV = 0.7f,
            sigmaH = 40f,
            sigmaS = 0.35f,
            sigmaV = 0.4f,
            weight = 0.08f
        )
        val comps = listOf(primary, secondary, tertiary)
        val wSum = comps.sumOf { it.weight.toDouble() }.toFloat().coerceAtLeast(1e-6f)
        return ColorGmm(comps.map { it.copy(weight = it.weight / wSum) })
    }

    private fun defaultGmm(hue: Float): ColorGmm {
        val primary = GaussianHsv(hue, 0.85f, 0.85f, 22f, 0.2f, 0.25f, 0.7f)
        val secondary = GaussianHsv(hue, 0.55f, 0.65f, 40f, 0.3f, 0.35f, 0.3f)
        return ColorGmm(listOf(primary, secondary))
    }
}
