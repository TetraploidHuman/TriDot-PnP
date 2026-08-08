package org.example.tridotpnp

/**
 * Inline RGB→HSV matching [android.graphics.Color.RGBToHSV] exactly,
 * without JNI per-call overhead. Outputs H in [0,360), S/V in [0,1].
 */
internal fun rgbToHsv(r: Int, g: Int, b: Int, out: FloatArray) {
    val min = minOf(r, g, b)
    val max = maxOf(r, g, b)
    val delta = (max - min).toFloat()

    out[1] = if (max == 0) 0f else delta / max
    out[2] = max / 255f

    var h: Float
    if (delta == 0f) {
        h = 0f
    } else if (r == max) {
        h = (g - b) / delta
    } else if (g == max) {
        h = 2f + (b - r) / delta
    } else {
        h = 4f + (r - g) / delta
    }
    h *= 60f
    if (h < 0f) h += 360f
    out[0] = h
}
