package org.example.tridotpnp.vision

/** Hysteresis margin grows with lock confidence so rival triples must beat by more. */
internal fun effectiveTrackSwitchMargin(
    baseMargin: Float,
    confidence: Float,
    confidenceBoost: Float
): Float {
    val base = baseMargin.coerceAtLeast(0f)
    val conf = confidence.coerceIn(0f, 1f)
    val boost = confidenceBoost.coerceAtLeast(0f)
    return base * (1f + boost * conf)
}
