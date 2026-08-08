package org.example.tridotpnp.vision

/**
 * Explicit detect output. Callers must use returned candidates / lock flag
 * instead of reading mutable fields on [BrightSpotDetector].
 */
data class BrightSpotDetectionResult(
    val spots: List<BrightSpot>,
    val redCandidates: List<BrightSpot> = emptyList(),
    val greenCandidates: List<BrightSpot> = emptyList(),
    val blueCandidates: List<BrightSpot> = emptyList(),
    /** True if detector hysteresis kept the prior RGB lock this frame. */
    val preferredTrackLock: Boolean = false
) {
    companion object {
        val Empty = BrightSpotDetectionResult(spots = emptyList())
    }
}

data class ColorCandidatePools(
    val reds: List<BrightSpot> = emptyList(),
    val greens: List<BrightSpot> = emptyList(),
    val blues: List<BrightSpot> = emptyList()
) {
    fun isEmpty(): Boolean = reds.isEmpty() || greens.isEmpty() || blues.isEmpty()
}
