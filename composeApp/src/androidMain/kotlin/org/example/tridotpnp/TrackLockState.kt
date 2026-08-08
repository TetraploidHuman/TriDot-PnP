package org.example.tridotpnp

import androidx.compose.ui.geometry.Offset
import kotlin.math.sqrt

/**
 * Multi-frame RGB track lock. Confidence rises while the same triangle is held,
 * making hysteresis harder for stray bright spots to steal.
 */
data class TrackLockState(
    val rgb: Triple<Offset, Offset, Offset>,
    val imageSize: Pair<Int, Int>,
    val stableFrames: Int = 1,
    /** 0..1 accumulated lock confidence from consecutive stable frames. */
    val confidence: Float = 0f
)

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

internal fun rgbTripleFromSpots(spots: List<BrightSpot>): Triple<Offset, Offset, Offset>? {
    var red: Offset? = null
    var green: Offset? = null
    var blue: Offset? = null
    for (spot in spots) {
        when (spot.color) {
            LedColor.RED -> red = spot.position
            LedColor.GREEN -> green = spot.position
            LedColor.BLUE -> blue = spot.position
            else -> Unit
        }
    }
    val r = red ?: return null
    val g = green ?: return null
    val b = blue ?: return null
    return Triple(r, g, b)
}

internal fun sameTrackLock(
    previous: Triple<Offset, Offset, Offset>,
    current: Triple<Offset, Offset, Offset>,
    maxDistPx: Float
): Boolean {
    val maxD2 = maxDistPx.coerceAtLeast(1f) * maxDistPx.coerceAtLeast(1f)
    fun near(a: Offset, b: Offset): Boolean {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return dx * dx + dy * dy <= maxD2
    }
    return near(previous.first, current.first) &&
        near(previous.second, current.second) &&
        near(previous.third, current.third)
}

/**
 * Update lock after a frame. [imageSpaceLockDistPx] should be in the same coordinate
 * space as [newRgb] (typically original/rotated image pixels).
 */
internal fun advanceTrackLockState(
    previous: TrackLockState?,
    newRgb: Triple<Offset, Offset, Offset>?,
    imageSize: Pair<Int, Int>?,
    preferredExistingLock: Boolean,
    imageSpaceLockDistPx: Float,
    confidenceRise: Float,
    confidenceDecay: Float,
    missStreak: Int,
    clearAfterMisses: Int = 12
): TrackLockState? {
    if (newRgb != null && imageSize != null) {
        val continued = previous != null && (
            preferredExistingLock ||
                sameTrackLock(previous.rgb, newRgb, imageSpaceLockDistPx)
            )
        return if (continued && previous != null) {
            previous.copy(
                rgb = newRgb,
                imageSize = imageSize,
                stableFrames = previous.stableFrames + 1,
                confidence = (previous.confidence + confidenceRise.coerceAtLeast(0f)).coerceAtMost(1f)
            )
        } else {
            TrackLockState(
                rgb = newRgb,
                imageSize = imageSize,
                stableFrames = 1,
                confidence = confidenceRise.coerceIn(0f, 1f)
            )
        }
    }

    if (previous == null) return null
    if (missStreak >= clearAfterMisses) return null
    return previous.copy(
        confidence = (previous.confidence - confidenceDecay.coerceAtLeast(0f)).coerceAtLeast(0f)
    )
}

internal fun trackLockCenterDistance(
    a: Triple<Offset, Offset, Offset>,
    b: Triple<Offset, Offset, Offset>
): Float {
    val acx = (a.first.x + a.second.x + a.third.x) / 3f
    val acy = (a.first.y + a.second.y + a.third.y) / 3f
    val bcx = (b.first.x + b.second.x + b.third.x) / 3f
    val bcy = (b.first.y + b.second.y + b.third.y) / 3f
    val dx = acx - bcx
    val dy = acy - bcy
    return sqrt(dx * dx + dy * dy)
}
