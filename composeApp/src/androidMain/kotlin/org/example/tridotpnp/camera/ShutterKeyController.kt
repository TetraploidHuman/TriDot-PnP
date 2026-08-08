package org.example.tridotpnp.camera


import org.example.tridotpnp.config.AppTuningSettings
import org.example.tridotpnp.config.ProbabilityMatrix32x24
import org.example.tridotpnp.config.buildManualProbabilityMatrix32x24
import org.example.tridotpnp.tracking.FrameTrackingSnapshot
import org.example.tridotpnp.tracking.TrackLockState
import org.example.tridotpnp.tracking.advanceTrackLockState
import org.example.tridotpnp.tracking.rgbTripleFromSpots
import org.example.tridotpnp.vision.BrightSpot
import org.example.tridotpnp.vision.BrightSpotDetector
import org.example.tridotpnp.ui.ProbabilityEditMode
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow

enum class ShutterKeyEvent {
    HALF_PRESS_DOWN,
    HALF_PRESS_UP,
    FULL_PRESS_DOWN,
    FULL_PRESS_UP
}

object ShutterKeyController {
    private val _events = MutableSharedFlow<ShutterKeyEvent>(extraBufferCapacity = 16)
    val events: SharedFlow<ShutterKeyEvent> = _events

    fun emit(event: ShutterKeyEvent) {
        _events.tryEmit(event)
    }
}
