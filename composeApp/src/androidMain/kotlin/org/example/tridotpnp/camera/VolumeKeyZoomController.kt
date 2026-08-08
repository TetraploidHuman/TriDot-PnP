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

object VolumeKeyZoomController {
    private val _zoomSteps = MutableSharedFlow<Int>(extraBufferCapacity = 8)
    val zoomSteps: SharedFlow<Int> = _zoomSteps

    fun zoomIn() {
        _zoomSteps.tryEmit(1)
    }

    fun zoomOut() {
        _zoomSteps.tryEmit(-1)
    }
}
