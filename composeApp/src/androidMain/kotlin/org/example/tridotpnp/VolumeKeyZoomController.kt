package org.example.tridotpnp

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
