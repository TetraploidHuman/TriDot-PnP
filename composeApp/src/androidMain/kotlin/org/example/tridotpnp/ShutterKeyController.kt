package org.example.tridotpnp

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
