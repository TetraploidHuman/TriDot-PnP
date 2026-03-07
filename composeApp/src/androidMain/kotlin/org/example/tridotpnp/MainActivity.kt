package org.example.tridotpnp

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.compose.ui.tooling.preview.Preview

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)

        setContent {
            App()
        }
    }

    @SuppressLint("RestrictedApi")
    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        when (event.keyCode) {
            KeyEvent.KEYCODE_VOLUME_UP -> {
                if (event.action == KeyEvent.ACTION_DOWN) {
                    VolumeKeyZoomController.zoomIn()
                }
                return true
            }

            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                if (event.action == KeyEvent.ACTION_DOWN) {
                    VolumeKeyZoomController.zoomOut()
                }
                return true
            }

            // Xperia 二段快门键：半按（FOCUS）/全按（CAMERA）
            KeyEvent.KEYCODE_FOCUS -> {
                if (event.action == KeyEvent.ACTION_DOWN) {
                    if (event.repeatCount > 0) return true
                    ShutterKeyController.emit(ShutterKeyEvent.HALF_PRESS_DOWN)
                } else if (event.action == KeyEvent.ACTION_UP) {
                    ShutterKeyController.emit(ShutterKeyEvent.HALF_PRESS_UP)
                }
                return true
            }

            KeyEvent.KEYCODE_CAMERA -> {
                if (event.action == KeyEvent.ACTION_DOWN) {
                    if (event.repeatCount > 0) return true
                    ShutterKeyController.emit(ShutterKeyEvent.FULL_PRESS_DOWN)
                } else if (event.action == KeyEvent.ACTION_UP) {
                    ShutterKeyController.emit(ShutterKeyEvent.FULL_PRESS_UP)
                }
                return true
            }
        }
        return super.dispatchKeyEvent(event)
    }
}

@Preview
@Composable
fun AppAndroidPreview() {
    App()
}
