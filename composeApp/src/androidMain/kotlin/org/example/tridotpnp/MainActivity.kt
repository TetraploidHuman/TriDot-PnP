package org.example.tridotpnp

import android.annotation.SuppressLint
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.tooling.preview.Preview
import androidx.core.view.WindowCompat
import org.example.tridotpnp.camera.ShutterKeyController
import org.example.tridotpnp.camera.ShutterKeyEvent
import org.example.tridotpnp.camera.VolumeKeyZoomController

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        // Camera preview is edge-to-edge. Android's default 3-button nav contrast scrim
        // paints a thin light/dark strip that flips with preview brightness and sits
        // right next to our zoom indicator — disable it explicitly and keep it off.
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.dark(Color.TRANSPARENT)
        )
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        disableSystemBarContrastScrim()

        setContent {
            // enableEdgeToEdge() defaults contrast enforcement back on; clear again
            // after composition attaches / recomposes.
            SideEffect { disableSystemBarContrastScrim() }
            App()
        }
    }

    override fun onResume() {
        super.onResume()
        disableSystemBarContrastScrim()
    }

    private fun disableSystemBarContrastScrim() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            window.isNavigationBarContrastEnforced = false
            window.isStatusBarContrastEnforced = false
        }
        @Suppress("DEPRECATION")
        run {
            window.statusBarColor = Color.TRANSPARENT
            window.navigationBarColor = Color.TRANSPARENT
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
