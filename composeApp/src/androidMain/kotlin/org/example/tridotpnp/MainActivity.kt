package org.example.tridotpnp

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

    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        if (event.action == KeyEvent.ACTION_DOWN) {
            when (event.keyCode) {
                KeyEvent.KEYCODE_VOLUME_UP -> {
                    VolumeKeyZoomController.zoomIn()
                    return true
                }

                KeyEvent.KEYCODE_VOLUME_DOWN -> {
                    VolumeKeyZoomController.zoomOut()
                    return true
                }
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
