package org.example.tridotpnp

import android.graphics.Bitmap
import android.util.Log
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

internal sealed interface CalibrationRunResult {
    data class Success(
        val calibration: ColorCalibration,
        val averageBrightness: Int
    ) : CalibrationRunResult

    data object NoValidColorsFound : CalibrationRunResult
}

internal fun runAutomaticCalibration(
    detector: BrightSpotDetector,
    originalBitmap: Bitmap
): CalibrationRunResult {
    val bitmap = originalBitmap.copy(
        originalBitmap.config ?: Bitmap.Config.ARGB_8888,
        false
    )

    try {
        val width = bitmap.width
        val height = bitmap.height
        val centerX = width / 2
        val centerY = height / 2
        val calibrationRadius = (width * 0.25f).toInt()

        Log.d("Calibration", "=== 开始自动校准 ===")
        Log.d("Calibration", "图像尺寸: ${width}x${height}")
        Log.d("Calibration", "搜索区域: 中心($centerX,$centerY), 半径=$calibrationRadius")

        val result = detector.findRGBColorsInCircle(
            bitmap,
            centerX,
            centerY,
            calibrationRadius
        ) ?: return CalibrationRunResult.NoValidColorsFound

        val (redColor, greenColor, blueColor) = result
        Log.d("Calibration", "=== 校准结果 ===")
        Log.d("Calibration", analyzeCalibrationColor("红色LED", redColor))
        Log.d("Calibration", analyzeCalibrationColor("绿色LED", greenColor))
        Log.d("Calibration", analyzeCalibrationColor("蓝色LED", blueColor))

        val calibration = ColorCalibration(
            redColor = redColor,
            greenColor = greenColor,
            blueColor = blueColor
        )
        val averageBrightness = (
            redColor.first + redColor.second + redColor.third +
                greenColor.first + greenColor.second + greenColor.third +
                blueColor.first + blueColor.second + blueColor.third
            ) / 3f

        return CalibrationRunResult.Success(
            calibration = calibration,
            averageBrightness = averageBrightness.toInt()
        )
    } finally {
        bitmap.recycle()
    }
}

private fun analyzeCalibrationColor(
    name: String,
    color: Triple<Float, Float, Float>
): String {
    val (r, g, b) = color
    val total = (r + g + b).coerceAtLeast(1f)
    val rPct = (r / total * 100).toInt()
    val gPct = (g / total * 100).toInt()
    val bPct = (b / total * 100).toInt()
    return "$name: RGB(${r.toInt()},${g.toInt()},${b.toInt()}) 占比:R${rPct}% G${gPct}% B${bPct}% 亮度:${total.toInt()}"
}

@Composable
internal fun CalibrationActionButton(
    isCalibrating: Boolean,
    hasCalibrationData: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier,
        shape = MaterialTheme.shapes.small,
        color = if (isCalibrating) Color(0xFFF0A229) else Color(0xFF0078D4),
        shadowElevation = 0.dp
    ) {
        TextButton(
            onClick = onClick,
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(0.dp)
        ) {
            Text(
                text = if (isCalibrating) "确认" else if (hasCalibrationData) "已校准" else "校准",
                color = Color.White,
                style = MaterialTheme.typography.labelMedium
            )
        }
    }
}
