package org.example.tridotpnp


import org.example.tridotpnp.ble.BleImuClient
import org.example.tridotpnp.config.AppTuningSettings
import org.example.tridotpnp.tracking.FrameTrackingSnapshot
import org.example.tridotpnp.tracking.NodeTrackerManager
import org.example.tridotpnp.tracking.PnPDistanceCalculator
import org.example.tridotpnp.tracking.applyTuning
import org.example.tridotpnp.tracking.computeFrameTracking
import org.example.tridotpnp.vision.BrightSpot
import org.example.tridotpnp.vision.BrightSpotDetector
import org.example.tridotpnp.vision.CalibrationRunResult
import org.example.tridotpnp.vision.ColorCalibration
import org.example.tridotpnp.vision.ColorCandidatePools
import org.example.tridotpnp.vision.runAutomaticCalibration
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.widget.Toast

/**
 * Composition-root owned domain handles: vision detect, PnP/tracking, BLE.
 * UI should depend on this façade instead of constructing algorithm objects ad hoc.
 */
class AppSession(appContext: Context) {
    val detector: BrightSpotDetector = BrightSpotDetector()
    val pnpCalculator: PnPDistanceCalculator = PnPDistanceCalculator()
    val nodeTracker: NodeTrackerManager = NodeTrackerManager(pnpCalculator)
    val bleClient: BleImuClient = BleImuClient(appContext.applicationContext)

    fun applyTuning(tuning: AppTuningSettings) {
        detector.applyTuning(tuning)
        pnpCalculator.applyTuning(tuning)
        nodeTracker.configure(tuning)
    }

    fun computeTracking(
        spots: List<BrightSpot>,
        imageSize: Pair<Int, Int>,
        tuning: AppTuningSettings,
        candidates: ColorCandidatePools = ColorCandidatePools()
    ): FrameTrackingSnapshot {
        return computeFrameTracking(
            nodeTracker = nodeTracker,
            pnpCalculator = pnpCalculator,
            spots = spots,
            imageSize = imageSize,
            tuning = tuning,
            candidates = candidates
        )
    }

    fun clearCalibration() {
        detector.setCalibration(null)
    }

    fun executeCalibration(
        context: Context,
        calibrationBitmap: Bitmap?
    ): ColorCalibration? {
        if (calibrationBitmap == null || calibrationBitmap.isRecycled) {
            Toast.makeText(context, "校准失败：未获取到有效画面", Toast.LENGTH_SHORT).show()
            detector.setCalibration(null)
            return null
        }

        return try {
            when (val result = runAutomaticCalibration(detector, calibrationBitmap)) {
                is CalibrationRunResult.Success -> {
                    detector.setCalibration(result.calibration)
                    Toast.makeText(
                        context,
                        "校准成功！平均亮度: ${result.averageBrightness}",
                        Toast.LENGTH_SHORT
                    ).show()
                    result.calibration
                }

                CalibrationRunResult.NoValidColorsFound -> {
                    Toast.makeText(
                        context,
                        "校准失败！\n请确保RGB三色LED都在圆框内",
                        Toast.LENGTH_LONG
                    ).show()
                    detector.setCalibration(null)
                    null
                }
            }
        } catch (e: Exception) {
            Log.e("Calibration", "校准出错", e)
            Toast.makeText(
                context,
                "校准出错：${e.message}",
                Toast.LENGTH_LONG
            ).show()
            detector.setCalibration(null)
            null
        }
    }

    fun dispose() {
        bleClient.disconnectAll()
    }
}
