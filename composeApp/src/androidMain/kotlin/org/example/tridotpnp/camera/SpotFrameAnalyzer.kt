package org.example.tridotpnp.camera

import android.graphics.Bitmap
import android.os.Handler
import androidx.camera.core.ImageProxy
import org.example.tridotpnp.config.AppTuningSettings
import org.example.tridotpnp.config.ProbabilityMatrix32x24
import org.example.tridotpnp.tracking.FrameTrackingSnapshot
import org.example.tridotpnp.tracking.TrackLockState
import org.example.tridotpnp.tracking.advanceTrackLockState
import org.example.tridotpnp.tracking.rgbTripleFromSpots
import org.example.tridotpnp.vision.BrightSpot
import org.example.tridotpnp.vision.BrightSpotDetector
import org.example.tridotpnp.vision.ColorCandidatePools
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max
import kotlin.math.min

/**
 * Analyzer-thread frame pipeline: detect → track-lock → optional tracking callback.
 * Keeps CameraPreview focused on CameraX binding and overlays.
 */
internal class SpotFrameAnalyzer(
    private val detector: BrightSpotDetector,
    private val mainHandler: Handler,
    private val onFpsUpdate: () -> ((Float) -> Unit),
    private val onTuning: () -> AppTuningSettings,
    private val onProbabilityMatrix: () -> ProbabilityMatrix32x24?,
    private val onEnableRoiOptimization: () -> Boolean,
    private val onCaptureBitmapForCalibration: () -> Boolean,
    private val onSpotsAnalyzed: () -> ((List<BrightSpot>, Pair<Int, Int>, ColorCandidatePools) -> FrameTrackingSnapshot?)?,
    private val onBrightSpotsDetected: () -> (List<BrightSpot>, Pair<Int, Int>?, FrameTrackingSnapshot?) -> Unit,
    private val onBitmapCaptured: (Bitmap) -> Unit,
    private val onRoiInfo: (RoiVisualizationInfo?) -> Unit,
    private val onManualRoiCleared: () -> Unit,
    private val onPreviewSpots: (List<BrightSpot>, Pair<Int, Int>?) -> Unit
) {
    private val frameCount = AtomicInteger(0)
    private val lastFrameTime = AtomicLong(System.nanoTime())
    private val trackLockRef = AtomicReference<TrackLockState?>(null)
    private val lockMissFrames = AtomicInteger(0)
    val forceGlobalSearchRequested = AtomicBoolean(false)

    /** Thread-safe ROI seeds written from UI, read on analyzer thread. */
    val manualRoiRef = AtomicReference<ManualRoiCoords?>(null)
    val limitRegionRef = AtomicReference<ManualRoiCoords?>(null)

    fun clearTrackLock() {
        trackLockRef.set(null)
        lockMissFrames.set(0)
    }

    fun analyze(imageProxy: ImageProxy) {
        val currentTime = System.nanoTime()
        val count = frameCount.incrementAndGet()
        val elapsedNs = currentTime - lastFrameTime.get()
        if (elapsedNs >= 1_000_000_000L) {
            val fps = count * 1_000_000_000f / elapsedNs
            frameCount.set(0)
            lastFrameTime.set(currentTime)
            mainHandler.post {
                onFpsUpdate()(fps)
            }
        }

        val lockState = trackLockRef.get()
        val tuningNow = onTuning()
        processImage(
            imageProxy = imageProxy,
            detector = detector,
            tuning = tuningNow,
            probabilityMatrix = onProbabilityMatrix(),
            enableRoiOptimization = onEnableRoiOptimization(),
            lastTriSpots = lockState?.rgb,
            lastTriSpotsImageSize = lockState?.imageSize,
            lockConfidence = lockState?.confidence ?: 0f,
            manualRoi = manualRoiRef.get(),
            limitRegion = limitRegionRef.get(),
            forceGlobalSearch = forceGlobalSearchRequested.getAndSet(false),
            onRoiInfo = { info ->
                mainHandler.post { onRoiInfo(info) }
            },
            onSpotsDetected = { spots, size, foundInManualRoi, candidates, preferredTrackLock ->
                if (foundInManualRoi && spots.size == 3) {
                    mainHandler.post {
                        manualRoiRef.set(null)
                        onManualRoiCleared()
                    }
                }

                val rgb = rgbTripleFromSpots(spots)
                val miss = if (rgb == null) {
                    lockMissFrames.incrementAndGet()
                } else {
                    lockMissFrames.set(0)
                    0
                }
                val imageLockDist = if (size != null) {
                    val scale = min(
                        1f,
                        tuningNow.workingMaxDimension.toFloat() /
                            max(size.first, size.second).toFloat()
                    )
                    tuningNow.trackLockMaxDistPx / scale.coerceAtLeast(0.05f)
                } else {
                    tuningNow.trackLockMaxDistPx
                }
                trackLockRef.set(
                    advanceTrackLockState(
                        previous = trackLockRef.get(),
                        newRgb = rgb,
                        imageSize = size,
                        preferredExistingLock = preferredTrackLock,
                        imageSpaceLockDistPx = imageLockDist,
                        confidenceRise = tuningNow.trackLockConfidenceRise,
                        confidenceDecay = tuningNow.trackLockConfidenceDecay,
                        missStreak = miss,
                        clearAfterMisses = 12
                    )
                )

                val snap = onSpotsAnalyzed()?.invoke(spots, size, candidates)

                mainHandler.post {
                    onPreviewSpots(spots, size)
                    onBrightSpotsDetected()(spots, size, snap)
                }
            },
            captureBitmapForCalibration = onCaptureBitmapForCalibration(),
            onBitmapCaptured = onBitmapCaptured
        )
    }
}
