package org.example.tridotpnp.tracking


import org.example.tridotpnp.config.AppTuningSettings
import org.example.tridotpnp.vision.BrightSpot
import org.example.tridotpnp.vision.BrightSpotDetector
import org.example.tridotpnp.vision.ColorCalibration
import org.example.tridotpnp.vision.LedColor
import org.example.tridotpnp.vision.GmmColorModel
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

data class ImuSample(
    val nodeId: Int,
    val ax: Float,
    val ay: Float,
    val az: Float,
    val gx: Float,
    val gy: Float,
    val gz: Float,
    val timestampMs: Long
)

data class EskfState(
    val position: PnPDistanceCalculator.Point3D,
    val velocity: PnPDistanceCalculator.Point3D,
    val rollDeg: Float,
    val pitchDeg: Float,
    val yawDeg: Float,
    val accelBias: PnPDistanceCalculator.Point3D = PnPDistanceCalculator.Point3D(0f, 0f, 0f),
    val gyroBias: PnPDistanceCalculator.Point3D = PnPDistanceCalculator.Point3D(0f, 0f, 0f)
)

/**
 * Lightweight error-state Kalman filter for sparse visual-inertial fusion.
 * State (nominal): p, v, rpy; error: δp, δv, δθ, δba, δbg (15)
 */
class ErrorStateKalmanFilter(
    private val angleThresholdDeg: Float = 5f,
    private val positionThresholdMm: Float = 50f
) {
    var nominal: EskfState = EskfState(
        position = PnPDistanceCalculator.Point3D(0f, 0f, 1000f),
        velocity = PnPDistanceCalculator.Point3D(0f, 0f, 0f),
        rollDeg = 0f,
        pitchDeg = 0f,
        yawDeg = 0f
    )
        private set

    // Diagonal covariance proxy for p(3), v(3), θ(3), ba(3), bg(3)
    private val P = FloatArray(15) { if (it < 9) 25f else 1f }
    private var lastImuMs: Long? = null
    private var initialized = false

    fun predict(sample: ImuSample) {
        val last = lastImuMs
        lastImuMs = sample.timestampMs
        if (last == null) return
        val dt = ((sample.timestampMs - last).coerceAtLeast(1)).toFloat() / 1000f
        if (dt <= 0f || dt > 0.5f) return

        // Gyro inputs are degrees/second (firmware convention). Keep work in degrees
        // to avoid repeated rad/deg conversions (same numeric result as before).
        val gxDeg = sample.gx - nominal.gyroBias.x
        val gyDeg = sample.gy - nominal.gyroBias.y
        val gzDeg = sample.gz - nominal.gyroBias.z

        val roll = nominal.rollDeg + gxDeg * dt
        val pitch = nominal.pitchDeg + gyDeg * dt
        val yaw = nominal.yawDeg + gzDeg * dt

        // Gravity-compensated accel in a simplified navigation frame.
        val ax = sample.ax - nominal.accelBias.x
        val ay = sample.ay - nominal.accelBias.y
        val az = sample.az - nominal.accelBias.z - 9.81f

        val rr = roll * DEG_TO_RAD
        val pp = pitch * DEG_TO_RAD
        val yy = yaw * DEG_TO_RAD
        val cr = cos(rr); val sr = sin(rr)
        val cp = cos(pp); val sp = sin(pp)
        val cy = cos(yy); val sy = sin(yy)

        // Body -> world (ZYX)
        val fx = cy * cp * ax + (cy * sp * sr - sy * cr) * ay + (cy * sp * cr + sy * sr) * az
        val fy = sy * cp * ax + (sy * sp * sr + cy * cr) * ay + (sy * sp * cr - cy * sr) * az
        val fz = -sp * ax + cp * sr * ay + cp * cr * az

        // Convert m/s^2 -> mm/s^2
        val vx = nominal.velocity.x + fx * 1000f * dt
        val vy = nominal.velocity.y + fy * 1000f * dt
        val vz = nominal.velocity.z + fz * 1000f * dt
        val px = nominal.position.x + vx * dt
        val py = nominal.position.y + vy * dt
        val pz = nominal.position.z + vz * dt

        nominal = EskfState(
            position = PnPDistanceCalculator.Point3D(px, py, pz),
            velocity = PnPDistanceCalculator.Point3D(vx, vy, vz),
            rollDeg = roll,
            pitchDeg = pitch,
            yawDeg = yaw,
            accelBias = nominal.accelBias,
            gyroBias = nominal.gyroBias
        )

        // Covariance random-walk growth (sparse, diagonal).
        for (i in 0 until 3) P[i] += 4f * dt          // position
        for (i in 3 until 6) P[i] += 16f * dt         // velocity
        for (i in 6 until 9) P[i] += 0.5f * dt        // attitude (deg^2 proxy)
        for (i in 9 until 15) P[i] += 0.01f * dt      // biases
    }

    fun updateFromVision(obs: VisionObservation) {
        if (!obs.accepted) return
        val pose = obs.pnp.pose6DOF
        val measP = obs.pnp.centerCoords
        val measRoll = pose?.roll ?: nominal.rollDeg
        val measPitch = pose?.pitch ?: nominal.pitchDeg
        val measYaw = pose?.yaw ?: nominal.yawDeg

        if (!initialized) {
            nominal = EskfState(
                position = measP,
                velocity = PnPDistanceCalculator.Point3D(0f, 0f, 0f),
                rollDeg = measRoll,
                pitchDeg = measPitch,
                yawDeg = measYaw
            )
            for (i in P.indices) P[i] = if (i < 9) 4f else 1f
            initialized = true
            return
        }

        // Scalar Kalman gains from diagonal P and observation noise R.
        val rp = (8f / (obs.confidence.coerceAtLeast(0.05f))).coerceAtLeast(1f)
        val rth = (4f / (obs.confidence.coerceAtLeast(0.05f))).coerceAtLeast(0.5f)

        fun updateScalar(idx: Int, innov: Float, r: Float): Float {
            val p = P[idx]
            val k = p / (p + r)
            P[idx] = (1f - k) * p
            return k * innov
        }

        val dpx = updateScalar(0, measP.x - nominal.position.x, rp)
        val dpy = updateScalar(1, measP.y - nominal.position.y, rp)
        val dpz = updateScalar(2, measP.z - nominal.position.z, rp)
        val droll = updateScalar(6, angleDiff(measRoll, nominal.rollDeg), rth)
        val dpitch = updateScalar(7, angleDiff(measPitch, nominal.pitchDeg), rth)
        val dyaw = updateScalar(8, angleDiff(measYaw, nominal.yawDeg), rth)

        // Mild velocity damping on vision update (sparse correction philosophy).
        for (i in 3 until 6) {
            P[i] *= 0.85f
        }

        nominal = nominal.copy(
            position = PnPDistanceCalculator.Point3D(
                nominal.position.x + dpx,
                nominal.position.y + dpy,
                nominal.position.z + dpz
            ),
            velocity = PnPDistanceCalculator.Point3D(
                nominal.velocity.x * 0.7f,
                nominal.velocity.y * 0.7f,
                nominal.velocity.z * 0.7f
            ),
            rollDeg = nominal.rollDeg + droll,
            pitchDeg = nominal.pitchDeg + dpitch,
            yawDeg = nominal.yawDeg + dyaw
        )
        initialized = true
    }

    fun needsVisualCalibration(): Boolean {
        if (!initialized) return false
        val posSigma = sqrt((P[0] + P[1] + P[2]) / 3f)
        val angSigma = sqrt((P[6] + P[7] + P[8]) / 3f)
        return 3f * posSigma > positionThresholdMm || 3f * angSigma > angleThresholdDeg
    }

    fun isInitialized(): Boolean = initialized

    fun attitudeUncertaintyDeg(): Float = sqrt((P[6] + P[7] + P[8]) / 3f)

    private fun angleDiff(a: Float, b: Float): Float {
        var d = a - b
        while (d > 180f) d -= 360f
        while (d < -180f) d += 360f
        return d
    }

    companion object {
        private const val DEG_TO_RAD = (Math.PI / 180.0).toFloat()
    }
}
