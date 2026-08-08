package org.example.tridotpnp.tracking


import org.example.tridotpnp.config.AppTuningSettings
import org.example.tridotpnp.vision.BrightSpot
import org.example.tridotpnp.vision.BrightSpotDetector
import org.example.tridotpnp.vision.ColorCalibration
import org.example.tridotpnp.vision.LedColor
import org.example.tridotpnp.vision.GmmColorModel
import kotlin.math.sqrt

/**
 * Soft inverse-kinematics style constraint via Levenberg-Marquardt on pairwise bone lengths.
 */
object IkSoftConstraint {

    data class Bone(val a: Int, val b: Int, val lengthMm: Float, val weight: Float = 1f)

    private val defaultBones = listOf(
        Bone(1, 2, 280f, 1.2f),   // torso-waist
        Bone(1, 3, 420f, 0.8f),   // torso-left wrist
        Bone(1, 4, 420f, 0.8f),   // torso-right wrist
        Bone(2, 5, 520f, 0.7f),   // waist-left ankle
        Bone(2, 6, 520f, 0.7f)    // waist-right ankle
    )

    fun optimize(
        positions: Map<Int, PnPDistanceCalculator.Point3D>,
        bones: List<Bone> = defaultBones,
        iterations: Int = 8,
        lambda0: Float = 1e-2f
    ): Map<Int, PnPDistanceCalculator.Point3D> {
        if (positions.size < 2) return positions
        val ids = positions.keys.toList()
        val xyz = ids.associateWith { id ->
            val p = positions.getValue(id)
            floatArrayOf(p.x, p.y, p.z)
        }.toMutableMap()

        var lambda = lambda0
        repeat(iterations.coerceAtLeast(1)) {
            var cost = 0f
            val grads = ids.associateWith { FloatArray(3) }.toMutableMap()

            for (bone in bones) {
                val pa = xyz[bone.a] ?: continue
                val pb = xyz[bone.b] ?: continue
                val dx = pa[0] - pb[0]
                val dy = pa[1] - pb[1]
                val dz = pa[2] - pb[2]
                val dist = sqrt(dx * dx + dy * dy + dz * dz).coerceAtLeast(1e-3f)
                val err = dist - bone.lengthMm
                cost += bone.weight * err * err
                val scale = 2f * bone.weight * err / dist
                val ga = grads.getValue(bone.a)
                val gb = grads.getValue(bone.b)
                ga[0] += scale * dx; ga[1] += scale * dy; ga[2] += scale * dz
                gb[0] -= scale * dx; gb[1] -= scale * dy; gb[2] -= scale * dz
            }

            // Soft pull toward original observations (anchor).
            for (id in ids) {
                val cur = xyz.getValue(id)
                val orig = positions.getValue(id)
                val w = 0.15f
                val ga = grads.getValue(id)
                ga[0] += 2f * w * (cur[0] - orig.x)
                ga[1] += 2f * w * (cur[1] - orig.y)
                ga[2] += 2f * w * (cur[2] - orig.z)
                cost += w * (
                    (cur[0] - orig.x).let { it * it } +
                        (cur[1] - orig.y).let { it * it } +
                        (cur[2] - orig.z).let { it * it }
                    )
            }

            // LM step with diagonal damping.
            var maxStep = 0f
            for (id in ids) {
                val g = grads.getValue(id)
                val cur = xyz.getValue(id)
                for (k in 0..2) {
                    val step = -g[k] / (1f + lambda)
                    cur[k] += step
                    maxStep = maxOf(maxStep, kotlin.math.abs(step))
                }
            }
            lambda = if (maxStep < 1f) lambda * 0.7f else lambda * 1.5f
            if (cost < 1f || maxStep < 0.05f) return@repeat
        }

        return ids.associateWith { id ->
            val v = xyz.getValue(id)
            PnPDistanceCalculator.Point3D(v[0], v[1], v[2])
        }
    }
}
