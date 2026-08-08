package org.example.tridotpnp.tracking


import org.example.tridotpnp.config.AppTuningSettings
import org.example.tridotpnp.vision.BrightSpot
import org.example.tridotpnp.vision.BrightSpotDetector
import org.example.tridotpnp.vision.ColorCalibration
import org.example.tridotpnp.vision.LedColor
import org.example.tridotpnp.vision.GmmColorModel
import kotlin.math.sqrt

enum class BodyNodeRole {
    Torso,
    Waist,
    LeftWrist,
    RightWrist,
    LeftAnkle,
    RightAnkle,
    Unknown
}

data class TopologyNodeHint(
    val nodeId: Int,
    val role: BodyNodeRole,
    val positionMm: PnPDistanceCalculator.Point3D
)

/**
 * Lightweight human topology prior to suppress identity flicker.
 * When decoded ID conflicts with spatial role expectations, defer switching.
 */
class BodyTopologyPrior(
    private val minConsistentFrames: Int = 2
) {
    private val pendingIdByTrack = mutableMapOf<Int, Pair<Int, Int>>() // trackKey -> (id, count)
    private val acceptedIdByTrack = mutableMapOf<Int, Int>()

    fun roleForId(nodeId: Int): BodyNodeRole = when (nodeId) {
        1 -> BodyNodeRole.Torso
        2 -> BodyNodeRole.Waist
        3 -> BodyNodeRole.LeftWrist
        4 -> BodyNodeRole.RightWrist
        5 -> BodyNodeRole.LeftAnkle
        6 -> BodyNodeRole.RightAnkle
        else -> BodyNodeRole.Unknown
    }

    /**
     * @param trackKey stable tracker instance id (not decoded node id)
     * @return accepted node id after topology gating, or null if still deferred
     */
    fun gateIdentity(
        trackKey: Int,
        decodedId: Int?,
        selfPos: PnPDistanceCalculator.Point3D?,
        peers: List<TopologyNodeHint>
    ): Int? {
        if (decodedId == null) return acceptedIdByTrack[trackKey]

        val role = roleForId(decodedId)
        val conflict = selfPos != null && peers.isNotEmpty() && hasTopologyConflict(role, selfPos, peers)

        if (conflict) {
            // Keep previous accepted id; require more evidence later.
            pendingIdByTrack.remove(trackKey)
            return acceptedIdByTrack[trackKey]
        }

        val pending = pendingIdByTrack[trackKey]
        if (pending == null || pending.first != decodedId) {
            pendingIdByTrack[trackKey] = decodedId to 1
            return acceptedIdByTrack[trackKey]
        }

        val nextCount = pending.second + 1
        pendingIdByTrack[trackKey] = decodedId to nextCount
        if (nextCount >= minConsistentFrames) {
            acceptedIdByTrack[trackKey] = decodedId
            return decodedId
        }
        return acceptedIdByTrack[trackKey]
    }

    private fun hasTopologyConflict(
        role: BodyNodeRole,
        self: PnPDistanceCalculator.Point3D,
        peers: List<TopologyNodeHint>
    ): Boolean {
        val torso = peers.firstOrNull { it.role == BodyNodeRole.Torso }?.positionMm
        val waist = peers.firstOrNull { it.role == BodyNodeRole.Waist }?.positionMm
        when (role) {
            BodyNodeRole.Torso -> {
                if (waist != null && self.y - waist.y < -80f) return true // torso should be above waist
            }
            BodyNodeRole.Waist -> {
                if (torso != null && self.y - torso.y > 80f) return true
            }
            BodyNodeRole.LeftWrist -> {
                val right = peers.firstOrNull { it.role == BodyNodeRole.RightWrist }?.positionMm
                if (right != null && self.x - right.x > 40f) return true // left should be more +X in camera? soft
                if (torso != null && dist(self, torso) > 900f) return true
            }
            BodyNodeRole.RightWrist -> {
                val left = peers.firstOrNull { it.role == BodyNodeRole.LeftWrist }?.positionMm
                if (left != null && self.x - left.x < -40f) return true
                if (torso != null && dist(self, torso) > 900f) return true
            }
            BodyNodeRole.LeftAnkle, BodyNodeRole.RightAnkle -> {
                if (torso != null && self.y < torso.y - 50f) return true // ankles below torso
            }
            BodyNodeRole.Unknown -> Unit
        }
        return false
    }

    private fun dist(a: PnPDistanceCalculator.Point3D, b: PnPDistanceCalculator.Point3D): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        val dz = a.z - b.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
}
