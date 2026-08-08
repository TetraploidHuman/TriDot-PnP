package org.example.tridotpnp

import androidx.compose.ui.geometry.Offset
import kotlin.math.sqrt

data class TrackedNodeState(
    val trackKey: Int,
    val nodeId: Int?,
    val spots: List<BrightSpot>,
    val observation: VisionObservation?,
    val fusedPose: EskfState?,
    val identityConfirmCount: Int,
    val identityState: IdentityDecoderState,
    val needsVisualCalibration: Boolean
)

/**
 * Multi-node tracker: one identity decoder + ESKF per track.
 */
class NodeTrackerManager(
    private val pnpCalculator: PnPDistanceCalculator,
    private var maxNodes: Int = 6
) {
    private data class Track(
        val key: Int,
        var center: Offset,
        var spots: List<BrightSpot> = emptyList(),
        val decoder: IdentityDecoder = IdentityDecoder(),
        val eskf: ErrorStateKalmanFilter = ErrorStateKalmanFilter(),
        var lastObs: VisionObservation? = null,
        var lastIdentity: IdentityDecodeResult? = null,
        var acceptedNodeId: Int? = null,
        var missedFrames: Int = 0
    )

    private val tracks = mutableListOf<Track>()
    private val topology = BodyTopologyPrior()
    private var nextKey = 1

    var edgeLengthMm: Float = 18f
    var visionMinConfidence: Float = 0.45f
    var enableIdentityDecode: Boolean = true
    var enableTopologyPrior: Boolean = true
    var enableEskfFusion: Boolean = true
    var enableIkPostProcess: Boolean = true
    var enableVisionGate: Boolean = true
    var visionMaxReprojectionErrorMm: Float = 25f
    var identityConfirmNeeded: Int = 3

    fun configure(tuning: AppTuningSettings) {
        synchronized(this) {
            edgeLengthMm = tuning.knownTriangleEdgeLengthMm
            visionMinConfidence = tuning.visionMinConfidence
            visionMaxReprojectionErrorMm = tuning.visionMaxReprojectionErrorMm
            enableIdentityDecode = tuning.enableIdentityDecode
            enableTopologyPrior = tuning.enableTopologyPrior
            enableEskfFusion = tuning.enableEskfFusion
            enableIkPostProcess = tuning.enableIkPostProcess
            enableVisionGate = tuning.enableVisionConfidenceGate
            maxNodes = tuning.maxTrackedNodes.coerceIn(1, 17)
            identityConfirmNeeded = tuning.identityConfirmNeeded.coerceIn(1, 10)
        }
    }

    fun reset() {
        synchronized(this) {
            tracks.clear()
            nextKey = 1
        }
    }

    fun onImu(sample: ImuSample) {
        synchronized(this) {
            if (!enableEskfFusion) return
            val track = tracks.firstOrNull { it.acceptedNodeId == sample.nodeId }
                ?: tracks.firstOrNull { it.lastObs?.nodeId == sample.nodeId }
                ?: return
            track.eskf.predict(sample)
        }
    }

    fun updateFromPrimaryTriple(
        spots: List<BrightSpot>,
        imageSize: Pair<Int, Int>,
        tuning: AppTuningSettings,
        extraTriples: List<DetectedTriangle> = emptyList(),
        timestampMs: Long = System.currentTimeMillis()
    ): List<TrackedNodeState> {
        synchronized(this) {
            // configure already holds lock; call body fields directly
            edgeLengthMm = tuning.knownTriangleEdgeLengthMm
            visionMinConfidence = tuning.visionMinConfidence
            visionMaxReprojectionErrorMm = tuning.visionMaxReprojectionErrorMm
            enableIdentityDecode = tuning.enableIdentityDecode
            enableTopologyPrior = tuning.enableTopologyPrior
            enableEskfFusion = tuning.enableEskfFusion
            enableIkPostProcess = tuning.enableIkPostProcess
            enableVisionGate = tuning.enableVisionConfidenceGate
            maxNodes = tuning.maxTrackedNodes.coerceIn(1, 17)
            identityConfirmNeeded = tuning.identityConfirmNeeded.coerceIn(1, 10)

            val triples = mutableListOf<DetectedTriangle>()
            if (spots.size == 3) {
                val center = Offset(
                    (spots[0].position.x + spots[1].position.x + spots[2].position.x) / 3f,
                    (spots[0].position.y + spots[1].position.y + spots[2].position.y) / 3f
                )
                triples += DetectedTriangle(spots, score = 1f, center = center)
            }
            if (tuning.enableMultiNodeTracking) {
                triples += extraTriples
            }
            return finalize(updateTriples(triples, imageSize, tuning, timestampMs))
        }
    }

    private fun updateTriples(
        triples: List<DetectedTriangle>,
        imageSize: Pair<Int, Int>,
        tuning: AppTuningSettings,
        timestampMs: Long
    ): List<TrackedNodeState> {
        val matchedKeys = mutableSetOf<Int>()
        val unique = triples.distinctBy {
            ((it.center.x / 8f).toInt() to (it.center.y / 8f).toInt())
        }

        for (tri in unique.take(maxNodes)) {
            val track = nearestTrack(tri.center) ?: Track(nextKey++, tri.center).also { tracks += it }
            track.center = tri.center
            track.spots = tri.spots
            track.missedFrames = 0
            matchedKeys += track.key

            val idResult = if (enableIdentityDecode) {
                track.decoder.updateConfig(
                    IdentityDecodeConfig(confirmNeeded = identityConfirmNeeded)
                )
                track.decoder.pushFromSpots(tri.spots)
            } else {
                IdentityDecodeResult(null, IdentityDecoderState.HuntingSync, 0, 0f, false)
            }
            track.lastIdentity = idResult

            val pnp = calculateRgbPnpResult(
                spots = tri.spots,
                imageSize = imageSize,
                tuning = tuning,
                pnpCalculator = pnpCalculator
            ) ?: continue

            val peers = tracks.mapNotNull { t ->
                val id = t.acceptedNodeId ?: return@mapNotNull null
                val pos = t.lastObs?.pnp?.centerCoords ?: return@mapNotNull null
                TopologyNodeHint(id, topology.roleForId(id), pos)
            }

            val gatedId = if (enableTopologyPrior) {
                topology.gateIdentity(
                    trackKey = track.key,
                    decodedId = idResult.nodeId,
                    selfPos = pnp.centerCoords,
                    peers = peers.filter { it.nodeId != idResult.nodeId }
                )
            } else {
                idResult.nodeId
            }
            if (gatedId != null) track.acceptedNodeId = gatedId

            val obs = if (enableVisionGate) {
                VisionConfidenceGate.evaluate(
                    spots = tri.spots,
                    pnp = pnp,
                    edgeLengthMm = edgeLengthMm,
                    identityConfirmCount = idResult.confirmCount,
                    nodeId = track.acceptedNodeId,
                    config = VisionGateConfig(
                        minConfidence = visionMinConfidence,
                        maxReprojectionErrorMm = visionMaxReprojectionErrorMm
                    ),
                    timestampMs = timestampMs
                )
            } else {
                VisionObservation(
                    nodeId = track.acceptedNodeId,
                    spots = tri.spots,
                    pnp = pnp,
                    confidence = 1f,
                    geometryScore = 1f,
                    reprojectionErrorMm = 0f,
                    identityConfirmCount = idResult.confirmCount,
                    timestampMs = timestampMs,
                    accepted = true
                )
            }
            track.lastObs = obs
            if (enableEskfFusion && obs.accepted) {
                track.eskf.updateFromVision(obs)
            }
        }

        tracks.forEach { t ->
            if (t.key !in matchedKeys) t.missedFrames++
        }
        tracks.removeAll { it.missedFrames > 45 }

        return tracks.map { t ->
            TrackedNodeState(
                trackKey = t.key,
                nodeId = t.acceptedNodeId,
                spots = t.spots,
                observation = t.lastObs,
                fusedPose = if (enableEskfFusion && t.eskf.isInitialized()) t.eskf.nominal else null,
                identityConfirmCount = t.lastIdentity?.confirmCount ?: 0,
                identityState = t.lastIdentity?.state ?: IdentityDecoderState.HuntingSync,
                needsVisualCalibration = enableEskfFusion && t.eskf.needsVisualCalibration()
            )
        }
    }

    private fun finalize(states: List<TrackedNodeState>): List<TrackedNodeState> {
        return if (enableIkPostProcess) applyIk(states) else states
    }

    private fun nearestTrack(center: Offset, maxDist: Float = 80f): Track? {
        var best: Track? = null
        var bestD = maxDist
        for (t in tracks) {
            val d = dist(t.center, center)
            if (d < bestD) {
                bestD = d
                best = t
            }
        }
        return best
    }

    private fun dist(a: Offset, b: Offset): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun applyIk(states: List<TrackedNodeState>): List<TrackedNodeState> {
        val poses = states.mapNotNull { s ->
            val id = s.nodeId ?: return@mapNotNull null
            val p = s.fusedPose?.position ?: s.observation?.pnp?.centerCoords ?: return@mapNotNull null
            id to p
        }.toMap()
        if (poses.size < 2) return states
        val optimized = IkSoftConstraint.optimize(poses)
        return states.map { s ->
            val id = s.nodeId ?: return@map s
            val np = optimized[id] ?: return@map s
            val fused = s.fusedPose?.copy(position = np)
            val obs = s.observation?.let { old ->
                old.copy(
                    pnp = old.pnp.copy(
                        centerCoords = np,
                        pose6DOF = old.pnp.pose6DOF?.copy(position = np)
                    )
                )
            }
            s.copy(fusedPose = fused ?: s.fusedPose, observation = obs ?: s.observation)
        }
    }
}
