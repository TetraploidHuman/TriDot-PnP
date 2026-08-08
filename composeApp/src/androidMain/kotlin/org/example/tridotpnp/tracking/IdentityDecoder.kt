package org.example.tridotpnp.tracking


import org.example.tridotpnp.config.AppTuningSettings
import org.example.tridotpnp.vision.BrightSpot
import org.example.tridotpnp.vision.BrightSpotDetector
import org.example.tridotpnp.vision.ColorCalibration
import org.example.tridotpnp.vision.LedColor
import org.example.tridotpnp.vision.GmmColorModel
import kotlin.math.sqrt

/**
 * Temporal LED identity decoder: sliding-window sync-header correlation + CRC + confirm FSM.
 *
 * Encoding frame (default, matches proposal):
 *   [sync 8b][nodeId 8b][crc4 4b]  — bits sampled from coding LED brightness over frames.
 */
enum class IdentityDecoderState {
    HuntingSync,
    ReceivingPayload,
    Confirmed
}

data class IdentityDecodeConfig(
    val syncHeader: Int = 0b10110010,
    val syncBits: Int = 8,
    val idBits: Int = 8,
    val crcBits: Int = 4,
    val syncThreshold: Float = 0.82f,
    val bitThreshold: Float = 0.5f,
    val confirmNeeded: Int = 3,
    val maxHistory: Int = 96
)

data class IdentityDecodeResult(
    val nodeId: Int?,
    val state: IdentityDecoderState,
    val confirmCount: Int,
    val lastCorrelation: Float,
    val crcOk: Boolean
)

class IdentityDecoder(
    private var config: IdentityDecodeConfig = IdentityDecodeConfig()
) {
    private val samples = ArrayDeque<Float>()
    private var state = IdentityDecoderState.HuntingSync
    private var payloadBits = mutableListOf<Int>()
    private var candidateId: Int? = null
    private var confirmCount = 0
    private var lastCorrelation = 0f
    private var lastCrcOk = false
    private var lockedId: Int? = null

    fun updateConfig(newConfig: IdentityDecodeConfig) {
        config = newConfig
    }

    fun reset() {
        samples.clear()
        state = IdentityDecoderState.HuntingSync
        payloadBits.clear()
        candidateId = null
        confirmCount = 0
        lastCorrelation = 0f
        lastCrcOk = false
        lockedId = null
    }

    /**
     * Push one frame sample for the coding LED (normalized brightness 0..1).
     */
    fun pushSample(brightness01: Float): IdentityDecodeResult {
        val bit = if (brightness01 >= config.bitThreshold) 1f else 0f
        samples.addLast(bit)
        while (samples.size > config.maxHistory) samples.removeFirst()

        when (state) {
            IdentityDecoderState.HuntingSync,
            IdentityDecoderState.Confirmed -> {
                val corr = bestSyncCorrelation()
                lastCorrelation = corr.second
                if (corr.second >= config.syncThreshold) {
                    payloadBits.clear()
                    state = IdentityDecoderState.ReceivingPayload
                    val start = corr.first + 1
                    for (i in start until samples.size) {
                        payloadBits.add(samples.elementAt(i).toInt())
                        if (payloadBits.size >= config.idBits + config.crcBits) break
                    }
                    if (payloadBits.size >= config.idBits + config.crcBits) {
                        finishPayload()
                    }
                }
            }

            IdentityDecoderState.ReceivingPayload -> {
                payloadBits.add(bit.toInt())
                if (payloadBits.size >= config.idBits + config.crcBits) {
                    finishPayload()
                }
            }
        }

        return IdentityDecodeResult(
            nodeId = lockedId,
            state = state,
            confirmCount = confirmCount,
            lastCorrelation = lastCorrelation,
            crcOk = lastCrcOk
        )
    }

    /**
     * Sample coding LEDs (green/blue). Red is treated as the constant anchor beacon.
     * Brightness fields are similarity*intensity and can exceed 255, so normalize by local max.
     */
    fun pushFromSpots(spots: List<BrightSpot>): IdentityDecodeResult {
        if (spots.size < 3) {
            return IdentityDecodeResult(lockedId, state, confirmCount, lastCorrelation, lastCrcOk)
        }
        val green = spots.firstOrNull { it.color == LedColor.GREEN }
        val blue = spots.firstOrNull { it.color == LedColor.BLUE }
        val codingValues = listOfNotNull(green?.brightness, blue?.brightness)
        if (codingValues.isEmpty()) {
            return IdentityDecodeResult(lockedId, state, confirmCount, lastCorrelation, lastCrcOk)
        }
        val codingRaw = codingValues.average().toFloat()
        val ref = spots.maxOf { it.brightness }.coerceAtLeast(1e-3f)
        val norm = (codingRaw / ref).coerceIn(0f, 1f)
        return pushSample(norm)
    }

    private fun finishPayload() {
        val id = bitsToInt(payloadBits.subList(0, config.idBits))
        val crc = bitsToInt(payloadBits.subList(config.idBits, config.idBits + config.crcBits))
        val expected = crc4(id)
        lastCrcOk = crc == expected
        payloadBits.clear()
        state = IdentityDecoderState.HuntingSync

        if (!lastCrcOk) return

        if (candidateId == id) {
            confirmCount++
        } else {
            candidateId = id
            confirmCount = 1
        }

        if (confirmCount >= config.confirmNeeded) {
            lockedId = id
            state = IdentityDecoderState.Confirmed
        }
    }

    private fun bestSyncCorrelation(): Pair<Int, Float> {
        val n = config.syncBits
        if (samples.size < n) return -1 to 0f
        val sync = IntArray(n) { i -> (config.syncHeader shr (n - 1 - i)) and 1 }
        var bestIdx = -1
        var best = 0f
        val lastStart = samples.size - n
        for (start in 0..lastStart) {
            var dot = 0f
            var o2 = 0f
            var s2 = 0f
            for (k in 0 until n) {
                val o = samples.elementAt(start + k)
                val s = sync[k].toFloat()
                dot += o * s
                o2 += o * o
                s2 += s * s
            }
            val denom = sqrt(o2 * s2)
            val r = if (denom > 1e-6f) dot / denom else 0f
            if (r > best) {
                best = r
                bestIdx = start + n - 1
            }
        }
        return bestIdx to best
    }

    private fun bitsToInt(bits: List<Int>): Int {
        var v = 0
        for (b in bits) {
            v = (v shl 1) or (b and 1)
        }
        return v
    }

    /** CRC-4-ITU (poly 0x3). */
    private fun crc4(data: Int): Int {
        var crc = 0
        for (i in config.idBits - 1 downTo 0) {
            val bit = (data shr i) and 1
            val msb = (crc shr 3) and 1
            crc = ((crc shl 1) and 0xF) xor (if ((msb xor bit) != 0) 0x3 else 0)
        }
        return crc and 0xF
    }
}
