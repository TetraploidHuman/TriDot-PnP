package org.example.tridotpnp

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.sqrt

/**
 * 三点 PnP 的时序滤波与双解消歧。
 *
 * 参考实现：`research/tridot_temporal/temporal.py`（Python 侧已用 test/hard_test 做过 A/B）。
 *
 * 1. 双解消歧：三色点共面 + 已知等边边长，PnP 存在两个几何上都成立的解（镜像解）。
 *    两解的平移只差几毫米到几十毫米，但旋转可能差 40°~150°。这里用多初值阻尼 Newton
 *    枚举所有根，再按“与上一帧姿态连续”选根（无历史时按“最接近单位旋转”，即板子大致
 *    正对相机）。
 * 2. 时序滤波：对目标中心位置做 alpha-beta 滤波（稳态 Kalman），带物理意义的新息门限：
 *    单帧跳变超过门限视为离群（检测跳到别的光斑/镜像解），用匀速预测滑行最多
 *    maxCoastFrames 帧；连续超限才重新锁定。
 */
class PnPTemporalTracker(
    var alpha: Float = 0.85f,
    var beta: Float = 0.15f,
    var gateAbsMm: Float = 60f,
    var gateRel: Float = 0.08f,
    var maxCoastFrames: Int = 5,
    var maxDtSeconds: Double = 0.25,
    var maxSpeedMmPerSec: Float = 4000f,
    var rotGateDeg: Float = 30f
) {
    class DepthRoot internal constructor(val depthsMm: FloatArray)

    /**
     * @param translationMm 目标中心在相机系下的位置（mm）
     * @param rotation 3x3 行主序旋转矩阵（目标系 -> 相机系）
     */
    class Snapshot(
        val translationMm: FloatArray,
        val rotation: FloatArray,
        val accepted: Boolean,
        val coasting: Boolean,
        val candidates: Int
    )

    private val pos = FloatArray(3)
    private val vel = FloatArray(3)
    private var rotation: FloatArray? = null
    private var initialized = false
    private var lastTimeSec = 0.0
    private var coastCount = 0
    private var lastDepthsMm: FloatArray? = null

    /**
     * 预测某个颜色角点在给定时刻的像素位置（用于单角点被遮挡时补点）。
     *
     * 用「目标中心位置 + 速度 * dt」预测中心，用最后一帧的旋转把目标系角点投影到相机系。
     * 这是 Python 侧 `research/tridot_temporal/completion.py` 的端侧版本（只用于单点缺失）。
     *
     * @param colorIndex 0=红 1=绿 2=蓝（与 [solveRoots] 的 COLOR 顺序一致）
     * @param fx/fy/cx/cy 相机内参（像素）
     * @return [u, v] 像素坐标；跟踪器未初始化或预测点跑到相机后方时返回 null
     */
    fun projectCorner(
        colorIndex: Int,
        edgeMm: Float,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        timeSec: Double
    ): FloatArray? {
        val rot = rotation ?: return null
        if (!initialized || edgeMm <= 0f) return null
        val rel = objectCornerRelative(colorIndex, edgeMm)
        var dt = timeSec - lastTimeSec
        if (dt < 0.0) dt = 0.0
        if (dt > maxDtSeconds) dt = maxDtSeconds
        val px = pos[0] + vel[0] * dt.toFloat()
        val py = pos[1] + vel[1] * dt.toFloat()
        val pz = pos[2] + vel[2] * dt.toFloat()
        val x = rot[0] * rel[0] + rot[1] * rel[1] + px
        val y = rot[3] * rel[0] + rot[4] * rel[1] + py
        val z = rot[6] * rel[0] + rot[7] * rel[1] + pz
        if (z <= 1f) return null
        return floatArrayOf(fx * x / z + cx, fy * y / z + cy)
    }

    /**
     * 只做预测、不更新滤波器状态（用于「缺一个角点」的降级帧）。
     *
     * 与 [track] 的区别：不吸收本帧观测，因此不会把预测误差写回状态；连续超过
     * [maxCoastFrames] 个降级帧后返回 null，让调用方停止外推、等待重新锁定。
     */
    fun predictSnapshot(timeSec: Double): Snapshot? {
        val rot = rotation ?: return null
        if (!initialized) return null
        coastCount += 1
        if (coastCount > maxCoastFrames) return null
        var dt = timeSec - lastTimeSec
        if (dt < 0.0) dt = 0.0
        if (dt > maxDtSeconds) dt = maxDtSeconds
        val pred = floatArrayOf(
            pos[0] + vel[0] * dt.toFloat(),
            pos[1] + vel[1] * dt.toFloat(),
            pos[2] + vel[2] * dt.toFloat()
        )
        return Snapshot(pred, rot.copyOf(), accepted = false, coasting = true, candidates = 0)
    }

    /** 目标系角点相对三点质心的偏移；质心与 track() 里做 alpha-beta 的中心一致。 */
    private fun objectCornerRelative(colorIndex: Int, edgeMm: Float): FloatArray {
        val rel = when (colorIndex) {
            0 -> floatArrayOf(0f, 0f)
            1 -> floatArrayOf(edgeMm, 0f)
            else -> floatArrayOf(edgeMm / 2f, edgeMm * 0.8660254f)
        }
        rel[0] -= edgeMm / 2f
        rel[1] -= edgeMm * 0.28867513f
        return rel
    }

    fun reset() {
        pos.fill(0f)
        vel.fill(0f)
        rotation = null
        initialized = false
        lastTimeSec = 0.0
        coastCount = 0
        lastDepthsMm = null
    }

    // ------------------------------------------------------------------ //
    // depth roots
    // ------------------------------------------------------------------ //

    /** 枚举所有几何上成立的深度解（按平均深度升序，通常 1~2 个）。 */
    fun solveRoots(u: FloatArray, v: FloatArray, edgeMm: Float): List<DepthRoot> {
        val d12 = hypot(u[0] - u[1], v[0] - v[1])
        val d23 = hypot(u[1] - u[2], v[1] - v[2])
        val d31 = hypot(u[2] - u[0], v[2] - v[0])
        val meanImg = max((d12 + d23 + d31) / 3f, 1e-6f)
        val typical = edgeMm / meanImg
        val seeds = ArrayList<FloatArray>(8)
        lastDepthsMm?.let { seeds.add(it.copyOf()) }
        seeds.add(floatArrayOf(typical, typical, typical))
        seeds.add(floatArrayOf(0.7f * typical, 0.85f * typical, typical))
        seeds.add(floatArrayOf(typical, 0.85f * typical, 0.7f * typical))
        seeds.add(floatArrayOf(2f * typical, 2f * typical, 2f * typical))
        seeds.add(floatArrayOf(0.4f * typical, 0.4f * typical, 0.4f * typical))

        val out = ArrayList<DepthRoot>(3)
        for (seed in seeds) {
            val z = solveDepths(u, v, edgeMm, seed) ?: continue
            if (z.any { !it.isFinite() || it <= 0f }) continue
            val dup = out.any { root ->
                (0 until 3).all { i ->
                    abs(z[i] - root.depthsMm[i]) / max(root.depthsMm[i], 1e-6f) < 0.02f
                }
            }
            if (!dup) out.add(DepthRoot(z))
        }
        out.sortBy { (it.depthsMm[0] + it.depthsMm[1] + it.depthsMm[2]) / 3f }
        return out
    }

    /** 当前状态的快照（只读副本）；未初始化时返回 null。 */
    fun currentSnapshot(): Snapshot? {
        if (!initialized) return null
        val r = rotation ?: return null
        return Snapshot(pos.copyOf(), r.copyOf(), accepted = true, coasting = false, candidates = 0)
    }

    /**
     * 校验一个候选三元组：解 PnP 并返回 (rotation 行主序, 三点质心平移 mm)。
     *
     * 只用于「远距回退检测」的候选筛选——单帧里有多个几何上都像三角形的组合时，
     * 用上一帧位姿是否连续（平移接近、旋转差小）来挑，比只看形状/亮度可靠得多。
     * 无几何解时返回 null。
     */
    fun posesForTriple(u: FloatArray, v: FloatArray, edgeMm: Float): List<Pair<FloatArray, FloatArray>> {
        if (u.size < 3 || v.size < 3 || edgeMm <= 0f) return emptyList()
        val out = ArrayList<Pair<FloatArray, FloatArray>>(2)
        for (root in solveRoots(u, v, edgeMm)) {
            val pts = pointsFromDepths(root.depthsMm, u, v)
            val rot = rotationFromPoints(pts) ?: continue
            out.add(rot to centroid(pts))
        }
        return out
    }

    /** 测试/回放用：直接把状态置为给定位姿（跳过跟踪器初始化）。 */
    internal fun seedState(translationMm: FloatArray, rotationRowMajor: FloatArray) {
        require(translationMm.size == 3 && rotationRowMajor.size == 9)
        translationMm.copyInto(pos)
        vel.fill(0f)
        rotation = rotationRowMajor.copyOf()
        initialized = true
        coastCount = 0
        lastDepthsMm = null
    }

    /** 给定像素（归一化坐标）返回滤波后的位姿快照；无可行解时返回 null。 */
    fun track(
        roots: List<DepthRoot>,
        u: FloatArray,
        v: FloatArray,
        edgeMm: Float,
        timeSec: Double
    ): Snapshot? {
        if (roots.isEmpty()) return null

        val predicted = FloatArray(3)
        if (initialized) {
            val dt = (timeSec - lastTimeSec).coerceIn(1.0 / 120.0, maxDtSeconds)
            for (i in 0 until 3) predicted[i] = pos[i] + vel[i] * dt.toFloat()
        }

        var bestR: FloatArray? = null
        var bestT: FloatArray? = null
        var bestDepth: FloatArray? = null
        var bestScoreR = Float.MAX_VALUE
        var bestScoreT = Float.MAX_VALUE
        for (root in roots) {
            val pts = pointsFromDepths(root.depthsMm, u, v)
            val r = rotationFromPoints(pts) ?: continue
            val t = centroid(pts)
            val dR = rotation?.let { rotationAngleDeg(r, it) } ?: 0f
            val dT = if (initialized) distance(t, predicted) else 0f
            val useIdentity = !initialized
            val scoreR = if (useIdentity) rotationAngleDeg(r, IDENTITY_ROTATION) else dR
            val scoreT = if (useIdentity) dT else distance(t, predicted)
            if (scoreR < bestScoreR || (scoreR == bestScoreR && scoreT < bestScoreT)) {
                bestScoreR = scoreR
                bestScoreT = scoreT
                bestR = r
                bestT = t
                bestDepth = root.depthsMm
            }
        }
        val chosenR = bestR ?: return null
        val chosenT = bestT ?: return null
        val chosenDepth = bestDepth ?: return null

        if (!initialized) {
            return reinitialize(chosenT, chosenR, chosenDepth, timeSec, roots.size)
        }

        val dt = timeSec - lastTimeSec
        if (dt <= 0.0 || dt > maxDtSeconds) {
            return reinitialize(chosenT, chosenR, chosenDepth, timeSec, roots.size)
        }
        val pred = FloatArray(3) { pos[it] + vel[it] * dt.toFloat() }
        val inno = FloatArray(3) { chosenT[it] - pred[it] }
        val innoNorm = norm(inno)
        val gate = max(gateAbsMm, gateRel * norm(pred))
        val dR = rotation?.let { rotationAngleDeg(chosenR, it) } ?: 0f

        lastTimeSec = timeSec
        if (innoNorm > gate || dR > rotGateDeg) {
            // 离群：用匀速预测滑行，速度衰减；连续超限才重新锁定（可能换镜像解）。
            coastCount += 1
            for (i in 0 until 3) {
                pos[i] = pred[i]
                vel[i] *= 0.5f
            }
            if (coastCount > maxCoastFrames) {
                return reinitialize(chosenT, chosenR, chosenDepth, timeSec, roots.size)
            }
            return Snapshot(pos.copyOf(), rotation?.copyOf() ?: chosenR.copyOf(),
                accepted = false, coasting = true, candidates = roots.size)
        }

        coastCount = 0
        for (i in 0 until 3) {
            pos[i] = pred[i] + alpha * inno[i]
        }
        val invDt = (1.0 / dt).toFloat()
        for (i in 0 until 3) {
            vel[i] += beta * inno[i] * invDt
        }
        val speed = norm(vel)
        if (speed > maxSpeedMmPerSec) {
            val k = maxSpeedMmPerSec / speed
            for (i in 0 until 3) vel[i] *= k
        }
        rotation = chosenR
        lastDepthsMm = chosenDepth.copyOf()
        return Snapshot(pos.copyOf(), rotation!!.copyOf(), accepted = true,
            coasting = false, candidates = roots.size)
    }

    private fun reinitialize(
        t: FloatArray,
        r: FloatArray,
        depth: FloatArray,
        timeSec: Double,
        candidates: Int
    ): Snapshot {
        initialized = true
        coastCount = 0
        lastTimeSec = timeSec
        for (i in 0 until 3) {
            pos[i] = t[i]
            vel[i] = 0f
        }
        rotation = r.copyOf()
        lastDepthsMm = depth.copyOf()
        return Snapshot(pos.copyOf(), rotation!!.copyOf(), accepted = true,
            coasting = false, candidates = candidates)
    }

    // ------------------------------------------------------------------ //
    // geometry helpers
    // ------------------------------------------------------------------ //

    private fun solveDepths(
        u: FloatArray,
        v: FloatArray,
        edgeMm: Float,
        initMm: FloatArray,
        maxIterations: Int = 50,
        convMm: Float = 0.05f,
        minDepthMm: Float = 100f
    ): FloatArray? {
        val floor = max(minDepthMm, 1f)
        val z = FloatArray(3) { max(initMm[it], floor) }
        val b = Array(3) { floatArrayOf(u[it], v[it], 1f) }
        val targets = floatArrayOf(edgeMm, edgeMm, edgeMm)
        val iIdx = intArrayOf(0, 1, 2)
        val jIdx = intArrayOf(1, 2, 0)
        var lambda = 1e-6f
        repeat(maxIterations) {
            val p = Array(3) { floatArrayOf(z[it] * u[it], z[it] * v[it], z[it]) }
            val res = FloatArray(3)
            val jac = Array(3) { FloatArray(3) }
            var maxRes = 0f
            for (e in 0 until 3) {
                val i = iIdx[e]
                val j = jIdx[e]
                val dx = p[i][0] - p[j][0]
                val dy = p[i][1] - p[j][1]
                val dz = p[i][2] - p[j][2]
                val d = sqrt(dx * dx + dy * dy + dz * dz)
                res[e] = d - targets[e]
                maxRes = max(maxRes, abs(res[e]))
                if (d > 1e-6f) {
                    val nx = dx / d
                    val ny = dy / d
                    val nz = dz / d
                    jac[e][i] = nx * b[i][0] + ny * b[i][1] + nz * b[i][2]
                    jac[e][j] = -(nx * b[j][0] + ny * b[j][1] + nz * b[j][2])
                }
            }
            if (maxRes <= convMm) return z

            val a = Array(3) { FloatArray(3) }
            val g = FloatArray(3)
            for (row in 0 until 3) {
                for (col in 0 until 3) {
                    var s = 0f
                    for (e in 0 until 3) s += jac[e][row] * jac[e][col]
                    a[row][col] = s
                }
                var s = 0f
                for (e in 0 until 3) s -= jac[e][row] * res[e]
                g[row] = s
            }
            a[0][0] += lambda
            a[1][1] += lambda
            a[2][2] += lambda
            val delta = solve3x3(a, g) ?: return null
            for (i in 0 until 3) {
                val step = delta[i].coerceIn(-0.5f * z[i], 0.5f * z[i])
                z[i] = max(z[i] + step, floor)
            }
            lambda *= 0.5f
        }
        return null
    }

    private fun solve3x3(a: Array<FloatArray>, b: FloatArray): FloatArray? {
        fun det3(m: Array<FloatArray>): Float {
            return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        }
        val det = det3(a)
        if (abs(det) < 1e-12f) return null
        val x = FloatArray(3)
        for (col in 0 until 3) {
            val m = Array(3) { row -> a[row].copyOf() }
            for (row in 0 until 3) m[row][col] = b[row]
            x[col] = det3(m) / det
        }
        return x
    }

    private fun pointsFromDepths(depths: FloatArray, u: FloatArray, v: FloatArray): Array<FloatArray> {
        return Array(3) { floatArrayOf(depths[it] * u[it], depths[it] * v[it], depths[it]) }
    }

    private fun centroid(p: Array<FloatArray>): FloatArray {
        return floatArrayOf(
            (p[0][0] + p[1][0] + p[2][0]) / 3f,
            (p[0][1] + p[1][1] + p[2][1]) / 3f,
            (p[0][2] + p[1][2] + p[2][2]) / 3f
        )
    }

    /**
     * 三点定姿：x 轴 = 红->绿，z 轴 = x × (蓝 - 中心)，y = z × x。
     * 与 [PnPDistanceCalculator.calculate6DOFPose] 的欧拉角约定一致。
     */
    private fun rotationFromPoints(p: Array<FloatArray>): FloatArray? {
        val c = centroid(p)
        var x0 = p[1][0] - p[0][0]
        var x1 = p[1][1] - p[0][1]
        var x2 = p[1][2] - p[0][2]
        val xl = sqrt(x0 * x0 + x1 * x1 + x2 * x2)
        if (xl < 1e-6f) return null
        x0 /= xl; x1 /= xl; x2 /= xl

        val bx = p[2][0] - c[0]
        val by = p[2][1] - c[1]
        val bz = p[2][2] - c[2]
        var z0 = x1 * bz - x2 * by
        var z1 = x2 * bx - x0 * bz
        var z2 = x0 * by - x1 * bx
        val zl = sqrt(z0 * z0 + z1 * z1 + z2 * z2)
        if (zl < 1e-6f) return null
        z0 /= zl; z1 /= zl; z2 /= zl

        val y0 = z1 * x2 - z2 * x1
        val y1 = z2 * x0 - z0 * x2
        val y2 = z0 * x1 - z1 * x0

        // 行主序，列 = (x, y, z)
        return floatArrayOf(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    }

    /** 两个旋转矩阵之间的测地角（度）。 */
    fun rotationAngleDeg(a: FloatArray, b: FloatArray): Float {
        var trace = 0f
        for (i in 0 until 3) {
            for (j in 0 until 3) trace += a[i * 3 + j] * b[i * 3 + j]
        }
        val c = ((trace - 1f) / 2f).coerceIn(-1f, 1f)
        return Math.toDegrees(acos(c.toDouble())).toFloat()
    }

    private fun norm(v: FloatArray): Float {
        return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    }

    private fun distance(a: FloatArray, b: FloatArray): Float {
        val dx = a[0] - b[0]
        val dy = a[1] - b[1]
        val dz = a[2] - b[2]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    private fun hypot(a: Float, b: Float): Float = sqrt(a * a + b * b)

    companion object {
        private val IDENTITY_ROTATION = floatArrayOf(
            1f, 0f, 0f,
            0f, 1f, 0f,
            0f, 0f, 1f
        )
    }
}
