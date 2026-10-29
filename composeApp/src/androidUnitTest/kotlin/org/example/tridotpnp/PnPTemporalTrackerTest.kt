package org.example.tridotpnp

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * 纯几何测试：不依赖 Android，直接喂归一化像素坐标。
 *
 * 参考实现与 Python 侧 `research/tests/test_temporal.py` 对应；
 * Python 侧的 A/B 结果见 `research/results/temporal_ab/`。
 */
class PnPTemporalTrackerTest {

    private val edge = 100f
    private val obj = arrayOf(
        floatArrayOf(0f, 0f, 0f),
        floatArrayOf(edge, 0f, 0f),
        floatArrayOf(edge / 2f, edge * sqrt(3f) / 2f, 0f)
    )

    /** 相机点 = R * obj + t（R 行主序）。 */
    private fun cameraPoints(rot: FloatArray, t: FloatArray): Array<FloatArray> {
        return Array(3) { i ->
            floatArrayOf(
                rot[0] * obj[i][0] + rot[1] * obj[i][1] + rot[2] * obj[i][2] + t[0],
                rot[3] * obj[i][0] + rot[4] * obj[i][1] + rot[5] * obj[i][2] + t[1],
                rot[6] * obj[i][0] + rot[7] * obj[i][1] + rot[8] * obj[i][2] + t[2]
            )
        }
    }

    private fun normalizedPixels(cam: Array<FloatArray>): Pair<FloatArray, FloatArray> {
        val u = FloatArray(3) { i -> cam[i][0] / cam[i][2] }
        val v = FloatArray(3) { i -> cam[i][1] / cam[i][2] }
        return u to v
    }

    private fun rotY(deg: Float): FloatArray {
        val r = Math.toRadians(deg.toDouble())
        val c = cos(r).toFloat()
        val s = sin(r).toFloat()
        return floatArrayOf(c, 0f, s, 0f, 1f, 0f, -s, 0f, c)
    }

    /** 目标质心在相机系下的坐标（跟踪器输出的是质心平移）。 */
    private fun centroidOf(cam: Array<FloatArray>): FloatArray = floatArrayOf(
        (cam[0][0] + cam[1][0] + cam[2][0]) / 3f,
        (cam[0][1] + cam[1][1] + cam[2][1]) / 3f,
        (cam[0][2] + cam[1][2] + cam[2][2]) / 3f
    )

    private fun translationError(a: FloatArray, b: FloatArray): Float {
        val dx = a[0] - b[0]
        val dy = a[1] - b[1]
        val dz = a[2] - b[2]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    @Test
    fun solvesStaticFrontFacingTarget() {
        val rot = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        val t = floatArrayOf(0f, 0f, 800f)
        val cam = cameraPoints(rot, t)
        val expected = centroidOf(cam)
        val (u, v) = normalizedPixels(cam)
        val tracker = PnPTemporalTracker()
        val roots = tracker.solveRoots(u, v, edge)
        assertTrue(roots.isNotEmpty())
        val snap = tracker.track(roots, u, v, edge, 0.0)!!
        assertTrue(translationError(snap.translationMm, expected) < 1f, "t=${snap.translationMm.toList()}")
        assertTrue(tracker.rotationAngleDeg(snap.rotation, rot) < 2f)
    }

    @Test
    fun rootsContainGroundTruthDepths() {
        val rot = rotY(15f)
        val t = floatArrayOf(30f, -20f, 900f)
        val cam = cameraPoints(rot, t)
        val (u, v) = normalizedPixels(cam)
        val tracker = PnPTemporalTracker()
        val roots = tracker.solveRoots(u, v, edge)
        assertTrue(roots.isNotEmpty())
        val gt = FloatArray(3) { cam[it][2] }
        val best = roots.minOf { r ->
            (0 until 3).maxOf { i -> abs(r.depthsMm[i] - gt[i]) / gt[i] }
        }
        assertTrue(best < 0.02f, "root depths vs gt: $best")
    }

    @Test
    fun tracksMovingTargetWithoutBranchFlips() {
        val tracker = PnPTemporalTracker()
        var prevRot: FloatArray? = null
        var maxJump = 0f
        var lastT = FloatArray(3)
        for (k in 0 until 40) {
            val rot = rotY(18f)
            val t = floatArrayOf(60f - 2f * k, 10f, 750f - 4f * k)
            val (u, v) = normalizedPixels(cameraPoints(rot, t))
            val snap = tracker.track(tracker.solveRoots(u, v, edge), u, v, edge, k / 30.0)!!
            prevRot?.let { maxJump = maxOf(maxJump, tracker.rotationAngleDeg(snap.rotation, it)) }
            prevRot = snap.rotation
            lastT = snap.translationMm
        }
        assertTrue(maxJump < 30f, "max frame-to-frame rotation jump $maxJump")
        assertTrue(abs(lastT[2] - (750f - 4f * 39)) < 40f, "z=${lastT[2]}")
    }

    @Test
    fun rejectsOutlierAndCoasts() {
        val tracker = PnPTemporalTracker()
        val rot = rotY(10f)
        var snap: PnPTemporalTracker.Snapshot? = null
        for (k in 0 until 10) {
            val t = floatArrayOf(0f, 0f, 800f)
            val (u, v) = normalizedPixels(cameraPoints(rot, t))
            snap = tracker.track(tracker.solveRoots(u, v, edge), u, v, edge, k / 30.0)
            assertEquals(true, snap!!.accepted)
        }
        val zBefore = snap!!.translationMm[2]
        val bad = floatArrayOf(0f, 0f, 1400f)
        val (u, v) = normalizedPixels(cameraPoints(rot, bad))
        val outlier = tracker.track(tracker.solveRoots(u, v, edge), u, v, edge, 10 / 30.0)!!
        assertTrue(outlier.coasting && !outlier.accepted)
        assertTrue(abs(outlier.translationMm[2] - zBefore) < 60f, "z=${outlier.translationMm[2]}")
    }

    @Test
    fun resetClearsState() {
        val tracker = PnPTemporalTracker()
        val rot = rotY(0f)
        val t = floatArrayOf(0f, 0f, 800f)
        val cam = cameraPoints(rot, t)
        val expected = centroidOf(cam)
        val (u, v) = normalizedPixels(cam)
        tracker.track(tracker.solveRoots(u, v, edge), u, v, edge, 0.0)
        tracker.reset()
        // 复位后重新开始，不能沿用旧的“已在跟踪”状态
        val snap = tracker.track(tracker.solveRoots(u, v, edge), u, v, edge, 100.0)!!
        assertTrue(snap.accepted && !snap.coasting)
        assertTrue(translationError(snap.translationMm, expected) < 1f)
    }

    @Test
    fun projectsMissingCornerFromTrackedPose() {
        val rot = rotY(10f)
        val t = floatArrayOf(30f, 10f, 700f)
        val cam = cameraPoints(rot, t)
        val (u, v) = normalizedPixels(cam)
        val tracker = PnPTemporalTracker()
        val roots = tracker.solveRoots(u, v, edge)
        val snap = tracker.track(roots, u, v, edge, 10.0)!!
        assertTrue(snap.accepted)

        val fx = 1108.5f
        val cx = 640f
        val cy = 360f
        for (i in 0..2) {
            val predicted = tracker.projectCorner(i, edge, fx, fx, cx, cy, 10.0)!!
            val expectedX = fx * cam[i][0] / cam[i][2] + cx
            val expectedY = fx * cam[i][1] / cam[i][2] + cy
            assertTrue(
                abs(predicted[0] - expectedX) < 1f && abs(predicted[1] - expectedY) < 1f,
                "corner $i: predicted=${predicted.toList()} expected=($expectedX, $expectedY)"
            )
        }
    }

    @Test
    fun projectCornerNeedsInitializedState() {
        val tracker = PnPTemporalTracker()
        assertEquals(null, tracker.projectCorner(1, edge, 1108.5f, 1108.5f, 640f, 360f, 0.0))
    }

    @Test
    fun projectCornerFollowsConstantVelocity() {
        // 目标沿 +X 匀速移动（200 mm/s，帧间隔 0.1s）；跟踪 6 帧后外推到最后一帧观测后 0.1s。
        // 期望值按「最后一次观测中心 + 速度 * dt」计算；alpha-beta 有稳态滞后，容差 15px。
        val rot = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        val tracker = PnPTemporalTracker()
        val fx = 1108.5f
        val cx = 640f
        val cy = 360f
        var lastCenterX = 0f
        var lastTime = 0.0
        for (frame in 0..5) {
            val t = floatArrayOf(frame * 20f, 0f, 700f)
            val cam = cameraPoints(rot, t)
            val (u, v) = normalizedPixels(cam)
            tracker.track(tracker.solveRoots(u, v, edge), u, v, edge, frame * 0.1)
            lastCenterX = frame * 20f
            lastTime = frame * 0.1
        }
        val futureTime = lastTime + 0.1
        val predicted = tracker.projectCorner(0, edge, fx, fx, cx, cy, futureTime)!!
        val expectedCenterX = lastCenterX + 20f // 200 mm/s * 0.1 s
        val expectedX = fx * expectedCenterX / 700f + cx
        val expectedY = fx * 0f / 700f + cy
        // alpha-beta 对匀速目标有约一帧位移的稳态滞后（(1-alpha)/beta * v*dt ≈ 20mm ≈ 31px），
        // 这里容差放到 35px，只验证「外推方向正确且远小于静止假设的误差（约 200mm≈316px）」。
        assertTrue(
            abs(predicted[0] - expectedX) < 35f && abs(predicted[1] - expectedY) < 15f,
            "predicted=${predicted.toList()} expected=($expectedX, $expectedY)"
        )
    }

    private fun fy(fx: Float): Float = fx

    @Test
    fun predictSnapshotNeedsLock() {
        val tracker = PnPTemporalTracker()
        assertEquals(null, tracker.predictSnapshot(0.0))
    }

    @Test
    fun predictSnapshotExtrapolatesAndIsBounded() {
        val rot = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        val cam = cameraPoints(rot, floatArrayOf(0f, 0f, 700f))
        val (u, v) = normalizedPixels(cam)
        val tracker = PnPTemporalTracker()
        tracker.track(tracker.solveRoots(u, v, edge), u, v, edge, 0.0)
        val center = centroidOf(cam)

        val snap = tracker.predictSnapshot(0.05)!!
        assertTrue(translationError(snap.translationMm, center) < 1f)
        assertTrue(snap.coasting && !snap.accepted)

        // 上面已经用掉 1 帧降级额度；再来 4 帧仍有效，第 6 帧必须停在外推等待重锁。
        for (i in 1..4) {
            assertTrue(tracker.predictSnapshot(0.05 + 0.01 * i) != null)
        }
        assertEquals(null, tracker.predictSnapshot(0.2))
    }

    @Test
    fun predictSnapshotDoesNotUpdateFilterState() {
        val rot = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        val cam = cameraPoints(rot, floatArrayOf(0f, 0f, 700f))
        val (u, v) = normalizedPixels(cam)
        val tracker = PnPTemporalTracker()
        tracker.track(tracker.solveRoots(u, v, edge), u, v, edge, 0.0)
        repeat(3) { tracker.predictSnapshot(0.01) }
        // 外推不应改状态：重新吸收同一观测后仍与中心一致。
        val snap = tracker.track(tracker.solveRoots(u, v, edge), u, v, edge, 0.1)!!
        assertTrue(translationError(snap.translationMm, centroidOf(cam)) < 1f)
    }
}
