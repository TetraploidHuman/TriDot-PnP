package org.example.tridotpnp

import androidx.compose.ui.geometry.Offset
import kotlin.math.*
import kotlin.math.abs

/**
 * PnP距离计算�?
 * 用于计算摄像头到目标的距离和方向
 */
class PnPDistanceCalculator {

    var estimatedFovDegrees: Float = 60f
    var depthInitFallbackMm: Float = 1000f
    var solverMaxIterations: Int = 50
    var solverConvergenceErrorMm: Float = 0.1f
    var solverMinDepthMm: Float = 100f
    private var lastDepthsMm: FloatArray? = null

    data class Point3D(
        val x: Float,                 // X坐标（毫米）
        val y: Float,                 // Y坐标（毫米）
        val z: Float                  // Z坐标（毫米，深度�?
    )

    data class Pose6DOF(
        val position: Point3D,        // 位置（中心点�?
        val roll: Float,              // 横滚角（度）
        val pitch: Float,             // 俯仰角（度）
        val yaw: Float                // 偏航角（度）
    )

    data class PnPResult(
        val distance: Float,           // 到摄像头的平均距离（毫米�?
        val azimuth: Float,           // 方位角（度）
        val elevation: Float,         // 仰角（度�?
        val point1Distance: Float,    // �?到摄像头的距�?
        val point2Distance: Float,    // �?到摄像头的距�?
        val point1Coords: Point3D,    // �?�?D坐标
        val point2Coords: Point3D,    // �?�?D坐标
        val centerCoords: Point3D,    // 中心点的3D坐标
        val pose6DOF: Pose6DOF? = null  // 完整6DOF姿态（�?点模式）
    )

    /**
     * 使用两点PnP算法计算距离和方�?
     *
     * @param point1 第一个亮点的像素坐标
     * @param point2 第二个亮点的像素坐标
     * @param realDistance 两点之间的实际距离（毫米�?
     * @param focalLength 相机焦距（像素单位），默认使用估算�?
     * @param imageWidth 图像宽度（像素）
     * @param imageHeight 图像高度（像素）
     * @return PnP计算结果
     */
    fun calculate2PointPnP(
        point1: Offset,
        point2: Offset,
        realDistance: Float,
        focalLength: Float? = null,
        imageWidth: Int,
        imageHeight: Int
    ): PnPResult {
        // 计算相机焦距（如果未提供�?
        val fx = focalLength ?: estimateFocalLength(imageWidth, imageHeight)
        val fy = fx // 假设焦距在x和y方向相同

        // 主点（图像中心）
        val cx = imageWidth / 2.0f
        val cy = imageHeight / 2.0f

        // 将像素坐标转换为归一化坐�?
        val p1x = (point1.x - cx) / fx
        val p1y = (point1.y - cy) / fy
        val p2x = (point2.x - cx) / fx
        val p2y = (point2.y - cy) / fy

        // 计算图像平面上两点之间的距离
        val imageDistance = sqrt((p2x - p1x).pow(2) + (p2y - p1y).pow(2))

        // 使用相似三角形原理计算深�?
        // realDistance / depth �?imageDistance / 1.0
        val averageDepth = if (imageDistance > 0.001f) {
            realDistance / imageDistance
        } else {
            realDistance * fx // 退回到简单估�?
        }

        // 计算每个点的深度（假设在同一深度平面�?
        val depth1 = averageDepth
        val depth2 = averageDepth

        // 计算3D坐标
        val x1 = p1x * depth1
        val y1 = p1y * depth1
        val z1 = depth1

        val x2 = p2x * depth2
        val y2 = p2y * depth2
        val z2 = depth2

        // 计算中点�?D坐标
        val centerX = (x1 + x2) / 2
        val centerY = (y1 + y2) / 2
        val centerZ = (z1 + z2) / 2

        // 计算方向向量
        val dirX = x2 - x1
        val dirY = y2 - y1
        val dirZ = z2 - z1

        // 计算方位角（azimuth�? 在XZ平面的角�?
        val azimuth = Math.toDegrees(atan2(dirX.toDouble(), dirZ.toDouble())).toFloat()

        // 计算仰角（elevation�?
        val horizontalDist = sqrt(dirX.pow(2) + dirZ.pow(2))
        val elevation = Math.toDegrees(atan2(dirY.toDouble(), horizontalDist.toDouble())).toFloat()

        // 计算到摄像头的距�?
        val distance1 = sqrt(x1.pow(2) + y1.pow(2) + z1.pow(2))
        val distance2 = sqrt(x2.pow(2) + y2.pow(2) + z2.pow(2))
        val averageDistance = (distance1 + distance2) / 2

        return PnPResult(
            distance = averageDistance,
            azimuth = azimuth,
            elevation = elevation,
            point1Distance = distance1,
            point2Distance = distance2,
            point1Coords = Point3D(x1, y1, z1),
            point2Coords = Point3D(x2, y2, z2),
            centerCoords = Point3D(centerX, centerY, centerZ)
        )
    }

    /**
     * 估算相机焦距
     * 基于常见的智能手机相机参�?
     */
    private fun estimateFocalLength(imageWidth: Int, imageHeight: Int): Float {
        // 假设水平视场角约�?0�?
        val fovDegrees = estimatedFovDegrees.toDouble().coerceIn(10.0, 170.0)
        val fovRadians = Math.toRadians(fovDegrees)

        // 焦距 = imageWidth / (2 * tan(fov/2))
        return (imageWidth / (2.0 * tan(fovRadians / 2.0))).toFloat()
    }

    /**
     * 格式化距离显�?
     */
    fun formatDistance(distance: Float): String {
        return when {
            distance < 1000 -> "%.0f mm".format(distance)
            distance < 10000 -> "%.2f m".format(distance / 1000)
            else -> "%.1f m".format(distance / 1000)
        }
    }

    /**
     * 格式化角度显�?
     */
    fun formatAngle(angle: Float): String {
        return "%.1f°".format(angle)
    }

    /**
     * 格式�?D坐标显示
     */
    fun formatCoordinate(value: Float): String {
        return when {
            abs(value) < 1000 -> "%.0f".format(value)
            else -> "%.1f".format(value / 1000)
        }
    }

    /**
     * 格式�?D坐标单位
     */
    fun getCoordinateUnit(value: Float): String {
        return if (abs(value) < 1000) "mm" else "m"
    }

    /**
     * 格式化完整坐�?
     */
    fun formatPoint3D(point: Point3D): String {
        val useMeters = maxOf(abs(point.x), abs(point.y), abs(point.z)) >= 1000
        return if (useMeters) {
            "(%.2f, %.2f, %.2f) m".format(
                point.x / 1000,
                point.y / 1000,
                point.z / 1000
            )
        } else {
            "(%.0f, %.0f, %.0f) mm".format(point.x, point.y, point.z)
        }
    }

    /**
     * 使用三点PnP算法计算完整6DOF姿�?
     *
     * @param redPoint 红色LED的像素坐�?
     * @param greenPoint 绿色LED的像素坐�?
     * @param bluePoint 蓝色LED的像素坐�?
     * @param triangleEdgeLength 正三角形的边长（毫米�?
     * @param focalLength 相机焦距（像素单位），默认使用估算�?
     * @param imageWidth 图像宽度（像素）
     * @param imageHeight 图像高度（像素）
     * @return PnP计算结果，包含完�?DOF姿�?
     */
    fun calculate3PointPnP(
        redPoint: Offset,
        greenPoint: Offset,
        bluePoint: Offset,
        triangleEdgeLength: Float,
        focalLength: Float? = null,
        imageWidth: Int,
        imageHeight: Int
    ): PnPResult {
        // 计算相机焦距
        val fx = focalLength ?: estimateFocalLength(imageWidth, imageHeight)
        val fy = fx

        // 主点（图像中心）
        val cx = imageWidth / 2.0f
        val cy = imageHeight / 2.0f

        // 将像素坐标转换为归一化坐�?
        val pr_x = (redPoint.x - cx) / fx
        val pr_y = (redPoint.y - cy) / fy
        val pg_x = (greenPoint.x - cx) / fx
        val pg_y = (greenPoint.y - cy) / fy
        val pb_x = (bluePoint.x - cx) / fx
        val pb_y = (bluePoint.y - cy) / fy

        // 计算图像平面上的归一化距�?
        val dist_rg_image = sqrt((pg_x - pr_x).pow(2) + (pg_y - pr_y).pow(2))
        val dist_gb_image = sqrt((pb_x - pg_x).pow(2) + (pb_y - pg_y).pow(2))
        val dist_br_image = sqrt((pr_x - pb_x).pow(2) + (pr_y - pb_y).pow(2))

        // 使用3点PnP算法计算三个点的独立深度
        // 基于已知边长约束，通过迭代求解深度
        val depths = solve3PointDepths(
            pr_x, pr_y, pg_x, pg_y, pb_x, pb_y,
            triangleEdgeLength, triangleEdgeLength, triangleEdgeLength
        )

        val depthRed = depths.first
        val depthGreen = depths.second
        val depthBlue = depths.third

        // 计算三个点的3D坐标（使用各自的深度�?
        val redCoords = Point3D(pr_x * depthRed, pr_y * depthRed, depthRed)
        val greenCoords = Point3D(pg_x * depthGreen, pg_y * depthGreen, depthGreen)
        val blueCoords = Point3D(pb_x * depthBlue, pb_y * depthBlue, depthBlue)

        // 计算中心�?
        val centerX = (redCoords.x + greenCoords.x + blueCoords.x) / 3f
        val centerY = (redCoords.y + greenCoords.y + blueCoords.y) / 3f
        val centerZ = (redCoords.z + greenCoords.z + blueCoords.z) / 3f
        val centerCoords = Point3D(centerX, centerY, centerZ)

        // 计算6DOF姿�?
        val pose = calculate6DOFPose(redCoords, greenCoords, blueCoords)

        // 计算距离和方�?
        val distance1 = sqrt(redCoords.x.pow(2) + redCoords.y.pow(2) + redCoords.z.pow(2))
        val distance2 = sqrt(greenCoords.x.pow(2) + greenCoords.y.pow(2) + greenCoords.z.pow(2))
        val averageDistance = (distance1 + distance2 + sqrt(blueCoords.x.pow(2) + blueCoords.y.pow(2) + blueCoords.z.pow(2))) / 3

        // 计算方位角和仰角（使用红绿连线的方向�?
        val dirX = greenCoords.x - redCoords.x
        val dirY = greenCoords.y - redCoords.y
        val dirZ = greenCoords.z - redCoords.z

        val azimuth = Math.toDegrees(atan2(dirX.toDouble(), dirZ.toDouble())).toFloat()
        val horizontalDist = sqrt(dirX.pow(2) + dirZ.pow(2))
        val elevation = Math.toDegrees(atan2(dirY.toDouble(), horizontalDist.toDouble())).toFloat()

        return PnPResult(
            distance = averageDistance,
            azimuth = azimuth,
            elevation = elevation,
            point1Distance = distance1,
            point2Distance = distance2,
            point1Coords = redCoords,
            point2Coords = greenCoords,
            centerCoords = centerCoords,
            pose6DOF = pose
        )
    }

    /**
     * 求解三个点的深度，使得它们满足已知的边长约束
     * 使用迭代方法求解非线性方程组
     *
     * @param u1, v1 �?的归一化坐�?
     * @param u2, v2 �?的归一化坐�?
     * @param u3, v3 �?的归一化坐�?
     * @param d12 �?和点2之间的实际距�?
     * @param d23 �?和点3之间的实际距�?
     * @param d31 �?和点1之间的实际距�?
     * @return Triple(�?深度, �?深度, �?深度)
     */
    /**
     * 三点 PnP（时序滤波 + 双解消歧）。
     *
     * 与 [calculate3PointPnP] 的区别：
     * 1. 用多初值阻尼 Newton 枚举三深度的所有几何解（镜像解），再按与上一帧的姿态连续性
     *    选根；没有历史时选“最接近单位旋转”的根（板子大致正对相机）。
     * 2. 对目标中心做 alpha-beta 滤波 + 新息门限，抑制单帧离群（跳到别的光斑 /
     *    镜像解）造成的数值与姿态跳变。
     *
     * @param timestampSec 帧时间戳（秒），必须单调递增；用 System.nanoTime() 换算。
     * @param tracker 帧间保持状态的跟踪器；切换目标或长时间丢失后应调用 reset()。
     * @return 滤波后的结果；本帧无可行几何解时返回 null（调用方可退回原始解或保持旧值）。
     */
    /**
     * 预测缺失角点的像素位置（时序补全）。
     *
     * 只在跟踪器已锁定、且预测点落在画面内时返回；否则返回 null，
     * 调用方应退回到「等三点齐全再解算」的行为。
     */
    fun predictMissingPixel(
        color: LedColor,
        edgeMm: Float,
        imageWidth: Int,
        imageHeight: Int,
        timestampSec: Double,
        tracker: PnPTemporalTracker
    ): Offset? {
        val index = when (color) {
            LedColor.RED -> 0
            LedColor.GREEN -> 1
            LedColor.BLUE -> 2
            else -> return null
        }
        val fx = estimateFocalLength(imageWidth, imageHeight)
        val w = imageWidth.toFloat()
        val h = imageHeight.toFloat()
        val uv = tracker.projectCorner(
            colorIndex = index,
            edgeMm = edgeMm,
            fx = fx,
            fy = fx,
            cx = w / 2f,
            cy = h / 2f,
            timeSec = timestampSec
        ) ?: return null
        if (uv[0] < 0f || uv[1] < 0f || uv[0] > w || uv[1] > h) return null
        return Offset(uv[0], uv[1])
    }

    /**
     * 三点 PnP 的「缺一降级」版本：不吸收本帧观测，只用上一帧位姿 + 速度预测。
     *
     * 与 [calculate3PointPnPTracked] 的区别：它会把本帧观测写进跟踪器状态；这个版本
     * 只在检测少一个颜色点、且跟踪器已锁定时使用，保证预测误差不会污染滤波器。
     */
    fun calculate3PointPnPCompleted(
        triangleEdgeLength: Float,
        timestampSec: Double,
        tracker: PnPTemporalTracker
    ): PnPResult? {
        val snapshot = tracker.predictSnapshot(timestampSec) ?: return null
        return buildTrackedResult(snapshot, triangleEdgeLength)
    }

    fun calculate3PointPnPTracked(
        redPoint: Offset,
        greenPoint: Offset,
        bluePoint: Offset,
        triangleEdgeLength: Float,
        imageWidth: Int,
        imageHeight: Int,
        timestampSec: Double,
        tracker: PnPTemporalTracker
    ): PnPResult? {
        val fx = estimateFocalLength(imageWidth, imageHeight)
        val fy = fx
        val cx = imageWidth / 2.0f
        val cy = imageHeight / 2.0f

        val u = floatArrayOf(
            (redPoint.x - cx) / fx,
            (greenPoint.x - cx) / fx,
            (bluePoint.x - cx) / fx
        )
        val v = floatArrayOf(
            (redPoint.y - cy) / fy,
            (greenPoint.y - cy) / fy,
            (bluePoint.y - cy) / fy
        )

        val roots = tracker.solveRoots(u, v, triangleEdgeLength)
        val snapshot = tracker.track(roots, u, v, triangleEdgeLength, timestampSec)
            ?: return null
        return buildTrackedResult(snapshot, triangleEdgeLength)
    }

    private fun buildTrackedResult(
        snapshot: PnPTemporalTracker.Snapshot,
        triangleEdgeLength: Float
    ): PnPResult {
        val r = snapshot.rotation
        val t = snapshot.translationMm
        val edge = triangleEdgeLength
        val sqrt3 = sqrt(3f)

        // 目标系：红(0,0,0)，绿(edge,0,0)，蓝(edge/2, edge*sqrt3/2, 0)
        val objX = floatArrayOf(0f, edge, edge / 2f)
        val objY = floatArrayOf(0f, 0f, edge * sqrt3 / 2f)
        val ocx = edge / 2f
        val ocy = edge * sqrt3 / 6f

        fun cameraPoint(i: Int): Point3D {
            val dx = objX[i] - ocx
            val dy = objY[i] - ocy
            val x = r[0] * dx + r[1] * dy + t[0]
            val y = r[3] * dx + r[4] * dy + t[1]
            val z = r[6] * dx + r[7] * dy + t[2]
            return Point3D(x, y, z)
        }

        val redCoords = cameraPoint(0)
        val greenCoords = cameraPoint(1)
        val blueCoords = cameraPoint(2)
        val centerCoords = Point3D(t[0], t[1], t[2])

        val distance1 = sqrt(redCoords.x.pow(2) + redCoords.y.pow(2) + redCoords.z.pow(2))
        val distance2 = sqrt(greenCoords.x.pow(2) + greenCoords.y.pow(2) + greenCoords.z.pow(2))
        val distance3 = sqrt(blueCoords.x.pow(2) + blueCoords.y.pow(2) + blueCoords.z.pow(2))
        val averageDistance = (distance1 + distance2 + distance3) / 3f

        val dirX = greenCoords.x - redCoords.x
        val dirY = greenCoords.y - redCoords.y
        val dirZ = greenCoords.z - redCoords.z
        val azimuth = Math.toDegrees(atan2(dirX.toDouble(), dirZ.toDouble())).toFloat()
        val horizontalDist = sqrt(dirX.pow(2) + dirZ.pow(2))
        val elevation = Math.toDegrees(atan2(dirY.toDouble(), horizontalDist.toDouble())).toFloat()

        return PnPResult(
            distance = averageDistance,
            azimuth = azimuth,
            elevation = elevation,
            point1Distance = distance1,
            point2Distance = distance2,
            point1Coords = redCoords,
            point2Coords = greenCoords,
            centerCoords = centerCoords,
            pose6DOF = poseFromRotation(r, centerCoords)
        )
    }

    /** 由“目标系 -> 相机系”旋转矩阵（行主序）求欧拉角，约定与 [calculate6DOFPose] 一致。 */
    private fun poseFromRotation(r: FloatArray, center: Point3D): Pose6DOF {
        // 列 = 目标系 x/y/z 轴在相机系下的方向
        val zx = r[2]
        val zy = r[5]
        val zz = r[8]
        val yx = r[1]
        val yy = r[4]
        val yz = r[7]

        val pitch = Math.toDegrees(asin(-zx.toDouble())).toFloat()
        val yaw = Math.toDegrees(atan2(zy.toDouble(), zz.toDouble())).toFloat()
        val roll = Math.toDegrees(atan2(yy.toDouble(), yz.toDouble())).toFloat()
        return Pose6DOF(position = center, roll = roll, pitch = pitch, yaw = yaw)
    }

    /**
     * Solve three depths so that the pairwise 3-D distances match the known target
     * edges.
     *
     * Damped Newton solve on the three unknowns (one depth per dot) using the exact
     * pinhole point model P_i = (u_i * z_i, v_i * z_i, z_i) and an analytic Jacobian.
     * The previous multiplicative update
     *     z_i *= 1 + (target / current - 1) * learning_rate
     * was an undamped fixed-point iteration: on noiseless ground-truth pixels it
     * increased the edge error every iteration until all three depths hit the
     * min-depth floor (see research/results/app_pnp_audit and
     * research/tridot_temporal/app_pnp.py).  This Newton version converges in a
     * handful of iterations.  The previous frame seeds the iteration so that
     * continuous motion stays on the same geometrically valid root.
     */
    private fun solve3PointDepths(
        u1: Float, v1: Float,
        u2: Float, v2: Float,
        u3: Float, v3: Float,
        d12: Float,
        d23: Float,
        d31: Float
    ): Triple<Float, Float, Float> {
        val floor = solverMinDepthMm.coerceAtLeast(1f)
        val dist12Image = sqrt((u2 - u1).pow(2) + (v2 - v1).pow(2))
        val dist23Image = sqrt((u3 - u2).pow(2) + (v3 - v2).pow(2))
        val dist31Image = sqrt((u1 - u3).pow(2) + (v1 - v3).pow(2))

        fun initDepth(edgeA: Float, imageA: Float, edgeB: Float, imageB: Float): Float =
            if (imageA > 0.001f && imageB > 0.001f) {
                sqrt((edgeA / imageA) * (edgeB / imageB)).coerceAtLeast(floor)
            } else {
                depthInitFallbackMm.coerceAtLeast(floor)
            }

        val bx = floatArrayOf(u1, u2, u3)
        val by = floatArrayOf(v1, v2, v3)
        val z = FloatArray(3)
        val previous = lastDepthsMm
        if (previous != null && previous.size == 3 && previous.all { it > 0f }) {
            for (i in 0 until 3) z[i] = previous[i].coerceAtLeast(floor)
        } else {
            z[0] = initDepth(d12, dist12Image, d31, dist31Image)
            z[1] = initDepth(d12, dist12Image, d23, dist23Image)
            z[2] = initDepth(d23, dist23Image, d31, dist31Image)
        }

        val edgeI = intArrayOf(0, 1, 2)
        val edgeJ = intArrayOf(1, 2, 0)
        val target = floatArrayOf(d12, d23, d31)
        var lambda = 1e-6f
        var converged = false

        for (iteration in 0 until solverMaxIterations.coerceAtLeast(1)) {
            val px = FloatArray(3)
            val py = FloatArray(3)
            val pz = FloatArray(3)
            for (i in 0 until 3) {
                px[i] = bx[i] * z[i]
                py[i] = by[i] * z[i]
                pz[i] = z[i]
            }
            val nx = FloatArray(3)
            val ny = FloatArray(3)
            val nz = FloatArray(3)
            val residual = FloatArray(3)
            var maxError = 0f
            for (e in 0 until 3) {
                val i = edgeI[e]
                val j = edgeJ[e]
                val dx = px[j] - px[i]
                val dy = py[j] - py[i]
                val dz = pz[j] - pz[i]
                val dist = sqrt(dx * dx + dy * dy + dz * dz).coerceAtLeast(1e-3f)
                nx[e] = dx / dist
                ny[e] = dy / dist
                nz[e] = dz / dist
                residual[e] = dist - target[e]
                maxError = maxOf(maxError, abs(residual[e]))
            }
            if (maxError < solverConvergenceErrorMm.coerceAtLeast(1e-4f)) {
                converged = true
                break
            }

            val jac = Array(3) { FloatArray(3) }
            for (e in 0 until 3) {
                val i = edgeI[e]
                val j = edgeJ[e]
                jac[e][i] = -(nx[e] * bx[i] + ny[e] * by[i] + nz[e])
                jac[e][j] = (nx[e] * bx[j] + ny[e] * by[j] + nz[e])
            }

            val normal = Array(3) { FloatArray(3) }
            val rhs = FloatArray(3)
            for (a in 0 until 3) {
                for (b in 0 until 3) {
                    var s = 0f
                    for (e in 0 until 3) s += jac[e][a] * jac[e][b]
                    normal[a][b] = if (a == b) s + lambda else s
                }
                var s = 0f
                for (e in 0 until 3) s += jac[e][a] * residual[e]
                rhs[a] = -s
            }

            val step = solve3x3(normal, rhs) ?: break
            for (k in 0 until 3) {
                val delta = step[k].coerceIn(-0.5f * z[k], 0.5f * z[k])
                z[k] = (z[k] + delta).coerceAtLeast(floor)
            }
            lambda = maxOf(lambda * 0.5f, 1e-9f)
        }

        lastDepthsMm = if (converged) floatArrayOf(z[0], z[1], z[2]) else null
        return Triple(z[0], z[1], z[2])
    }

    /** Cramer's rule for a 3x3 linear system; null when the matrix is singular. */
    private fun solve3x3(a: Array<FloatArray>, b: FloatArray): FloatArray? {
        fun det(m: Array<FloatArray>): Float =
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])

        val d = det(a)
        if (abs(d) < 1e-12f) return null
        val out = FloatArray(3)
        for (k in 0 until 3) {
            val m = Array(3) { r -> FloatArray(3) { c -> if (c == k) b[r] else a[r][c] } }
            out[k] = det(m) / d
        }
        return out
    }


    /**
     * 从三个点计算6DOF姿态（欧拉角）
     */
    private fun calculate6DOFPose(
        red: Point3D,
        green: Point3D,
        blue: Point3D
    ): Pose6DOF {
        // 计算中心�?
        val center = Point3D(
            (red.x + green.x + blue.x) / 3f,
            (red.y + green.y + blue.y) / 3f,
            (red.z + green.z + blue.z) / 3f
        )

        // 计算本地坐标�?
        // X轴：从红到绿
        val xAxisRaw = Point3D(
            green.x - red.x,
            green.y - red.y,
            green.z - red.z
        )
        val xLen = sqrt(xAxisRaw.x.pow(2) + xAxisRaw.y.pow(2) + xAxisRaw.z.pow(2))
        val xAxis = Point3D(xAxisRaw.x / xLen, xAxisRaw.y / xLen, xAxisRaw.z / xLen)

        // 临时向量：中心到蓝点
        val toBl = Point3D(
            blue.x - center.x,
            blue.y - center.y,
            blue.z - center.z
        )

        // Z轴：通过叉积计算法线（X × toBl�?
        val zAxisRaw = Point3D(
            xAxis.y * toBl.z - xAxis.z * toBl.y,
            xAxis.z * toBl.x - xAxis.x * toBl.z,
            xAxis.x * toBl.y - xAxis.y * toBl.x
        )
        val zLen = sqrt(zAxisRaw.x.pow(2) + zAxisRaw.y.pow(2) + zAxisRaw.z.pow(2))
        val zAxis = Point3D(zAxisRaw.x / zLen, zAxisRaw.y / zLen, zAxisRaw.z / zLen)

        // 从旋转矩阵计算欧拉角 (ZYX顺序)
        val pitch = Math.toDegrees(asin(-zAxis.x.toDouble())).toFloat()
        val yaw = Math.toDegrees(atan2(zAxis.y.toDouble(), zAxis.z.toDouble())).toFloat()

        // Roll计算需要Y�?
        val yAxisRaw = Point3D(
            zAxis.y * xAxis.z - zAxis.z * xAxis.y,
            zAxis.z * xAxis.x - zAxis.x * xAxis.z,
            zAxis.x * xAxis.y - zAxis.y * xAxis.x
        )
        val roll = Math.toDegrees(atan2(yAxisRaw.y.toDouble(), yAxisRaw.z.toDouble())).toFloat()

        return Pose6DOF(
            position = center,
            roll = roll,
            pitch = pitch,
            yaw = yaw
        )
    }
}



