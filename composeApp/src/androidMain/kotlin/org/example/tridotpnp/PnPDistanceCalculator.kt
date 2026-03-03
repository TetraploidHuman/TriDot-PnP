package org.example.tridotpnp

import androidx.compose.ui.geometry.Offset
import kotlin.math.*
import kotlin.math.abs

/**
 * PnP距离计算器
 * 用于计算摄像头到目标的距离和方向
 */
class PnPDistanceCalculator {

    data class Point3D(
        val x: Float,                 // X坐标（毫米）
        val y: Float,                 // Y坐标（毫米）
        val z: Float                  // Z坐标（毫米，深度）
    )

    data class Pose6DOF(
        val position: Point3D,        // 位置（中心点）
        val roll: Float,              // 横滚角（度）
        val pitch: Float,             // 俯仰角（度）
        val yaw: Float                // 偏航角（度）
    )

    data class PnPResult(
        val distance: Float,           // 到摄像头的平均距离（毫米）
        val azimuth: Float,           // 方位角（度）
        val elevation: Float,         // 仰角（度）
        val point1Distance: Float,    // 点1到摄像头的距离
        val point2Distance: Float,    // 点2到摄像头的距离
        val point1Coords: Point3D,    // 点1的3D坐标
        val point2Coords: Point3D,    // 点2的3D坐标
        val centerCoords: Point3D,    // 中心点的3D坐标
        val pose6DOF: Pose6DOF? = null  // 完整6DOF姿态（仅3点模式）
    )

    /**
     * 使用两点PnP算法计算距离和方向
     *
     * @param point1 第一个亮点的像素坐标
     * @param point2 第二个亮点的像素坐标
     * @param realDistance 两点之间的实际距离（毫米）
     * @param focalLength 相机焦距（像素单位），默认使用估算值
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
        // 计算相机焦距（如果未提供）
        val fx = focalLength ?: estimateFocalLength(imageWidth, imageHeight)
        val fy = fx // 假设焦距在x和y方向相同

        // 主点（图像中心）
        val cx = imageWidth / 2.0f
        val cy = imageHeight / 2.0f

        // 将像素坐标转换为归一化坐标
        val p1x = (point1.x - cx) / fx
        val p1y = (point1.y - cy) / fy
        val p2x = (point2.x - cx) / fx
        val p2y = (point2.y - cy) / fy

        // 计算图像平面上两点之间的距离
        val imageDistance = sqrt((p2x - p1x).pow(2) + (p2y - p1y).pow(2))

        // 使用相似三角形原理计算深度
        // realDistance / depth ≈ imageDistance / 1.0
        val averageDepth = if (imageDistance > 0.001f) {
            realDistance / imageDistance
        } else {
            realDistance * fx // 退回到简单估算
        }

        // 计算每个点的深度（假设在同一深度平面）
        val depth1 = averageDepth
        val depth2 = averageDepth

        // 计算3D坐标
        val x1 = p1x * depth1
        val y1 = p1y * depth1
        val z1 = depth1

        val x2 = p2x * depth2
        val y2 = p2y * depth2
        val z2 = depth2

        // 计算中点的3D坐标
        val centerX = (x1 + x2) / 2
        val centerY = (y1 + y2) / 2
        val centerZ = (z1 + z2) / 2

        // 计算方向向量
        val dirX = x2 - x1
        val dirY = y2 - y1
        val dirZ = z2 - z1

        // 计算方位角（azimuth）- 在XZ平面的角度
        val azimuth = Math.toDegrees(atan2(dirX.toDouble(), dirZ.toDouble())).toFloat()

        // 计算仰角（elevation）
        val horizontalDist = sqrt(dirX.pow(2) + dirZ.pow(2))
        val elevation = Math.toDegrees(atan2(dirY.toDouble(), horizontalDist.toDouble())).toFloat()

        // 计算到摄像头的距离
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
     * 基于常见的智能手机相机参数
     */
    private fun estimateFocalLength(imageWidth: Int, imageHeight: Int): Float {
        // 假设水平视场角约为60度
        val fovDegrees = 60.0
        val fovRadians = Math.toRadians(fovDegrees)

        // 焦距 = imageWidth / (2 * tan(fov/2))
        return (imageWidth / (2.0 * tan(fovRadians / 2.0))).toFloat()
    }

    /**
     * 格式化距离显示
     */
    fun formatDistance(distance: Float): String {
        return when {
            distance < 1000 -> "%.0f mm".format(distance)
            distance < 10000 -> "%.2f m".format(distance / 1000)
            else -> "%.1f m".format(distance / 1000)
        }
    }

    /**
     * 格式化角度显示
     */
    fun formatAngle(angle: Float): String {
        return "%.1f°".format(angle)
    }

    /**
     * 格式化3D坐标显示
     */
    fun formatCoordinate(value: Float): String {
        return when {
            abs(value) < 1000 -> "%.0f".format(value)
            else -> "%.1f".format(value / 1000)
        }
    }

    /**
     * 格式化3D坐标单位
     */
    fun getCoordinateUnit(value: Float): String {
        return if (abs(value) < 1000) "mm" else "m"
    }

    /**
     * 格式化完整坐标
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
     * 使用三点PnP算法计算完整6DOF姿态
     *
     * @param redPoint 红色LED的像素坐标
     * @param greenPoint 绿色LED的像素坐标
     * @param bluePoint 蓝色LED的像素坐标
     * @param triangleEdgeLength 正三角形的边长（毫米）
     * @param focalLength 相机焦距（像素单位），默认使用估算值
     * @param imageWidth 图像宽度（像素）
     * @param imageHeight 图像高度（像素）
     * @return PnP计算结果，包含完整6DOF姿态
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

        // 将像素坐标转换为归一化坐标
        val pr_x = (redPoint.x - cx) / fx
        val pr_y = (redPoint.y - cy) / fy
        val pg_x = (greenPoint.x - cx) / fx
        val pg_y = (greenPoint.y - cy) / fy
        val pb_x = (bluePoint.x - cx) / fx
        val pb_y = (bluePoint.y - cy) / fy

        // 计算图像平面上的归一化距离
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

        // 计算三个点的3D坐标（使用各自的深度）
        val redCoords = Point3D(pr_x * depthRed, pr_y * depthRed, depthRed)
        val greenCoords = Point3D(pg_x * depthGreen, pg_y * depthGreen, depthGreen)
        val blueCoords = Point3D(pb_x * depthBlue, pb_y * depthBlue, depthBlue)

        // 计算中心点
        val centerX = (redCoords.x + greenCoords.x + blueCoords.x) / 3f
        val centerY = (redCoords.y + greenCoords.y + blueCoords.y) / 3f
        val centerZ = (redCoords.z + greenCoords.z + blueCoords.z) / 3f
        val centerCoords = Point3D(centerX, centerY, centerZ)

        // 计算6DOF姿态
        val pose = calculate6DOFPose(redCoords, greenCoords, blueCoords)

        // 计算距离和方向
        val distance1 = sqrt(redCoords.x.pow(2) + redCoords.y.pow(2) + redCoords.z.pow(2))
        val distance2 = sqrt(greenCoords.x.pow(2) + greenCoords.y.pow(2) + greenCoords.z.pow(2))
        val averageDistance = (distance1 + distance2 + sqrt(blueCoords.x.pow(2) + blueCoords.y.pow(2) + blueCoords.z.pow(2))) / 3

        // 计算方位角和仰角（使用红绿连线的方向）
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
     * @param u1, v1 点1的归一化坐标
     * @param u2, v2 点2的归一化坐标
     * @param u3, v3 点3的归一化坐标
     * @param d12 点1和点2之间的实际距离
     * @param d23 点2和点3之间的实际距离
     * @param d31 点3和点1之间的实际距离
     * @return Triple(点1深度, 点2深度, 点3深度)
     */
    private fun solve3PointDepths(
        u1: Float, v1: Float,  // 点1的归一化坐标
        u2: Float, v2: Float,  // 点2的归一化坐标
        u3: Float, v3: Float,  // 点3的归一化坐标
        d12: Float,            // 点1和点2之间的实际距离
        d23: Float,            // 点2和点3之间的实际距离
        d31: Float             // 点3和点1之间的实际距离
    ): Triple<Float, Float, Float> {
        // 初始估算：使用平均图像距离估算初始深度
        val dist12_image = sqrt((u2 - u1).pow(2) + (v2 - v1).pow(2))
        val dist23_image = sqrt((u3 - u2).pow(2) + (v3 - v2).pow(2))
        val dist31_image = sqrt((u1 - u3).pow(2) + (v1 - v3).pow(2))

        // 使用几何平均估算初始深度
        var z1 = if (dist12_image > 0.001f && dist31_image > 0.001f) {
            sqrt((d12 / dist12_image) * (d31 / dist31_image))
        } else {
            1000f // 默认深度
        }

        var z2 = if (dist12_image > 0.001f && dist23_image > 0.001f) {
            sqrt((d12 / dist12_image) * (d23 / dist23_image))
        } else {
            1000f
        }

        var z3 = if (dist23_image > 0.001f && dist31_image > 0.001f) {
            sqrt((d23 / dist23_image) * (d31 / dist31_image))
        } else {
            1000f
        }

        // 使用迭代优化求解（Gauss-Newton方法）
        // 目标：最小化实际距离与计算距离的差异
        for (iteration in 0..50) {
            // 计算当前3D坐标
            val p1 = Point3D(u1 * z1, v1 * z1, z1)
            val p2 = Point3D(u2 * z2, v2 * z2, z2)
            val p3 = Point3D(u3 * z3, v3 * z3, z3)

            // 计算当前距离
            val dist12_curr = sqrt((p2.x - p1.x).pow(2) + (p2.y - p1.y).pow(2) + (p2.z - p1.z).pow(2))
            val dist23_curr = sqrt((p3.x - p2.x).pow(2) + (p3.y - p2.y).pow(2) + (p3.z - p2.z).pow(2))
            val dist31_curr = sqrt((p1.x - p3.x).pow(2) + (p1.y - p3.y).pow(2) + (p1.z - p3.z).pow(2))

            // 计算误差
            val error12 = dist12_curr - d12
            val error23 = dist23_curr - d23
            val error31 = dist31_curr - d31

            // 如果误差足够小，停止迭代
            val maxError = maxOf(abs(error12), abs(error23), abs(error31))
            if (maxError < 0.1f) {
                break
            }

            // 根据误差调整深度
            val learningRate = 0.5f

            // 根据误差调整z1
            if (dist12_curr > 0.001f) {
                val scale12 = d12 / dist12_curr
                z1 *= (1f + (scale12 - 1f) * learningRate)
            }
            if (dist31_curr > 0.001f) {
                val scale31 = d31 / dist31_curr
                z1 *= (1f + (scale31 - 1f) * learningRate)
            }

            // 根据误差调整z2
            if (dist12_curr > 0.001f) {
                val scale12 = d12 / dist12_curr
                z2 *= (1f + (scale12 - 1f) * learningRate)
            }
            if (dist23_curr > 0.001f) {
                val scale23 = d23 / dist23_curr
                z2 *= (1f + (scale23 - 1f) * learningRate)
            }

            // 根据误差调整z3
            if (dist23_curr > 0.001f) {
                val scale23 = d23 / dist23_curr
                z3 *= (1f + (scale23 - 1f) * learningRate)
            }
            if (dist31_curr > 0.001f) {
                val scale31 = d31 / dist31_curr
                z3 *= (1f + (scale31 - 1f) * learningRate)
            }

            // 确保深度为正且合理
            z1 = z1.coerceAtLeast(100f)
            z2 = z2.coerceAtLeast(100f)
            z3 = z3.coerceAtLeast(100f)
        }

        return Triple(z1, z2, z3)
    }

    /**
     * 从三个点计算6DOF姿态（欧拉角）
     */
    private fun calculate6DOFPose(
        red: Point3D,
        green: Point3D,
        blue: Point3D
    ): Pose6DOF {
        // 计算中心点
        val center = Point3D(
            (red.x + green.x + blue.x) / 3f,
            (red.y + green.y + blue.y) / 3f,
            (red.z + green.z + blue.z) / 3f
        )

        // 计算本地坐标系
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

        // Z轴：通过叉积计算法线（X × toBl）
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

        // Roll计算需要Y轴
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

