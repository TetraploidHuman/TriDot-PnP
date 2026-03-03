package org.example.tridotpnp

import android.Manifest
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowLeft
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.DividerDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.accompanist.permissions.shouldShowRationale

@Composable
@Preview
fun App() {
    MaterialTheme {
        BrightSpotDetectionApp()
    }
}

@androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun BrightSpotDetectionApp() {
    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)
    var detectedSpotsCount by remember { mutableIntStateOf(0) }
    var detectedSpots by remember { mutableStateOf<List<BrightSpot>>(emptyList()) }
    var maxSpots by remember { mutableIntStateOf(1) } // 默认显示最亮的1个点
    var exposureCompensation by remember { mutableIntStateOf(0) } // 曝光补偿，0为默认值
    var gridSize by remember { mutableIntStateOf(50) } // 检测网格大小，默认50x50
    var knownDistance by remember { mutableFloatStateOf(100f) } // 已知的两点距离（毫米）
    var pnpResult by remember { mutableStateOf<PnPDistanceCalculator.PnPResult?>(null) }
    var imageSize by remember { mutableStateOf<Pair<Int, Int>?>(null) }
    var fps by remember { mutableFloatStateOf(0f) } // 识别帧率
    var enableRoiOptimization by remember { mutableStateOf(true) } // ROI优化开关，默认启用
    var isControlPanelExpanded by remember { mutableStateOf(true) } // 控制面板展开状态，默认展开
    var showSettings by remember { mutableStateOf(false) } // 设置页面显示状态

    // 颜色校准相关
    var isCalibrating by remember { mutableStateOf(false) }
    var calibrationData by remember { mutableStateOf<ColorCalibration?>(null) }
    var calibrationBitmap by remember { mutableStateOf<android.graphics.Bitmap?>(null) }

    val context = androidx.compose.ui.platform.LocalContext.current
    val pnpCalculator = remember { PnPDistanceCalculator() }
    val detector = remember { BrightSpotDetector() }

    Box(modifier = Modifier.fillMaxSize()) {
        when {
            cameraPermissionState.status.isGranted -> {
                // 相机预览
                CameraPreview(
                    modifier = Modifier.fillMaxSize(),
                    maxSpots = maxSpots,
                    exposureCompensation = exposureCompensation,
                    gridSize = gridSize,
                    detector = detector,  // 传递detector实例
                    enableRoiOptimization = enableRoiOptimization,
                    onFpsUpdate = { newFps ->
                        fps = newFps
                    },
                    onBrightSpotsDetected = { spots, size ->
                        detectedSpotsCount = spots.size
                        detectedSpots = spots
                        imageSize = size

                        // 根据选择的点数执行不同的PnP计算
                        when (maxSpots) {// 3点模式：RGB三色LED，计算完整6DOF姿态
                            3 if spots.size == 3 && size != null -> {
                                // 识别红、绿、蓝三个点
                                val redSpot = spots.find { it.color == LedColor.RED }
                                val greenSpot = spots.find { it.color == LedColor.GREEN }
                                val blueSpot = spots.find { it.color == LedColor.BLUE }

                                pnpResult =
                                    if (redSpot != null && greenSpot != null && blueSpot != null) {
                                        pnpCalculator.calculate3PointPnP(
                                            redPoint = redSpot.position,
                                            greenPoint = greenSpot.position,
                                            bluePoint = blueSpot.position,
                                            triangleEdgeLength = knownDistance,
                                            focalLength = null,
                                            imageWidth = size.first,
                                            imageHeight = size.second
                                        )
                                    } else {
                                        null  // 未检测到完整的RGB三色
                                    }
                            }

                            // 2点模式：相同颜色LED，计算距离和方向
                            2 if spots.size == 2 && size != null -> {
                                pnpResult = pnpCalculator.calculate2PointPnP(
                                    point1 = spots[0].position,
                                    point2 = spots[1].position,
                                    realDistance = knownDistance,
                                    focalLength = null,
                                    imageWidth = size.first,
                                    imageHeight = size.second
                                )
                            }

                            else -> {
                                pnpResult = null
                            }
                        }
                    },
                    onBitmapCaptured = { bitmap ->
                        // 释放旧的bitmap
                        calibrationBitmap?.recycle()
                        calibrationBitmap = bitmap
                    }
                )

                // 校准标记（校准模式下显示）
                if (isCalibrating && maxSpots == 3) {
                    // Canvas层：不拦截触摸事件，让事件穿透到CameraPreview进行ROI选择
                    Canvas(
                        modifier = Modifier.fillMaxSize()
                    ) {
                        val centerX = size.width / 2f
                        val centerY = size.height / 2f
                        // 使用屏幕最小尺寸计算，确保在横屏和竖屏下都有合适的大小
                        val calibrationRadius = kotlin.math.min(size.width, size.height) * 0.20f

                        // 绘制校准圆框（彩虹渐变效果）
                        // 外圈 - 半透明白色
                        drawCircle(
                            color = Color(0xAAFFFFFF),
                            radius = calibrationRadius + 10f,
                            center = Offset(centerX, centerY),
                            style = Stroke(width = 4f)
                        )

                        // 中圈 - 彩色渐变提示
                        drawCircle(
                            color = Color(0xCCFFFFFF),
                            radius = calibrationRadius,
                            center = Offset(centerX, centerY),
                            style = Stroke(width = 8f)
                        )

                        // 内圈 - 半透明
                        drawCircle(
                            color = Color(0x33FFFFFF),
                            radius = calibrationRadius - 10f,
                            center = Offset(centerX, centerY)
                        )

                        // 中心十字线
                        drawLine(
                            color = Color(0xAAFFFFFF),
                            start = Offset(centerX - 40f, centerY),
                            end = Offset(centerX + 40f, centerY),
                            strokeWidth = 3f
                        )
                        drawLine(
                            color = Color(0xAAFFFFFF),
                            start = Offset(centerX, centerY - 40f),
                            end = Offset(centerX, centerY + 40f),
                            strokeWidth = 3f
                        )

                        // 绘制RGB标识（圆框边缘）
                        drawContext.canvas.nativeCanvas.apply {
                            val paint = android.graphics.Paint().apply {
                                textSize = 36f
                                textAlign = android.graphics.Paint.Align.CENTER
                                isFakeBoldText = true
                                setShadowLayer(8f, 0f, 0f, android.graphics.Color.BLACK)
                            }

                            // 红色标识（左）
                            paint.color = android.graphics.Color.rgb(255, 51, 51)
                            drawText("R", centerX - calibrationRadius - 50f, centerY + 12f, paint)

                            // 绿色标识（上）
                            paint.color = android.graphics.Color.rgb(51, 255, 51)
                            drawText("G", centerX, centerY - calibrationRadius - 40f, paint)

                            // 蓝色标识（右）
                            paint.color = android.graphics.Color.rgb(51, 102, 255)
                            drawText("B", centerX + calibrationRadius + 50f, centerY + 12f, paint)
                        }
                    }
                }

                // 右侧控制面板（横屏适配）- 支持展开/折叠
                val panelWidth by animateDpAsState(
                    targetValue = if (isControlPanelExpanded) 320.dp else 0.dp,
                    animationSpec = tween(durationMillis = 300),
                    label = "panelWidth"
                )
                Surface(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .width(panelWidth)
                        .fillMaxHeight(),
                    color = Color(0xE6FFFFFF)
                ) {
                    if (isControlPanelExpanded) {
                        Box(
                            modifier = Modifier.fillMaxSize()
                        ) {
                            val scrollState = rememberScrollState()
                            // 控制面板部分（可滚动）
                            Column(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .verticalScroll(scrollState)
                                    .padding(12.dp),
                                verticalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                // 控制面板内容
                                Column(
                                    verticalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    // 亮点数量
                                    Text(
                                        text = "TARGETS",
                                        color = Color(0xFF888888),
                                        style = MaterialTheme.typography.labelSmall
                                    )
                                    Row(
                                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                                    ) {
                                        for (count in 1..3) {
                                            Surface(
                                                modifier = Modifier
                                                    .weight(1f)
                                                    .height(52.dp),
                                                shape = MaterialTheme.shapes.small,
                                                color = if (maxSpots == count) Color(0xFF0078D4) else Color(
                                                    0xFFF0F0F0
                                                ),
                                                shadowElevation = if (maxSpots == count) 4.dp else 0.dp
                                            ) {
                                                TextButton(
                                                    onClick = { maxSpots = count },
                                                    modifier = Modifier.fillMaxSize(),
                                                    contentPadding = PaddingValues(0.dp)
                                                ) {
                                                    Text(
                                                        text = "$count",
                                                        color = if (maxSpots == count) Color.White else Color(
                                                            0xFF333333
                                                        ),
                                                        style = MaterialTheme.typography.headlineMedium
                                                    )
                                                }
                                            }
                                        }
                                    }

                                    // 精度控制
                                    Text(
                                        text = "PRECISION",
                                        color = Color(0xFF888888),
                                        style = MaterialTheme.typography.labelSmall,
                                        modifier = Modifier.padding(top = 4.dp)
                                    )
                                    Column(
                                        verticalArrangement = Arrangement.spacedBy(4.dp)
                                    ) {
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Text(
                                                text = "${gridSize}×${gridSize}",
                                                color = Color(0xFF0078D4),
                                                style = MaterialTheme.typography.titleMedium
                                            )
                                            Text(
                                                text = "${gridSize * gridSize}",
                                                color = Color(0xFF888888),
                                                style = MaterialTheme.typography.bodySmall
                                            )
                                        }
                                        Slider(
                                            value = gridSize.toFloat(),
                                            onValueChange = { value ->
                                                val rounded = kotlin.math.round(value / 8f).toInt() * 8
                                                gridSize = rounded.coerceIn(32, 256)
                                            },
                                            valueRange = 32f..256f,
                                            steps = ((256 - 32) / 8 - 1),
                                            modifier = Modifier.fillMaxWidth()
                                        )
                                    }

                                    // 曝光控制（横向）
                                    Text(
                                        text = "EXPOSURE",
                                        color = Color(0xFF888888),
                                        style = MaterialTheme.typography.labelSmall,
                                        modifier = Modifier.padding(top = 4.dp)
                                    )
                                    Surface(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .height(80.dp),
                                        shape = MaterialTheme.shapes.small,
                                        color = Color(0xFFF0F0F0)
                                    ) {
                                        Row(
                                            modifier = Modifier.fillMaxSize(),
                                            horizontalArrangement = Arrangement.SpaceEvenly,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            IconButton(
                                                onClick = { if (exposureCompensation > -6) exposureCompensation-- }
                                            ) {
                                                Text(
                                                    "−",
                                                    style = MaterialTheme.typography.headlineMedium,
                                                    color = Color(0xFF333333)
                                                )
                                            }
                                            Text(
                                                text = when {
                                                    exposureCompensation > 0 -> "+$exposureCompensation"
                                                    else -> "$exposureCompensation"
                                                },
                                                color = Color(0xFF0078D4),
                                                style = MaterialTheme.typography.headlineLarge
                                            )
                                            IconButton(
                                                onClick = { if (exposureCompensation < 6) exposureCompensation++ }
                                            ) {
                                                Text(
                                                    "+",
                                                    style = MaterialTheme.typography.headlineMedium,
                                                    color = Color(0xFF333333)
                                                )
                                            }
                                        }
                                    }

                                    // ROI优化开关（仅在3点模式显示）
                                    if (maxSpots == 3) {
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Text(
                                                text = "ROI OPTIMIZATION",
                                                color = Color(0xFF888888),
                                                style = MaterialTheme.typography.labelSmall
                                            )
                                            Surface(
                                                modifier = Modifier
                                                    .height(36.dp)
                                                    .width(64.dp),
                                                shape = MaterialTheme.shapes.small,
                                                color = if (enableRoiOptimization) Color(0xFF4CAF50) else Color(
                                                    0xFFCCCCCC
                                                ),
                                                shadowElevation = if (enableRoiOptimization) 4.dp else 0.dp
                                            ) {
                                                TextButton(
                                                    onClick = {
                                                        enableRoiOptimization = !enableRoiOptimization
                                                    },
                                                    modifier = Modifier.fillMaxSize(),
                                                    contentPadding = PaddingValues(0.dp)
                                                ) {
                                                    Text(
                                                        text = if (enableRoiOptimization) "ON" else "OFF",
                                                        color = if (enableRoiOptimization) Color.White else Color(
                                                            0xFF666666
                                                        ),
                                                        style = MaterialTheme.typography.labelMedium
                                                    )
                                                }
                                            }
                                        }

                                        // 校准按钮（仅在3点模式显示）
                                        HorizontalDivider(
                                            modifier = Modifier.padding(vertical = 4.dp),
                                            thickness = DividerDefaults.Thickness,
                                            color = Color(0xFFE0E0E0)
                                        )
                                        Text(
                                            text = "CALIBRATION",
                                            color = Color(0xFF888888),
                                            style = MaterialTheme.typography.labelSmall
                                        )
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            // 清除校准按钮（仅在已校准状态显示）
                                            if (calibrationData != null && !isCalibrating) {
                                                Surface(
                                                    modifier = Modifier
                                                        .weight(1f)
                                                        .height(40.dp),
                                                    shape = MaterialTheme.shapes.small,
                                                    color = Color(0xFFFF6666),
                                                    shadowElevation = 4.dp
                                                ) {
                                                    TextButton(
                                                        onClick = {
                                                            calibrationData = null
                                                            detector.setCalibration(null)
                                                            android.util.Log.d(
                                                                "Calibration",
                                                                "已清除校准数据"
                                                            )
                                                            android.widget.Toast.makeText(
                                                                context,
                                                                "已清除校准",
                                                                android.widget.Toast.LENGTH_SHORT
                                                            ).show()
                                                        },
                                                        modifier = Modifier.fillMaxSize(),
                                                        contentPadding = PaddingValues(0.dp)
                                                    ) {
                                                        Text(
                                                            text = "✗ 清除",
                                                            color = Color.White,
                                                            style = MaterialTheme.typography.labelMedium
                                                        )
                                                    }
                                                }
                                            }

                                            // 校准/确认按钮
                                            Surface(
                                                modifier = Modifier
                                                    .weight(1f)
                                                    .height(40.dp),
                                                shape = MaterialTheme.shapes.small,
                                                color = if (isCalibrating) Color(0xFFFFAA00) else Color(
                                                    0xFF0078D4
                                                ),
                                                shadowElevation = 4.dp
                                            ) {
                                                TextButton(
                                                    onClick = {
                                                        if (isCalibrating) {
                                                            // 执行校准
                                                            try {
                                                                calibrationBitmap?.let { originalBitmap ->
                                                                    if (!originalBitmap.isRecycled) {
                                                                        // 立即创建一个不可变副本，防止在校准过程中被回收
                                                                        val bitmap =
                                                                            originalBitmap.copy(
                                                                                originalBitmap.config
                                                                                    ?: android.graphics.Bitmap.Config.ARGB_8888,
                                                                                false
                                                                            )

                                                                        val width = bitmap.width
                                                                        val height = bitmap.height

                                                                        // 定义校准圆框的范围（图像坐标）
                                                                        val centerX = width / 2
                                                                        val centerY = height / 2
                                                                        val calibrationRadius =
                                                                            (width * 0.25f).toInt()  // 圆框半径为图像宽度的25%

                                                                        android.util.Log.d(
                                                                            "Calibration",
                                                                            "=== 开始自动校准 ==="
                                                                        )
                                                                        android.util.Log.d(
                                                                            "Calibration",
                                                                            "图像尺寸: ${width}x${height}"
                                                                        )
                                                                        android.util.Log.d(
                                                                            "Calibration",
                                                                            "搜索区域: 中心($centerX,$centerY), 半径=$calibrationRadius"
                                                                        )

                                                                        // 在圆框内搜索最红、最绿、最蓝的点
                                                                        val result =
                                                                            detector.findRGBColorsInCircle(
                                                                                bitmap,
                                                                                centerX,
                                                                                centerY,
                                                                                calibrationRadius
                                                                            )

                                                                        // 使用完后立即回收副本
                                                                        bitmap.recycle()

                                                                        if (result != null) {
                                                                            val (redColor, greenColor, blueColor) = result

                                                                            // 显示找到的颜色详情
                                                                            fun analyzeColor(
                                                                                name: String,
                                                                                color: Triple<Float, Float, Float>
                                                                            ): String {
                                                                                val (r, g, b) = color
                                                                                val total = r + g + b
                                                                                val rPct =
                                                                                    (r / total * 100).toInt()
                                                                                val gPct =
                                                                                    (g / total * 100).toInt()
                                                                                val bPct =
                                                                                    (b / total * 100).toInt()
                                                                                return "$name: RGB(${r.toInt()},${g.toInt()},${b.toInt()}) 占比:R${rPct}% G${gPct}% B${bPct}% 亮度:${total.toInt()}"
                                                                            }

                                                                            android.util.Log.d(
                                                                                "Calibration",
                                                                                "=== 校准结果 ==="
                                                                            )
                                                                            android.util.Log.d(
                                                                                "Calibration",
                                                                                analyzeColor(
                                                                                    "🔴 红色LED",
                                                                                    redColor
                                                                                )
                                                                            )
                                                                            android.util.Log.d(
                                                                                "Calibration",
                                                                                analyzeColor(
                                                                                    "🟢 绿色LED",
                                                                                    greenColor
                                                                                )
                                                                            )
                                                                            android.util.Log.d(
                                                                                "Calibration",
                                                                                analyzeColor(
                                                                                    "🔵 蓝色LED",
                                                                                    blueColor
                                                                                )
                                                                            )

                                                                            // 保存校准数据
                                                                            calibrationData =
                                                                                ColorCalibration(
                                                                                    redColor,
                                                                                    greenColor,
                                                                                    blueColor
                                                                                )
                                                                            detector.setCalibration(
                                                                                calibrationData
                                                                            )
                                                                            android.util.Log.d(
                                                                                "Calibration",
                                                                                "✓ 校准数据已保存，现在将使用这些颜色进行检测"
                                                                            )

                                                                            // 显示成功提示
                                                                            val totalBrightness =
                                                                                (redColor.first + redColor.second + redColor.third +
                                                                                        greenColor.first + greenColor.second + greenColor.third +
                                                                                        blueColor.first + blueColor.second + blueColor.third) / 3
                                                                            android.widget.Toast.makeText(
                                                                                context,
                                                                                "✓ 校准成功！平均亮度: ${totalBrightness.toInt()}",
                                                                                android.widget.Toast.LENGTH_SHORT
                                                                            ).show()
                                                                        } else {
                                                                            // 搜索失败
                                                                            android.util.Log.e(
                                                                                "Calibration",
                                                                                "✗ 未在圆框内找到有效的RGB颜色"
                                                                            )

                                                                            android.widget.Toast.makeText(
                                                                                context,
                                                                                "校准失败！\n请确保RGB三色LED都在圆框内",
                                                                                android.widget.Toast.LENGTH_LONG
                                                                            ).show()

                                                                            // 清除之前的校准数据
                                                                            calibrationData = null
                                                                            detector.setCalibration(null)
                                                                        }
                                                                    }
                                                                }
                                                            } catch (e: Exception) {
                                                                e.printStackTrace()
                                                                android.widget.Toast.makeText(
                                                                    context,
                                                                    "校准出错：${e.message}",
                                                                    android.widget.Toast.LENGTH_LONG
                                                                ).show()
                                                            }
                                                        }
                                                        isCalibrating = !isCalibrating
                                                    },
                                                    modifier = Modifier.fillMaxSize(),
                                                    contentPadding = PaddingValues(0.dp)
                                                ) {
                                                    Text(
                                                        text = if (isCalibrating) "✓ 确认" else if (calibrationData != null) "✓ 已校准" else "校准",
                                                        color = Color.White,
                                                        style = MaterialTheme.typography.labelMedium
                                                    )
                                                }
                                            }
                                        }
                                    }

                                    // PnP距离设置（在2点或3点模式显示）
                                    if (maxSpots == 2 || maxSpots == 3) {
                                        HorizontalDivider(
                                            modifier = Modifier.padding(vertical = 4.dp),
                                            thickness = DividerDefaults.Thickness,
                                            color = Color(0xFFE0E0E0)
                                        )
                                        Text(
                                            text = if (maxSpots == 3) "△ EDGE LENGTH" else "PnP DISTANCE",
                                            color = Color(0xFF888888),
                                            style = MaterialTheme.typography.labelSmall
                                        )
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            IconButton(
                                                onClick = { if (knownDistance > 10) knownDistance -= 10 },
                                                modifier = Modifier.size(40.dp)
                                            ) {
                                                Text(
                                                    "−",
                                                    style = MaterialTheme.typography.headlineMedium,
                                                    color = Color(0xFF333333)
                                                )
                                            }
                                            Column(
                                                horizontalAlignment = Alignment.CenterHorizontally,
                                                modifier = Modifier.weight(1f)
                                            ) {
                                                Text(
                                                    text = "%.0f".format(knownDistance),
                                                    color = Color(0xFF0078D4),
                                                    style = MaterialTheme.typography.headlineMedium
                                                )
                                                Text(
                                                    text = "mm",
                                                    color = Color(0xFF999999),
                                                    style = MaterialTheme.typography.labelSmall
                                                )
                                            }
                                            IconButton(
                                                onClick = { knownDistance += 10 },
                                                modifier = Modifier.size(40.dp)
                                            ) {
                                                Text(
                                                    "+",
                                                    style = MaterialTheme.typography.headlineMedium,
                                                    color = Color(0xFF333333)
                                                )
                                            }
                                        }
                                    }

                                    HorizontalDivider(
                                        modifier = Modifier.padding(vertical = 8.dp),
                                        thickness = DividerDefaults.Thickness, color = Color(0xFFE0E0E0)
                                    )

                                    // 状态信息部分
                                    Column(
                                        verticalArrangement = Arrangement.spacedBy(12.dp)
                                    ) {
                                        // 状态信息标题
                                        Text(
                                            text = "STATUS",
                                            color = Color(0xFF888888),
                                            style = MaterialTheme.typography.labelSmall
                                        )

                                        // 检测状态
                                        Row(
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                                        ) {
                                            Surface(
                                                modifier = Modifier.size(12.dp),
                                                shape = MaterialTheme.shapes.small,
                                                color = if (detectedSpotsCount > 0) Color(0xFF4CAF50) else Color(
                                                    0xFFCCCCCC
                                                ),
                                                shadowElevation = if (detectedSpotsCount > 0) 4.dp else 0.dp
                                            ) {}
                                            Column(modifier = Modifier.weight(1f)) {
                                                Text(
                                                    text = "DETECTED",
                                                    color = Color(0xFF999999),
                                                    style = MaterialTheme.typography.labelSmall
                                                )
                                                Text(
                                                    text = "$detectedSpotsCount / $maxSpots",
                                                    color = if (detectedSpotsCount == maxSpots) Color(
                                                        0xFF4CAF50
                                                    ) else Color(0xFFFF9800),
                                                    style = MaterialTheme.typography.titleLarge
                                                )
                                            }
                                        }

                                        // 网格和区块信息（横向排列）
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween
                                        ) {
                                            Column {
                                                Text(
                                                    text = "GRID",
                                                    color = Color(0xFF999999),
                                                    style = MaterialTheme.typography.labelSmall
                                                )
                                                Text(
                                                    text = "${gridSize}×${gridSize}",
                                                    color = Color(0xFF333333),
                                                    style = MaterialTheme.typography.titleMedium
                                                )
                                            }

                                            Column(horizontalAlignment = Alignment.End) {
                                                Text(
                                                    text = "BLOCKS",
                                                    color = Color(0xFF999999),
                                                    style = MaterialTheme.typography.labelSmall
                                                )
                                                Text(
                                                    text = "${gridSize * gridSize}",
                                                    color = Color(0xFF333333),
                                                    style = MaterialTheme.typography.titleMedium
                                                )
                                            }
                                        }

                                        // 帧率显示
                                        Column {
                                            Text(
                                                text = "FPS",
                                                color = Color(0xFF999999),
                                                style = MaterialTheme.typography.labelSmall
                                            )
                                            Text(
                                                text = "%.1f".format(fps),
                                                color = if (fps > 20f) Color(0xFF4CAF50) else if (fps > 10f) Color(
                                                    0xFFFF9800
                                                ) else Color(0xFFE74C3C),
                                                style = MaterialTheme.typography.titleLarge
                                            )
                                        }

                                        // PnP结果显示（在2点或3点模式且有结果时显示）
                                        if ((maxSpots == 2 || maxSpots == 3) && pnpResult != null) {
                                            HorizontalDivider(
                                                modifier = Modifier.padding(vertical = 12.dp),
                                                thickness = DividerDefaults.Thickness,
                                                color = Color(0xFFE0E0E0)
                                            )

                                            // 距离、方位角、仰角（横向排列）
                                            Row(
                                                modifier = Modifier.fillMaxWidth(),
                                                horizontalArrangement = Arrangement.SpaceBetween
                                            ) {
                                                Column {
                                                    Text(
                                                        text = "DISTANCE",
                                                        color = Color(0xFF999999),
                                                        style = MaterialTheme.typography.labelSmall
                                                    )
                                                    Text(
                                                        text = pnpCalculator.formatDistance(pnpResult!!.distance),
                                                        color = Color(0xFF0078D4),
                                                        style = MaterialTheme.typography.titleMedium
                                                    )
                                                }

                                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                                    Text(
                                                        text = "AZIMUTH",
                                                        color = Color(0xFF999999),
                                                        style = MaterialTheme.typography.labelSmall
                                                    )
                                                    Text(
                                                        text = pnpCalculator.formatAngle(pnpResult!!.azimuth),
                                                        color = Color(0xFF333333),
                                                        style = MaterialTheme.typography.titleMedium
                                                    )
                                                }

                                                Column(horizontalAlignment = Alignment.End) {
                                                    Text(
                                                        text = "ELEVATION",
                                                        color = Color(0xFF999999),
                                                        style = MaterialTheme.typography.labelSmall
                                                    )
                                                    Text(
                                                        text = pnpCalculator.formatAngle(pnpResult!!.elevation),
                                                        color = Color(0xFF333333),
                                                        style = MaterialTheme.typography.titleMedium
                                                    )
                                                }
                                            }

                                            HorizontalDivider(
                                                modifier = Modifier.padding(vertical = 8.dp),
                                                thickness = DividerDefaults.Thickness,
                                                color = Color(0xFFE0E0E0)
                                            )

                                            if (maxSpots == 3 && pnpResult!!.pose6DOF != null) {
                                                // 3点模式：显示完整6DOF姿态
                                                Column(
                                                    verticalArrangement = Arrangement.spacedBy(8.dp)
                                                ) {
                                                    Text(
                                                        text = "6DOF POSE",
                                                        color = Color(0xFF0078D4),
                                                        style = MaterialTheme.typography.titleSmall
                                                    )

                                                    Column(
                                                        verticalArrangement = Arrangement.spacedBy(4.dp)
                                                    ) {
                                                        Row(
                                                            modifier = Modifier.fillMaxWidth(),
                                                            horizontalArrangement = Arrangement.SpaceBetween
                                                        ) {
                                                            Text(
                                                                text = "CENTER",
                                                                color = Color(0xFF95A5A6),
                                                                style = MaterialTheme.typography.labelSmall
                                                            )
                                                            Text(
                                                                text = pnpCalculator.formatPoint3D(
                                                                    pnpResult!!.pose6DOF!!.position
                                                                ),
                                                                color = Color(0xFF0078D4),
                                                                style = MaterialTheme.typography.bodySmall
                                                            )
                                                        }
                                                    }

                                                    HorizontalDivider(
                                                        modifier = Modifier.padding(vertical = 4.dp),
                                                        thickness = DividerDefaults.Thickness,
                                                        color = Color(0xFFE0E0E0)
                                                    )

                                                    Text(
                                                        text = "ORIENTATION",
                                                        color = Color(0xFF888888),
                                                        style = MaterialTheme.typography.labelSmall
                                                    )

                                                    Row(
                                                        modifier = Modifier.fillMaxWidth(),
                                                        horizontalArrangement = Arrangement.SpaceBetween
                                                    ) {
                                                        Column {
                                                            Text(
                                                                text = "ROLL",
                                                                color = Color(0xFF999999),
                                                                style = MaterialTheme.typography.labelSmall
                                                            )
                                                            Text(
                                                                text = pnpCalculator.formatAngle(
                                                                    pnpResult!!.pose6DOF!!.roll
                                                                ),
                                                                color = Color(0xFFE74C3C),
                                                                style = MaterialTheme.typography.bodyMedium
                                                            )
                                                        }

                                                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                                            Text(
                                                                text = "PITCH",
                                                                color = Color(0xFF999999),
                                                                style = MaterialTheme.typography.labelSmall
                                                            )
                                                            Text(
                                                                text = pnpCalculator.formatAngle(
                                                                    pnpResult!!.pose6DOF!!.pitch
                                                                ),
                                                                color = Color(0xFF27AE60),
                                                                style = MaterialTheme.typography.bodyMedium
                                                            )
                                                        }

                                                        Column(horizontalAlignment = Alignment.End) {
                                                            Text(
                                                                text = "YAW",
                                                                color = Color(0xFF999999),
                                                                style = MaterialTheme.typography.labelSmall
                                                            )
                                                            Text(
                                                                text = pnpCalculator.formatAngle(
                                                                    pnpResult!!.pose6DOF!!.yaw
                                                                ),
                                                                color = Color(0xFF3498DB),
                                                                style = MaterialTheme.typography.bodyMedium
                                                            )
                                                        }
                                                    }
                                                }
                                            } else {
                                                // 2点模式：显示3D坐标
                                                Column(
                                                    verticalArrangement = Arrangement.spacedBy(6.dp)
                                                ) {
                                                    Text(
                                                        text = "3D COORDINATES",
                                                        color = Color(0xFF888888),
                                                        style = MaterialTheme.typography.labelSmall
                                                    )

                                                    Row(
                                                        modifier = Modifier.fillMaxWidth(),
                                                        horizontalArrangement = Arrangement.SpaceBetween
                                                    ) {
                                                        Column(modifier = Modifier.weight(1f)) {
                                                            Text(
                                                                text = "POINT 1",
                                                                color = Color(0xFF999999),
                                                                style = MaterialTheme.typography.labelSmall
                                                            )
                                                            Text(
                                                                text = pnpCalculator.formatPoint3D(
                                                                    pnpResult!!.point1Coords
                                                                ),
                                                                color = Color(0xFF333333),
                                                                style = MaterialTheme.typography.bodySmall
                                                            )
                                                        }

                                                        Column(
                                                            modifier = Modifier.weight(1f),
                                                            horizontalAlignment = Alignment.End
                                                        ) {
                                                            Text(
                                                                text = "POINT 2",
                                                                color = Color(0xFF999999),
                                                                style = MaterialTheme.typography.labelSmall
                                                            )
                                                            Text(
                                                                text = pnpCalculator.formatPoint3D(
                                                                    pnpResult!!.point2Coords
                                                                ),
                                                                color = Color(0xFF333333),
                                                                style = MaterialTheme.typography.bodySmall,
                                                                textAlign = TextAlign.End
                                                            )
                                                        }
                                                    }

                                                    Row(
                                                        modifier = Modifier.fillMaxWidth(),
                                                        horizontalArrangement = Arrangement.Center
                                                    ) {
                                                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                                            Text(
                                                                text = "CENTER",
                                                                color = Color(0xFF999999),
                                                                style = MaterialTheme.typography.labelSmall
                                                            )
                                                            Text(
                                                                text = pnpCalculator.formatPoint3D(
                                                                    pnpResult!!.centerCoords
                                                                ),
                                                                color = Color(0xFF0078D4),
                                                                style = MaterialTheme.typography.bodyMedium
                                                            )
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // 右上角展开/折叠按钮
                val buttonEndPadding by animateDpAsState(
                    targetValue = if (isControlPanelExpanded) 332.dp else 12.dp,
                    animationSpec = tween(durationMillis = 300),
                    label = "buttonEndPadding"
                )

                IconButton(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(end = buttonEndPadding, top = 12.dp),
                    colors = IconButtonDefaults.iconButtonColors(
                        containerColor = Color(0x50000000)
                    ),
                    shape = CircleShape,
                    onClick = { isControlPanelExpanded = !isControlPanelExpanded }
                ) {
                    Icon(
                        imageVector = if (isControlPanelExpanded) Icons.AutoMirrored.Filled.KeyboardArrowRight else Icons.AutoMirrored.Filled.KeyboardArrowLeft,
                        contentDescription = if (isControlPanelExpanded) "折叠" else "展开",
                        tint = Color.White
                    )
                }

                // 左上角设置按钮
                IconButton(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(12.dp),
                    colors = IconButtonDefaults.iconButtonColors(
                        containerColor = Color(0x50000000)
                    ),
                    shape = CircleShape,
                    onClick = { showSettings = true }
                ) {
                    Icon(
                        imageVector = Icons.Filled.Settings,
                        contentDescription = "设置",
                        tint = Color.White
                    )
                }


                // 设置页面（全屏覆盖）
                if (showSettings) {
                    SettingsScreen(
                        onDismiss = { showSettings = false },
                        modifier = Modifier.zIndex(20f)
                    )
                }
            }
            cameraPermissionState.status.shouldShowRationale -> {
                // 需要向用户解释为什么需要相机权限
                PermissionRationaleContent(
                    onRequestPermission = { cameraPermissionState.launchPermissionRequest() }
                )
            }
            else -> {
                // 首次请求权限
                PermissionRequestContent(
                    onRequestPermission = { cameraPermissionState.launchPermissionRequest() }
                )
            }
        }
    }
}

@Composable
fun SettingsScreen(
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier.fillMaxSize(),
        color = Color(0xFF1E1E1E)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp)
        ) {
            // 顶部栏
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "设置",
                    style = MaterialTheme.typography.headlineLarge,
                    color = Color.White
                )
                IconButton(onClick = onDismiss) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "返回",
                        tint = Color.White
                    )
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            // 设置内容（占位）
            Column(
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text(
                    text = "设置选项",
                    style = MaterialTheme.typography.titleLarge,
                    color = Color.White,
                    modifier = Modifier.padding(bottom = 8.dp)
                )

                Text(
                    text = "设置页面内容待完善",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color(0xFFAAAAAA)
                )
            }
        }
    }
}

@Composable
fun PermissionRequestContent(onRequestPermission: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "需要相机权限",
            style = MaterialTheme.typography.headlineMedium,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 16.dp)
        )
        Text(
            text = "此应用需要访问您的相机来检测亮点。",
            style = MaterialTheme.typography.bodyLarge,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 32.dp)
        )
        Button(onClick = onRequestPermission) {
            Text("授予相机权限")
        }
    }
}

@Composable
fun PermissionRationaleContent(onRequestPermission: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "需要相机权限",
            style = MaterialTheme.typography.headlineMedium,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 16.dp)
        )
        Text(
            text = "为了实时检测画面中的亮点，我们需要访问您的相机。\n\n" +
                    "没有相机权限，应用将无法正常工作。",
            style = MaterialTheme.typography.bodyLarge,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 32.dp)
        )
        Button(onClick = onRequestPermission) {
            Text("授予权限")
        }
    }
}