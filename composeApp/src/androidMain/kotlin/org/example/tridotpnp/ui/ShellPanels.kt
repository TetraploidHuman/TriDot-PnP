package org.example.tridotpnp.ui


import org.example.tridotpnp.config.AppTuningSettings
import org.example.tridotpnp.config.updateCamera
import org.example.tridotpnp.config.updateDetect
import org.example.tridotpnp.config.updateTrackLock
import org.example.tridotpnp.config.updatePnp
import org.example.tridotpnp.config.updateFusion
import org.example.tridotpnp.config.updateBle
import org.example.tridotpnp.tracking.PnPDistanceCalculator
import org.example.tridotpnp.tracking.TrackedNodeState
import org.example.tridotpnp.tracking.VisionObservation
import org.example.tridotpnp.vision.CalibrationActionButton
import org.example.tridotpnp.vision.ColorCalibration
import org.example.tridotpnp.vision.LedColor
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DividerDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.unit.dp

@Composable
internal fun DetectionControlSection(
    tuning: AppTuningSettings,
    onTuningChange: (AppTuningSettings) -> Unit,
    lastGridSizeHapticStep: Int,
    onGridHapticStepChange: (Int) -> Unit,
    gridSliderInteractionSource: MutableInteractionSource,
    targetSpotCount: Int,
    calibrationData: ColorCalibration?,
    isCalibrating: Boolean,
    onClearCalibration: () -> Unit,
    onCalibrationAction: () -> Unit
) {
    val haptic = LocalHapticFeedback.current

    Text(
        text = "网格精度",
        color = FlatUiColors.TextMuted,
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
                text = "${tuning.gridSize}×${tuning.gridSize}",
                color = FlatUiColors.Accent,
                style = MaterialTheme.typography.titleMedium
            )
            Text(
                text = "${tuning.gridSize * tuning.gridSize}",
                color = FlatUiColors.TextMuted,
                style = MaterialTheme.typography.bodySmall
            )
        }
        Slider(
            value = tuning.gridSize.toFloat(),
            onValueChange = { value ->
                val rounded = kotlin.math.round(value / 8f).toInt() * 8
                val newGridSize = rounded.coerceIn(32, 512)
                if (newGridSize != lastGridSizeHapticStep) {
                    haptic.performHapticFeedback(HapticFeedbackType.SegmentTick)
                    onGridHapticStepChange(newGridSize)
                }
                onTuningChange(tuning.updateCamera { copy(gridSize = newGridSize) })
            },
            onValueChangeFinished = {
                haptic.performHapticFeedback(HapticFeedbackType.GestureEnd)
            },
            valueRange = 32f..512f,
            steps = ((512 - 32) / 32 - 1),
            interactionSource = gridSliderInteractionSource,
            modifier = Modifier.fillMaxWidth()
        )
    }

    Text(
        text = "曝光补偿",
        color = FlatUiColors.TextMuted,
        style = MaterialTheme.typography.labelSmall,
        modifier = Modifier.padding(top = 4.dp)
    )
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .height(80.dp),
        shape = MaterialTheme.shapes.small,
        color = FlatUiColors.Panel
    ) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                    if (tuning.exposureCompensation > -6) {
                        onTuningChange(
                            tuning.updateCamera { copy(exposureCompensation = tuning.exposureCompensation - 1) }
                        )
                    }
                }
            ) {
                Text(
                    "-",
                    style = MaterialTheme.typography.headlineMedium,
                    color = FlatUiColors.TextPrimary
                )
            }
            Text(
                text = if (tuning.exposureCompensation > 0) "+${tuning.exposureCompensation}" else "${tuning.exposureCompensation}",
                color = FlatUiColors.Accent,
                style = MaterialTheme.typography.headlineLarge
            )
            IconButton(
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                    if (tuning.exposureCompensation < 6) {
                        onTuningChange(
                            tuning.updateCamera { copy(exposureCompensation = tuning.exposureCompensation + 1) }
                        )
                    }
                }
            ) {
                Text(
                    "+",
                    style = MaterialTheme.typography.headlineMedium,
                    color = FlatUiColors.TextPrimary
                )
            }
        }
    }

    if (targetSpotCount == 3) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "ROI 优化",
                color = FlatUiColors.TextMuted,
                style = MaterialTheme.typography.labelSmall
            )
            Surface(
                modifier = Modifier
                    .height(36.dp)
                    .width(64.dp),
                shape = MaterialTheme.shapes.small,
                color = if (tuning.enableRoiOptimization) Color(0xFF2CA36B) else Color(0xFFCCCCCC),
                shadowElevation = 0.dp
            ) {
                TextButton(
                    onClick = {
                        haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                        onTuningChange(tuning.updateCamera { copy(enableRoiOptimization = !tuning.enableRoiOptimization) })
                    },
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(0.dp)
                ) {
                    Text(
                        text = if (tuning.enableRoiOptimization) "开" else "关",
                        color = if (tuning.enableRoiOptimization) Color.White else Color(0xFF666666),
                        style = MaterialTheme.typography.labelMedium
                    )
                }
            }
        }

        HorizontalDivider(
            modifier = Modifier.padding(vertical = 4.dp),
            thickness = DividerDefaults.Thickness,
            color = FlatUiColors.Border
        )
        Text(
            text = "颜色校准",
            color = FlatUiColors.TextMuted,
            style = MaterialTheme.typography.labelSmall
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (calibrationData != null && !isCalibrating) {
                Surface(
                    modifier = Modifier
                        .weight(1f)
                        .height(40.dp),
                    shape = MaterialTheme.shapes.small,
                    color = Color(0xFFE35D5B),
                    shadowElevation = 0.dp
                ) {
                    TextButton(
                        onClick = {
                            haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                            onClearCalibration()
                        },
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(0.dp)
                    ) {
                        Text(
                            text = "清除",
                            color = Color.White,
                            style = MaterialTheme.typography.labelMedium
                        )
                    }
                }
            }

            CalibrationActionButton(
                isCalibrating = isCalibrating,
                hasCalibrationData = calibrationData != null,
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                    onCalibrationAction()
                },
                modifier = Modifier
                    .weight(1f)
                    .height(40.dp)
            )
        }
    }

    if (targetSpotCount == 3) {
        HorizontalDivider(
            modifier = Modifier.padding(vertical = 4.dp),
            thickness = DividerDefaults.Thickness,
            color = FlatUiColors.Border
        )
        Text(
            text = "三角形边长",
            color = FlatUiColors.TextMuted,
            style = MaterialTheme.typography.labelSmall
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                    if (tuning.knownTriangleEdgeLengthMm > 1f) {
                        onTuningChange(
                            tuning.updatePnp {
                                copy(
                                    knownTriangleEdgeLengthMm =
                                        (tuning.knownTriangleEdgeLengthMm - 1f).coerceAtLeast(1f)
                                )
                            }
                        )
                    }
                },
                modifier = Modifier.size(40.dp)
            ) {
                Text(
                    "-",
                    style = MaterialTheme.typography.headlineMedium,
                    color = FlatUiColors.TextPrimary
                )
            }
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = "%.0f".format(tuning.knownTriangleEdgeLengthMm),
                    color = FlatUiColors.Accent,
                    style = MaterialTheme.typography.headlineMedium
                )
                Text(
                    text = "mm",
                    color = FlatUiColors.TextMuted,
                    style = MaterialTheme.typography.labelSmall
                )
            }
            IconButton(
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
                    onTuningChange(
                        tuning.updatePnp { copy(knownTriangleEdgeLengthMm = tuning.knownTriangleEdgeLengthMm + 1f) }
                    )
                },
                modifier = Modifier.size(40.dp)
            ) {
                Text(
                    "+",
                    style = MaterialTheme.typography.headlineMedium,
                    color = FlatUiColors.TextPrimary
                )
            }
        }
    }

    HorizontalDivider(
        modifier = Modifier.padding(vertical = 8.dp),
        thickness = DividerDefaults.Thickness,
        color = FlatUiColors.Border
    )
}

@Composable
internal fun StatusSection(
    detectedSpotsCount: Int,
    targetSpotCount: Int,
    tuning: AppTuningSettings,
    fps: Float,
    pnpResult: PnPDistanceCalculator.PnPResult?,
    pnpCalculator: PnPDistanceCalculator,
    visionObservation: VisionObservation? = null,
    trackedNodes: List<TrackedNodeState> = emptyList(),
    bleConnectedCount: Int = 0,
    bleScanning: Boolean = false
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            text = "状态",
            color = FlatUiColors.TextMuted,
            style = MaterialTheme.typography.labelSmall
        )

        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Surface(
                modifier = Modifier.size(12.dp),
                shape = MaterialTheme.shapes.small,
                color = if (detectedSpotsCount > 0) Color(0xFF2CA36B) else Color(0xFFCCCCCC),
                shadowElevation = 0.dp
            ) {}
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "已检出",
                    color = FlatUiColors.TextMuted,
                    style = MaterialTheme.typography.labelSmall
                )
                Text(
                    text = "$detectedSpotsCount / $targetSpotCount",
                    color = if (detectedSpotsCount == targetSpotCount) Color(0xFF4CAF50) else Color(0xFFF0A229),
                    style = MaterialTheme.typography.titleLarge
                )
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column {
                Text(
                    text = "网格",
                    color = FlatUiColors.TextMuted,
                    style = MaterialTheme.typography.labelSmall
                )
                Text(
                    text = "${tuning.gridSize}×${tuning.gridSize}",
                    color = FlatUiColors.TextPrimary,
                    style = MaterialTheme.typography.titleMedium
                )
            }

            Column(horizontalAlignment = Alignment.End) {
                Text(
                    text = "格数",
                    color = FlatUiColors.TextMuted,
                    style = MaterialTheme.typography.labelSmall
                )
                Text(
                    text = "${tuning.gridSize * tuning.gridSize}",
                    color = FlatUiColors.TextPrimary,
                    style = MaterialTheme.typography.titleMedium
                )
            }
        }

        Column {
            Text(
                text = "FPS",
                color = FlatUiColors.TextMuted,
                style = MaterialTheme.typography.labelSmall
            )
            Text(
                text = "%.1f".format(fps),
                color = if (fps > 20f) Color(0xFF2CA36B) else if (fps > 10f) Color(0xFFFF9800) else Color(0xFFE35D5B),
                style = MaterialTheme.typography.titleLarge
            )
        }

        visionObservation?.let { obs ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text(
                        text = "置信度",
                        color = FlatUiColors.TextMuted,
                        style = MaterialTheme.typography.labelSmall
                    )
                    Text(
                        text = "%.0f%%".format(obs.confidence * 100f),
                        color = if (obs.accepted) Color(0xFF2CA36B) else Color(0xFFE35D5B),
                        style = MaterialTheme.typography.titleMedium
                    )
                }
                Column(horizontalAlignment = Alignment.End) {
                    Text(
                        text = "视觉门控",
                        color = FlatUiColors.TextMuted,
                        style = MaterialTheme.typography.labelSmall
                    )
                    Text(
                        text = if (obs.accepted) "通过" else "拒绝",
                        color = if (obs.accepted) Color(0xFF2CA36B) else Color(0xFFE35D5B),
                        style = MaterialTheme.typography.titleMedium
                    )
                }
            }
        }

        if (trackedNodes.isNotEmpty()) {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                    text = "节点 ${trackedNodes.size}",
                    color = FlatUiColors.TextMuted,
                    style = MaterialTheme.typography.labelSmall
                )
                trackedNodes.take(6).forEach { node ->
                    val idText = node.nodeId?.toString() ?: "--"
                    val calib = if (node.needsVisualCalibration) " · 需视觉校准" else ""
                    Text(
                        text = "#${node.trackKey} id=$idText 置信=${
                            "%.0f".format((node.observation?.confidence ?: 0f) * 100f)
                        }%$calib",
                        color = FlatUiColors.TextPrimary,
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }

        if (tuning.enableBleImu) {
            Text(
                text = if (bleScanning) "BLE 扫描中… ($bleConnectedCount)" else "BLE 已连接 $bleConnectedCount",
                color = if (bleConnectedCount > 0) Color(0xFF2CA36B) else FlatUiColors.TextMuted,
                style = MaterialTheme.typography.labelMedium
            )
        }

        pnpResult?.let { result ->
            HorizontalDivider(
                modifier = Modifier.padding(vertical = 12.dp),
                thickness = DividerDefaults.Thickness,
                color = FlatUiColors.Border
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text(
                        text = "距离",
                        color = FlatUiColors.TextMuted,
                        style = MaterialTheme.typography.labelSmall
                    )
                    Text(
                        text = pnpCalculator.formatDistance(result.distance),
                        color = FlatUiColors.Accent,
                        style = MaterialTheme.typography.titleMedium
                    )
                }

                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "方位角",
                        color = FlatUiColors.TextMuted,
                        style = MaterialTheme.typography.labelSmall
                    )
                    Text(
                        text = pnpCalculator.formatAngle(result.azimuth),
                        color = FlatUiColors.TextPrimary,
                        style = MaterialTheme.typography.titleMedium
                    )
                }

                Column(horizontalAlignment = Alignment.End) {
                    Text(
                        text = "仰角",
                        color = FlatUiColors.TextMuted,
                        style = MaterialTheme.typography.labelSmall
                    )
                    Text(
                        text = pnpCalculator.formatAngle(result.elevation),
                        color = FlatUiColors.TextPrimary,
                        style = MaterialTheme.typography.titleMedium
                    )
                }
            }

            HorizontalDivider(
                modifier = Modifier.padding(vertical = 8.dp),
                thickness = DividerDefaults.Thickness,
                color = FlatUiColors.Border
            )

            result.pose6DOF?.let { pose ->
                Column(
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = "六自由度位姿",
                        color = FlatUiColors.Accent,
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
                                text = "中心",
                                color = Color(0xFF95A5A6),
                                style = MaterialTheme.typography.labelSmall
                            )
                            Text(
                                text = pnpCalculator.formatPoint3D(pose.position),
                                color = FlatUiColors.Accent,
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }

                    HorizontalDivider(
                        modifier = Modifier.padding(vertical = 4.dp),
                        thickness = DividerDefaults.Thickness,
                        color = FlatUiColors.Border
                    )

                    Text(
                        text = "姿态角",
                        color = FlatUiColors.TextMuted,
                        style = MaterialTheme.typography.labelSmall
                    )

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column {
                            Text(
                                text = "横滚",
                                color = FlatUiColors.TextMuted,
                                style = MaterialTheme.typography.labelSmall
                            )
                            Text(
                                text = pnpCalculator.formatAngle(pose.roll),
                                color = Color(0xFFE35D5B),
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }

                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(
                                text = "俯仰",
                                color = FlatUiColors.TextMuted,
                                style = MaterialTheme.typography.labelSmall
                            )
                            Text(
                                text = pnpCalculator.formatAngle(pose.pitch),
                                color = Color(0xFF2CA36B),
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }

                        Column(horizontalAlignment = Alignment.End) {
                            Text(
                                text = "偏航",
                                color = FlatUiColors.TextMuted,
                                style = MaterialTheme.typography.labelSmall
                            )
                            Text(
                                text = pnpCalculator.formatAngle(pose.yaw),
                                color = Color(0xFF2F8FCE),
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }
                }
            }
        }
    }
}

