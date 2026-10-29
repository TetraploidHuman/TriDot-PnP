package org.example.tridotpnp

import android.view.MotionEvent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.hapticfeedback.HapticFeedback
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.input.pointer.pointerInteropFilter
import androidx.camera.view.PreviewView
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

@Composable
internal fun ProbabilityEditOverlay(
    probabilityMatrix: ProbabilityMatrix32x24,
    previewSize: Pair<Int, Int>?,
    previewViewRef: PreviewView?,
    probabilityEditMode: ProbabilityEditMode,
    probabilityEditStep: Float,
    liveProbabilityMatrix: ProbabilityMatrix32x24?,
    dragAnchorProbabilityDisplayCell: Pair<Int, Int>?,
    onLiveProbabilityMatrixChange: (ProbabilityMatrix32x24) -> Unit,
    onSelectedProbabilityDisplayCellChange: (Pair<Int, Int>?) -> Unit,
    onLastEditedProbabilityDisplayCellChange: (Pair<Int, Int>?) -> Unit,
    onDragAnchorProbabilityDisplayCellChange: (Pair<Int, Int>?) -> Unit,
    onProbabilityCellSelected: (Pair<Int, Int>?) -> Unit,
    onProbabilityMatrixChange: (ProbabilityMatrix32x24) -> Unit,
    haptic: HapticFeedback
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .pointerInteropFilter { event ->
                fun offsetToDisplayCell(x: Float, y: Float): Pair<Int, Int>? {
                    val sizeInfo = previewSize ?: return null
                    val imageWidth = sizeInfo.first
                    val imageHeight = sizeInfo.second
                    val drawWidth = previewViewRef?.width?.toFloat()?.coerceAtLeast(1f) ?: return null
                    val drawHeight = previewViewRef?.height?.toFloat()?.coerceAtLeast(1f) ?: return null
                    val scale = min(drawWidth / imageWidth.toFloat(), drawHeight / imageHeight.toFloat())
                    val imageDrawWidth = imageWidth * scale
                    val imageDrawHeight = imageHeight * scale
                    if (x < 0f || y < 0f || x > imageDrawWidth || y > imageDrawHeight) {
                        return null
                    }

                    val rows = probabilityMatrix.rowsForImage(imageWidth, imageHeight)
                    val cols = probabilityMatrix.colsForImage(imageWidth, imageHeight)
                    val cellWidth = imageDrawWidth / cols.toFloat()
                    val cellHeight = imageDrawHeight / rows.toFloat()
                    val col = (x / cellWidth).toInt().coerceIn(0, cols - 1)
                    val row = (y / cellHeight).toInt().coerceIn(0, rows - 1)
                    return row to col
                }

                fun displayPath(from: Pair<Int, Int>, to: Pair<Int, Int>): List<Pair<Int, Int>> {
                    val rowDelta = to.first - from.first
                    val colDelta = to.second - from.second
                    val steps = max(abs(rowDelta), abs(colDelta))
                    if (steps == 0) return listOf(from)

                    return buildList {
                        for (step in 0..steps) {
                            val t = step.toFloat() / steps.toFloat()
                            val row = round(from.first + rowDelta * t).toInt()
                            val col = round(from.second + colDelta * t).toInt()
                            val cell = row to col
                            if (lastOrNull() != cell) add(cell)
                        }
                    }
                }

                fun applyProbabilityEditPath(targetCell: Pair<Int, Int>) {
                    val sizeInfo = previewSize ?: return
                    val imageWidth = sizeInfo.first
                    val imageHeight = sizeInfo.second
                    val delta = if (probabilityEditMode == ProbabilityEditMode.Increase) {
                        probabilityEditStep
                    } else {
                        -probabilityEditStep
                    }
                    val startCell = dragAnchorProbabilityDisplayCell ?: targetCell
                    if (dragAnchorProbabilityDisplayCell != null && startCell == targetCell) {
                        onSelectedProbabilityDisplayCellChange(targetCell)
                        return
                    }
                    val pathCells = if (dragAnchorProbabilityDisplayCell == null) {
                        listOf(targetCell)
                    } else {
                        displayPath(startCell, targetCell).drop(1)
                    }
                    if (pathCells.isEmpty()) return

                    var nextMatrix = liveProbabilityMatrix ?: probabilityMatrix
                    pathCells.forEach { (row, col) ->
                        nextMatrix = nextMatrix.updateDisplayedCell(
                            row = row,
                            col = col,
                            imageWidth = imageWidth,
                            imageHeight = imageHeight,
                            delta = delta
                        )
                        onLastEditedProbabilityDisplayCellChange(row to col)
                    }

                    onLiveProbabilityMatrixChange(nextMatrix)
                    onSelectedProbabilityDisplayCellChange(targetCell)
                    onDragAnchorProbabilityDisplayCellChange(targetCell)
                    val mappedCell = if (imageWidth > imageHeight) {
                        targetCell.second to targetCell.first
                    } else {
                        targetCell
                    }
                    onProbabilityCellSelected(mappedCell)
                    onProbabilityMatrixChange(nextMatrix)
                    haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                }

                when (event.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        onLastEditedProbabilityDisplayCellChange(null)
                        onDragAnchorProbabilityDisplayCellChange(null)
                        offsetToDisplayCell(event.x, event.y)?.let { applyProbabilityEditPath(it) }
                        true
                    }

                    MotionEvent.ACTION_MOVE -> {
                        for (index in 0 until event.historySize) {
                            offsetToDisplayCell(
                                event.getHistoricalX(index),
                                event.getHistoricalY(index)
                            )?.let { applyProbabilityEditPath(it) }
                        }
                        offsetToDisplayCell(event.x, event.y)?.let { applyProbabilityEditPath(it) }
                        true
                    }

                    MotionEvent.ACTION_UP,
                    MotionEvent.ACTION_CANCEL -> {
                        onLastEditedProbabilityDisplayCellChange(null)
                        onDragAnchorProbabilityDisplayCellChange(null)
                        true
                    }

                    else -> true
                }
            }
    )
}
