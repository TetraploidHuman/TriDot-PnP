package org.example.tridotpnp

data class ProbabilityMatrix32x24(
    val values: FloatArray
) {
    companion object {
        const val ROWS = 32
        const val COLS = 24
        const val SIZE = ROWS * COLS
        const val MIN_WEIGHT = 0.01f
        const val MAX_WEIGHT = 1.5f
    }

    init {
        require(values.size == SIZE) {
            "ProbabilityMatrix32x24 requires $SIZE values, got ${values.size}"
        }
    }

    fun rowsForImage(imageWidth: Int, imageHeight: Int): Int {
        return if (imageWidth > imageHeight) COLS else ROWS
    }

    fun colsForImage(imageWidth: Int, imageHeight: Int): Int {
        return if (imageWidth > imageHeight) ROWS else COLS
    }

    fun weightAtGridCell(row: Int, col: Int, imageWidth: Int, imageHeight: Int): Float {
        val isLandscape = imageWidth > imageHeight
        val (mappedRow, mappedCol) = if (isLandscape) {
            // 横屏下显示网格采用 24x32，底层矩阵做转置读取，保证绘制和检测一致。
            col to row
        } else {
            row to col
        }
        val safeRow = mappedRow.coerceIn(0, ROWS - 1)
        val safeCol = mappedCol.coerceIn(0, COLS - 1)
        return values[safeRow * COLS + safeCol]
    }

    fun weightAtPixel(x: Int, y: Int, imageWidth: Int, imageHeight: Int): Float {
        if (imageWidth <= 0 || imageHeight <= 0) return 1f
        val rows = rowsForImage(imageWidth, imageHeight)
        val cols = colsForImage(imageWidth, imageHeight)
        val col = ((x.toFloat() / imageWidth) * cols).toInt().coerceIn(0, cols - 1)
        val row = ((y.toFloat() / imageHeight) * rows).toInt().coerceIn(0, rows - 1)
        return weightAtGridCell(row, col, imageWidth, imageHeight)
    }

    fun weightAt(row: Int, col: Int): Float {
        val safeRow = row.coerceIn(0, ROWS - 1)
        val safeCol = col.coerceIn(0, COLS - 1)
        return values[safeRow * COLS + safeCol]
    }

    fun updateCell(row: Int, col: Int, delta: Float): ProbabilityMatrix32x24 {
        val safeRow = row.coerceIn(0, ROWS - 1)
        val safeCol = col.coerceIn(0, COLS - 1)
        val index = safeRow * COLS + safeCol
        val nextValues = values.copyOf()
        nextValues[index] = (nextValues[index] + delta).coerceIn(MIN_WEIGHT, MAX_WEIGHT)
        return ProbabilityMatrix32x24(nextValues)
    }

    fun updateDisplayedCell(
        row: Int,
        col: Int,
        imageWidth: Int,
        imageHeight: Int,
        delta: Float
    ): ProbabilityMatrix32x24 {
        val isLandscape = imageWidth > imageHeight
        val mappedRow = if (isLandscape) col else row
        val mappedCol = if (isLandscape) row else col
        return updateCell(mappedRow, mappedCol, delta)
    }
}

data class ProbabilityRegion(
    val rows: IntRange,
    val cols: IntRange,
    val weight: Float
)

private fun region(rows: IntRange, cols: IntRange, weight: Float): ProbabilityRegion {
    return ProbabilityRegion(rows = rows, cols = cols, weight = weight)
}

fun buildManualProbabilityMatrix32x24(): ProbabilityMatrix32x24 {
    val rows = ProbabilityMatrix32x24.ROWS
    val cols = ProbabilityMatrix32x24.COLS

    val values = FloatArray(ProbabilityMatrix32x24.SIZE) { 0f }

    val manualRegions = listOf(
        // 1) 全局基础概率（使用 region 覆盖）
        region(rows = 0 until rows, cols = 0 until cols, weight = 0.4f),
        // 2) 局部高/低概率区域继续覆盖

        region(rows = 12..12, cols = 16..16, weight = 0.9f)
    )

    manualRegions.forEach { region ->
        val rStart = region.rows.first.coerceIn(0, rows - 1)
        val rEnd = region.rows.last.coerceIn(rStart, rows - 1)
        val cStart = region.cols.first.coerceIn(0, cols - 1)
        val cEnd = region.cols.last.coerceIn(cStart, cols - 1)
        val w = region.weight.coerceIn(
            ProbabilityMatrix32x24.MIN_WEIGHT,
            ProbabilityMatrix32x24.MAX_WEIGHT
        )

        for (r in rStart..rEnd) {
            val base = r * cols
            for (c in cStart..cEnd) {
                values[base + c] = w
            }
        }
    }

    return ProbabilityMatrix32x24(values)
}
