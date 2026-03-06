package org.example.tridotpnp

data class ProbabilityMatrix128x96(
    val values: FloatArray
) {
    companion object {
        const val ROWS = 128
        const val COLS = 96
        const val SIZE = ROWS * COLS
    }

    init {
        require(values.size == SIZE) {
            "ProbabilityMatrix128x96 requires $SIZE values, got ${values.size}"
        }
    }

    fun weightAtPixel(x: Int, y: Int, imageWidth: Int, imageHeight: Int): Float {
        if (imageWidth <= 0 || imageHeight <= 0) return 1f
        val col = ((x.toFloat() / imageWidth) * COLS).toInt().coerceIn(0, COLS - 1)
        val row = ((y.toFloat() / imageHeight) * ROWS).toInt().coerceIn(0, ROWS - 1)
        return values[row * COLS + col]
    }
}

private const val MIN_PROBABILITY_WEIGHT = 0.01f
private const val MAX_PROBABILITY_WEIGHT = 1.5f

data class ProbabilityRegion(
    val rows: IntRange,
    val cols: IntRange,
    val weight: Float
)

private fun region(rows: IntRange, cols: IntRange, weight: Float): ProbabilityRegion {
    return ProbabilityRegion(rows = rows, cols = cols, weight = weight)
}

fun buildManualProbabilityMatrix128x96(): ProbabilityMatrix128x96 {
    val rows = ProbabilityMatrix128x96.ROWS
    val cols = ProbabilityMatrix128x96.COLS

    val values = FloatArray(ProbabilityMatrix128x96.SIZE) { 0f }

    val manualRegions = listOf(
        // 1) 全局基础概率（使用 region 覆盖）
        region(rows = 0 until rows, cols = 0 until cols, weight = 0.4f),
        // 2) 局部高/低概率区域继续覆盖
        region(rows = 44..84, cols = 28..67, weight = 0.60f),
        region(rows = 50..78, cols = 34..61, weight = 0.80f),
        region(rows = 0..18, cols = 0..95, weight = 0.4f),
        region(rows = 108..127, cols = 0..95, weight = 0.4f)
    )

    manualRegions.forEach { region ->
        val rStart = region.rows.first.coerceIn(0, rows - 1)
        val rEnd = region.rows.last.coerceIn(rStart, rows - 1)
        val cStart = region.cols.first.coerceIn(0, cols - 1)
        val cEnd = region.cols.last.coerceIn(cStart, cols - 1)
        val w = region.weight.coerceIn(MIN_PROBABILITY_WEIGHT, MAX_PROBABILITY_WEIGHT)

        for (r in rStart..rEnd) {
            val base = r * cols
            for (c in cStart..cEnd) {
                values[base + c] = w
            }
        }
    }

    return ProbabilityMatrix128x96(values)
}
