package org.example.tridotpnp

import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DotPatchClassifierTest {

    private val root: File = File(System.getProperty("user.dir") ?: ".")
    private fun projectFile(rel: String): File {
        val direct = File(root, rel)
        if (direct.exists()) return direct
        return File(root, "composeApp/$rel")
    }

    private fun classifier(): DotPatchClassifier {
        val bin = projectFile("src/androidMain/assets/dot_cls/dot_cls.bin")
        assertTrue(bin.exists(), "missing weights: ${bin.absolutePath}")
        return DotPatchClassifier(DotPatchWeights.fromStream(bin.inputStream()))
    }

    @Test
    fun matchesPythonReferenceProbabilities() {
        val clf = classifier()
        val fixtures = projectFile("src/androidUnitTest/resources/dot_cls/fixtures.tsv")
        assertTrue(fixtures.exists(), "missing fixtures: ${fixtures.absolutePath}")
        val lines = fixtures.readLines().filter { it.isNotBlank() }
        val rows = lines.drop(1)
        assertTrue(rows.size >= 20, "too few fixtures")
        var maxDiff = 0f
        for (line in rows) {
            val cells = line.split("\t")
            val expected = cells[1].toFloat()
            val patch = FloatArray(225) { i -> cells[2 + i].toFloat() }
            val prob = clf.score(patch)
            maxDiff = maxOf(maxDiff, kotlin.math.abs(prob - expected))
        }
        // float32 accumulation order differs from PyTorch; 2e-3 is still far
        // below the score-blend resolution used by the detector.
        assertTrue(maxDiff < 2e-3f, "max |kotlin-python| = $maxDiff")
    }

    @Test
    fun separatesPositivesFromNegatives() {
        val clf = classifier()
        val fixtures = projectFile("src/androidUnitTest/resources/dot_cls/fixtures.tsv")
        val rows = fixtures.readLines().filter { it.isNotBlank() }.drop(1)
        var posSum = 0f
        var negSum = 0f
        var posN = 0
        var negN = 0
        for (line in rows) {
            val cells = line.split("\t")
            val label = cells[0].toInt()
            val patch = FloatArray(225) { i -> cells[2 + i].toFloat() }
            val p = clf.score(patch)
            if (label == 1) { posSum += p; posN++ } else { negSum += p; negN++ }
        }
        assertTrue(posN > 0 && negN > 0)
        assertTrue(posSum / posN > negSum / negN + 0.1f,
            "pos mean ${posSum / posN} vs neg mean ${negSum / negN}")
    }

    @Test
    fun weightsLoaderRejectsMalformedByteArrays() {
        var failed = false
        try {
            DotPatchWeights.fromBytes(ByteArray(10))
        } catch (e: IllegalArgumentException) {
            failed = true
        }
        assertTrue(failed)
    }

    @Test
    fun patchExtractorMatchesPythonShapeAndConstantMap() {
        val w = 64
        val h = 48
        val map = FloatArray(w * h) { 120f }
        val p = DotPatchExtractor.scaled(map, w, h, 30f, 20f, 6f)
        assertEquals(225, p.size)
        assertTrue(p.all { kotlin.math.abs(it - 120f) < 1e-3f })
    }
}

/**
 * round 14: 没有上一帧三点时（first frame / tracking lost），尺度归一化 patch
 * 仍然要用「本帧候选的中位等效半径」而不是退化成固定 15px 窗口。
 */
class FarSpotPriorFreeScaleTest {

    private val root: java.io.File = java.io.File(System.getProperty("user.dir") ?: ".")
    private fun projectFile(rel: String): java.io.File {
        val direct = java.io.File(root, rel)
        return if (direct.exists()) direct else java.io.File(root, "composeApp/$rel")
    }

    private fun classifier(): DotPatchClassifier {
        val bin = projectFile("src/androidMain/assets/dot_cls/dot_cls.bin")
        return DotPatchClassifier(DotPatchWeights.fromStream(bin.inputStream()))
    }

    @Test
    fun detectsSpotsWithoutPreviousPoseOrEdge() {
        val w = 160
        val h = 120
        val pixels = IntArray(w * h) { 0xFF101010.toInt() }
        val dots = listOf(50 to 40, 90 to 45, 70 to 78)
        val colors = listOf(0xFFFF2020.toInt(), 0xFF20FF20.toInt(), 0xFF2020FF.toInt())
        for ((i, p) in dots.withIndex()) {
            // 2x2 dots -> the classifier has to cope with tiny patches and no prior
            for (dy in 0..1) for (dx in 0..1) {
                pixels[(p.second + dy) * w + p.first + dx] = colors[i]
            }
        }
        val det = FarSpotDetector()
        det.classifier = classifier()
        val spots = det.detectRgb(pixels, w, h, expectedEdgePx = 0f)
        assertTrue(spots != null, "prior-free detection returned null")
        assertEquals(3, spots!!.size)
        val found = spots.map { it.color }.toSet()
        assertEquals(3, found.size)
    }
}
