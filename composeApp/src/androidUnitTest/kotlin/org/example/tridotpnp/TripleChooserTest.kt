package org.example.tridotpnp

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * 主线 vs 对比度前端的取舍策略。
 *
 * 依据：Python 端侧链路模拟里主线在 realistic 域 det@10≈0.22、对比度前端≈0.80，
 * 因此默认「对比度前端优先」，但只在它可信（位姿校验通过或几何自洽）时生效。
 */
class TripleChooserTest {
    @Test
    fun farPreferredWhenValidated() {
        assertEquals(
            TripleSource.FAR,
            TripleChooser.choose(
                mainPlausible = true, farAvailable = true, farValidated = true,
                farPlausible = true, farPrimary = true
            )
        )
    }

    @Test
    fun farPreferredOnlyIfPlausible() {
        // 对比度前端给了三点但几何自洽不过（例如镜像解）→ 仍用主线
        assertEquals(
            TripleSource.MAIN,
            TripleChooser.choose(
                mainPlausible = true, farAvailable = true, farValidated = false,
                farPlausible = false, farPrimary = true
            )
        )
    }

    @Test
    fun legacySemanticsWhenPrimaryOff() {
        // 关掉「对比度前端优先」后，只要主线可信就不动
        assertEquals(
            TripleSource.MAIN,
            TripleChooser.choose(
                mainPlausible = true, farAvailable = true, farValidated = true,
                farPlausible = true, farPrimary = false
            )
        )
        // 主线不可信时才切到对比度前端
        assertEquals(
            TripleSource.FAR,
            TripleChooser.choose(
                mainPlausible = false, farAvailable = true, farValidated = false,
                farPlausible = true, farPrimary = false
            )
        )
    }

    @Test
    fun fallsBackToMainWhenFarUnavailable() {
        assertEquals(
            TripleSource.MAIN,
            TripleChooser.choose(
                mainPlausible = false, farAvailable = false, farValidated = false,
                farPlausible = false, farPrimary = true
            )
        )
    }
}
