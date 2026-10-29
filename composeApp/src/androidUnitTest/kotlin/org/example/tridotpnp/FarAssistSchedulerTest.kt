package org.example.tridotpnp

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class FarAssistSchedulerTest {
    @Test
    fun periodOneAlwaysRuns() {
        assertTrue(FarAssistScheduler.shouldRun(1, mainPlausible = true, period = 1))
        assertTrue(FarAssistScheduler.shouldRun(7, mainPlausible = true, period = 0))
    }

    @Test
    fun implausibleMainAlwaysRuns() {
        assertTrue(FarAssistScheduler.shouldRun(1, mainPlausible = false, period = 3))
        assertTrue(FarAssistScheduler.shouldRun(2, mainPlausible = false, period = 3))
    }

    @Test
    fun periodicWhenMainPlausible() {
        assertTrue(FarAssistScheduler.shouldRun(2, mainPlausible = true, period = 2))
        assertFalse(FarAssistScheduler.shouldRun(3, mainPlausible = true, period = 2))
        assertTrue(FarAssistScheduler.shouldRun(4, mainPlausible = true, period = 2))
    }

    @Test
    fun honorsLargerPeriods() {
        assertFalse(FarAssistScheduler.shouldRun(1, mainPlausible = true, period = 3))
        assertTrue(FarAssistScheduler.shouldRun(3, mainPlausible = true, period = 3))
        assertTrue(FarAssistScheduler.shouldRun(0, mainPlausible = true, period = 3))
    }
}
