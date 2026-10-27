package com.google.mediapipe.examples.facelandmarker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FatigueTemporalTest {

    @Test
    fun eyeAnalyzerCountsOneNormalBlink() {
        val analyzer = EyeTemporalAnalyzer(
            earThreshold = 0.14
        )

        analyzer.update(0L, 0.26, true)
        analyzer.update(100L, 0.03, true)
        val result = analyzer.update(
            300L,
            0.26,
            true
        )

        assertTrue(result.blinkEvent)
        assertEquals(1, result.blinkCount)
        assertEquals(0.0, result.closureDurationS, 1e-9)
    }

    @Test
    fun longClosureRaisesPerclosButIsNotBlink() {
        val analyzer = EyeTemporalAnalyzer(
            earThreshold = 0.14
        )

        analyzer.update(0L, 0.26, true)
        analyzer.update(1000L, 0.03, true)
        analyzer.update(2000L, 0.03, true)
        val result = analyzer.update(
            3000L,
            0.26,
            true
        )

        assertEquals(2.0 / 3.0, result.perclos, 1e-9)
        assertEquals(0, result.blinkCount)
        assertFalse(result.blinkEvent)
    }

    @Test
    fun invalidEyeFrameCancelsPendingBlink() {
        val analyzer = EyeTemporalAnalyzer(
            earThreshold = 0.14
        )

        analyzer.update(0L, 0.26, true)
        analyzer.update(100L, 0.03, true)
        analyzer.update(
            200L,
            Double.NaN,
            false
        )
        val result = analyzer.update(
            300L,
            0.26,
            true
        )

        assertEquals(0, result.blinkCount)
        assertFalse(result.blinkEvent)
        assertEquals(2.0 / 3.0, result.validRatio, 1e-9)
    }

    @Test
    fun mouthAnalyzerCountsSustainedOpeningOnce() {
        val analyzer = MouthTemporalAnalyzer(
            marThreshold = 0.27,
            minYawnS = 1.5
        )

        analyzer.update(0L, 0.01, true)
        analyzer.update(100L, 0.54, true)
        val event = analyzer.update(
            1700L,
            0.54,
            true
        )
        val continued = analyzer.update(
            3000L,
            0.54,
            true
        )

        assertTrue(event.yawnEvent)
        assertEquals(1, event.yawnCount)
        assertFalse(continued.yawnEvent)
        assertEquals(1, continued.yawnCount)
    }

    @Test
    fun briefMouthOpenDoesNotCountAsYawn() {
        val analyzer = MouthTemporalAnalyzer(
            marThreshold = 0.27,
            minYawnS = 1.5
        )

        analyzer.update(0L, 0.54, true)
        analyzer.update(900L, 0.54, true)
        val result = analyzer.update(
            1000L,
            0.01,
            true
        )

        assertEquals(0, result.yawnCount)
        assertFalse(result.yawnEvent)
    }

    @Test
    fun invalidMouthFrameCancelsOpening() {
        val analyzer = MouthTemporalAnalyzer(
            marThreshold = 0.27,
            minYawnS = 1.5
        )

        analyzer.update(0L, 0.54, true)
        analyzer.update(1000L, 0.54, true)
        analyzer.update(
            1100L,
            Double.NaN,
            false
        )
        analyzer.update(2000L, 0.54, true)
        val result = analyzer.update(
            3000L,
            0.54,
            true
        )

        assertEquals(0, result.yawnCount)
        assertFalse(result.yawnEvent)
    }

    @Test
    fun gazeAnalyzerCountsSustainedHorizontalAwayOnce() {
        val analyzer = GazeTemporalAnalyzer(
            horizontalLow = 0.37,
            horizontalHigh = 0.62,
            verticalEnabled = false,
            minAwayS = 1.5,
            smoothingTauS = 0.01
        )

        analyzer.update(0L, 0.23, 0.40, true)
        val event = analyzer.update(
            1600L,
            0.23,
            0.40,
            true
        )
        val continued = analyzer.update(
            2500L,
            0.23,
            0.40,
            true
        )

        assertEquals("IZQUIERDA", event.direction)
        assertTrue(event.gazeAwayEvent)
        assertEquals(1, event.gazeAwayCount)
        assertFalse(continued.gazeAwayEvent)
        assertEquals(1, continued.gazeAwayCount)
    }

    @Test
    fun briefGazeAndInvalidFrameDoNotCreateEvent() {
        val analyzer = GazeTemporalAnalyzer(
            verticalEnabled = false,
            minAwayS = 1.5,
            smoothingTauS = 0.01
        )

        analyzer.update(0L, 0.23, 0.40, true)
        analyzer.update(500L, 0.50, 0.40, true)
        analyzer.update(1000L, 0.72, 0.40, true)
        analyzer.update(
            2000L,
            Double.NaN,
            Double.NaN,
            false
        )
        val result = analyzer.update(
            3000L,
            0.50,
            0.40,
            true
        )

        assertEquals(0, result.gazeAwayCount)
        assertFalse(result.gazeAwayEvent)
        assertEquals("CENTRO", result.direction)
    }

    @Test
    fun verticalGazeDoesNotTriggerWhenAxisIsDisabled() {
        val analyzer = GazeTemporalAnalyzer(
            verticalEnabled = false
        )

        val result = analyzer.update(
            0L,
            0.50,
            0.10,
            true
        )

        assertEquals("CENTRO", result.direction)
        assertFalse(result.offCenter)
    }

    @Test(expected = IllegalArgumentException::class)
    fun timestampsMustIncreaseStrictly() {
        val analyzer = EyeTemporalAnalyzer()

        analyzer.update(100L, 0.26, true)
        analyzer.update(100L, 0.26, true)
    }
}
