package com.google.mediapipe.examples.facelandmarker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class HeadPoseTemporalTest {

    @Test
    fun baselineIsClassifiedAsFrontal() {
        val analyzer = analyzer()

        val result = analyzer.update(
            0L,
            12.2,
            -2.9,
            0.1,
            true
        )

        assertEquals("FRONTAL", result.direction)
        assertFalse(result.offCenter)
    }

    @Test
    fun sustainedLeftTurnCreatesOneEvent() {
        val analyzer = analyzer()

        analyzer.update(
            0L,
            12.2,
            -42.0,
            0.1,
            true
        )
        val event = analyzer.update(
            1600L,
            12.2,
            -42.0,
            0.1,
            true
        )
        val continued = analyzer.update(
            2500L,
            12.2,
            -42.0,
            0.1,
            true
        )

        assertTrue(
            event.direction.contains("IZQUIERDA")
        )
        assertTrue(event.poseAwayEvent)
        assertEquals(1, event.poseAwayCount)
        assertFalse(continued.poseAwayEvent)
        assertEquals(1, continued.poseAwayCount)
    }

    @Test
    fun briefTurnDoesNotCreateEvent() {
        val analyzer = analyzer()

        analyzer.update(
            0L,
            12.2,
            42.0,
            0.1,
            true
        )
        analyzer.update(
            800L,
            12.2,
            42.0,
            0.1,
            true
        )
        val result = analyzer.update(
            900L,
            12.2,
            -2.9,
            0.1,
            true
        )

        assertEquals(0, result.poseAwayCount)
        assertFalse(result.poseAwayEvent)
    }

    @Test
    fun downAndReturnCreatesOneNod() {
        val analyzer = analyzer()

        analyzer.update(
            0L,
            30.0,
            -2.9,
            0.1,
            true
        )
        analyzer.update(
            600L,
            30.0,
            -2.9,
            0.1,
            true
        )
        val result = analyzer.update(
            900L,
            12.2,
            -2.9,
            0.1,
            true
        )

        assertTrue(result.nodEvent)
        assertEquals(1, result.nodCount)
    }

    @Test
    fun prolongedDownPoseIsNotNod() {
        val analyzer = analyzer()

        analyzer.update(
            0L,
            30.0,
            -2.9,
            0.1,
            true
        )
        analyzer.update(
            3000L,
            30.0,
            -2.9,
            0.1,
            true
        )
        val result = analyzer.update(
            3100L,
            12.2,
            -2.9,
            0.1,
            true
        )

        assertFalse(result.nodEvent)
        assertEquals(0, result.nodCount)
    }

    @Test
    fun invalidFrameCancelsPendingEvents() {
        val analyzer = analyzer()

        analyzer.update(
            0L,
            30.0,
            -42.0,
            0.1,
            true
        )
        analyzer.update(
            700L,
            30.0,
            -42.0,
            0.1,
            true
        )
        analyzer.update(
            800L,
            Double.NaN,
            Double.NaN,
            Double.NaN,
            false
        )
        val result = analyzer.update(
            2000L,
            12.2,
            -2.9,
            0.1,
            true
        )

        assertEquals(0, result.poseAwayCount)
        assertEquals(0, result.nodCount)
        assertFalse(result.poseAwayEvent)
        assertFalse(result.nodEvent)
    }

    private fun analyzer(): HeadPoseTemporalAnalyzer {
        return HeadPoseTemporalAnalyzer(
            smoothingTauS = 0.01
        )
    }
}
