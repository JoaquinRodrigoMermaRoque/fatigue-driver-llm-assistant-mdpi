package com.google.mediapipe.examples.facelandmarker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DriverContextTest {

    @Test
    fun summaryProducesCompactStructuredPromptFacts() {
        val nowMs = 20L * 60L * 60L * 1000L
        val wakeMs = 6L * 60L * 60L * 1000L

        val summary = DriverContextSummary(
            entryCount7d = 3,
            latestSleepLastNightHours = 5.5,
            latestSleepAverage3dHours = 6.2,
            latestWakeTimestampMs = wakeMs,
            latestSubjectiveSleepiness = 7,
            latestWorkShiftHours = 10.0,
            latestNightShift = true,
            latestPhysicalDemand =
                RecentActivityLevel.MODERATE,
            latestMentalDemand =
                RecentActivityLevel.INTENSE,
            latestTimestampMs = 1_000L
        )

        val facts = summary.toPromptFacts(
            sessionDurationMinutes = 42L,
            nowMs = nowMs
        )

        assertTrue(facts.contains("session_min=42"))
        assertTrue(
            facts.contains("user_context_entries_7d=3")
        )
        assertTrue(
            facts.contains("sleep_last_night_h=5.5")
        )
        assertTrue(
            facts.contains("sleep_avg_3d_h=6.2")
        )
        assertTrue(facts.contains("hours_awake=14.0"))
        assertTrue(
            facts.contains("sleepiness_1_9=7")
        )
        assertTrue(facts.contains("work_shift_h=10.0"))
        assertTrue(facts.contains("night_shift=true"))
        assertTrue(
            facts.contains(
                "physical_demand=moderate"
            )
        )
        assertTrue(
            facts.contains(
                "mental_demand=intense"
            )
        )
    }

    @Test
    fun emptySummaryDoesNotInventPersonalContext() {
        val summary = DriverContextSummary(
            entryCount7d = 0,
            latestSleepLastNightHours = null,
            latestSleepAverage3dHours = null,
            latestWakeTimestampMs = null,
            latestSubjectiveSleepiness = null,
            latestWorkShiftHours = null,
            latestNightShift = null,
            latestPhysicalDemand = null,
            latestMentalDemand = null,
            latestTimestampMs = null
        )

        val facts = summary.toPromptFacts(
            sessionDurationMinutes = 5L,
            nowMs = 10_000L
        )

        assertEquals(
            listOf(
                "session_min=5",
                "user_context_entries_7d=0",
                "user_context=none"
            ),
            facts
        )
    }
}
