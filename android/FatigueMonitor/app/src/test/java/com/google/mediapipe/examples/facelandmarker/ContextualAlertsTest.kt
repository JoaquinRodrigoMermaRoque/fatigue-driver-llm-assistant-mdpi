package com.google.mediapipe.examples.facelandmarker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ContextualAlertsTest {

    @Test
    fun normalStateDoesNotEmitAlert() {
        val planner = ContextualAlertPlanner()

        val decision = planner.evaluate(
            snapshot(),
            emptyList()
        )

        assertNull(decision)
    }

    @Test
    fun suspectedTransitionCreatesMediumAlert() {
        val planner = ContextualAlertPlanner()

        val decision = planner.evaluate(
            snapshot(
                state = FatigueState.SUSPECTED,
                stateEvent = "state_suspected",
                score = 0.35,
                reasons = listOf("perclos"),
                perclos = 0.28
            ),
            emptyList()
        )

        requireNotNull(decision)
        assertEquals(
            AlertUrgency.MEDIUM,
            decision.urgency
        )
        assertEquals(
            AlertIntent.MONITOR_ATTENTION,
            decision.intent
        )
        assertTrue(
            decision.languageContext
                .fallbackMessage
                .contains("cierre ocular")
        )
    }

    @Test
    fun criticalAlertUsesCurrentAndHistoricalEvidence() {
        val planner = ContextualAlertPlanner()
        val now = 2_000_000L
        val history = listOf(
            record(
                timestampMs = now - 20_000L,
                eventType = "STATE_TRANSITION",
                state = FatigueState.CRITICAL,
                reasons = listOf("closure"),
                score = 0.9
            ),
            record(
                timestampMs = now - 10_000L,
                eventType = "STATE_TRANSITION",
                state = FatigueState.CRITICAL,
                reasons = listOf("closure"),
                score = 0.8
            )
        )

        val decision = planner.evaluate(
            snapshot(
                timestampMs = now,
                state = FatigueState.CRITICAL,
                stateEvent =
                    "prolonged_eye_closure",
                score = 0.85,
                reasons = listOf(
                    "closure",
                    "perclos"
                ),
                closureS = 2.1,
                perclos = 0.35
            ),
            history
        )

        requireNotNull(decision)
        assertEquals(
            AlertUrgency.CRITICAL,
            decision.urgency
        )
        assertEquals(
            AlertIntent.STOP_SAFELY,
            decision.intent
        )
        assertEquals(
            2,
            decision.history.criticalTransitions
        )
        assertTrue(
            decision.languageContext
                .fallbackMessage
                .contains("2 episodios críticos")
        )
        assertTrue(
            decision.languageContext
                .currentFacts
                .contains("closure_s=2.100")
        )
    }

    @Test
    fun criticalRepeatUsesCooldown() {
        val planner = ContextualAlertPlanner(
            criticalRepeatMs = 30_000L
        )
        val history = listOf(
            record(
                timestampMs = 1000L,
                eventType = "STATE_TRANSITION",
                state = FatigueState.CRITICAL,
                reasons = listOf("closure"),
                score = 0.8
            )
        )
        val first = planner.evaluate(
            snapshot(
                timestampMs = 1000L,
                state = FatigueState.CRITICAL,
                stateEvent = "state_critical",
                score = 0.8,
                reasons = listOf("closure")
            ),
            history
        )
        val early = planner.evaluate(
            snapshot(
                timestampMs = 20_000L,
                state = FatigueState.CRITICAL,
                score = 0.0
            ),
            history
        )
        val repeat = planner.evaluate(
            snapshot(
                timestampMs = 31_000L,
                state = FatigueState.CRITICAL,
                score = 0.0
            ),
            history
        )

        requireNotNull(first)
        assertNull(early)
        requireNotNull(repeat)
        assertEquals(
            "critical_repeat",
            repeat.trigger
        )
        assertTrue(
            repeat.languageContext
                .fallbackMessage
                .contains("cierre ocular prolongado")
        )
    }

    @Test
    fun acknowledgedCriticalStateDoesNotRepeat() {
        val planner = ContextualAlertPlanner()

        val decision = planner.evaluate(
            snapshot(
                state = FatigueState.CRITICAL,
                score = 0.8,
                alertAcknowledged = true
            ),
            emptyList()
        )

        assertNull(decision)
    }

    @Test
    fun briefFaceLossDoesNotEmitVisibilityAlert() {
        val planner = ContextualAlertPlanner(
            visibilityDelayMs = 3000L
        )

        val first = planner.evaluate(
            snapshot(
                timestampMs = 1000L,
                faceValid = false,
                dataSufficient = false
            ),
            emptyList()
        )
        val recovered = planner.evaluate(
            snapshot(
                timestampMs = 3000L,
                faceValid = true
            ),
            emptyList()
        )

        assertNull(first)
        assertNull(recovered)
    }

    @Test
    fun sustainedFaceLossEmitsOneVisibilityAlert() {
        val planner = ContextualAlertPlanner(
            visibilityDelayMs = 3000L,
            visibilityRepeatMs = 60_000L
        )

        planner.evaluate(
            snapshot(
                timestampMs = 1000L,
                faceValid = false,
                dataSufficient = false
            ),
            emptyList()
        )
        val alert = planner.evaluate(
            snapshot(
                timestampMs = 4000L,
                faceValid = false,
                dataSufficient = false
            ),
            emptyList()
        )
        val duplicate = planner.evaluate(
            snapshot(
                timestampMs = 5000L,
                faceValid = false,
                dataSufficient = false
            ),
            emptyList()
        )

        requireNotNull(alert)
        assertEquals(
            AlertIntent.RESTORE_VISIBILITY,
            alert.intent
        )
        assertNull(duplicate)
    }

    @Test
    fun recoveredFaceStartsANewVisibilityEpisode() {
        val planner = ContextualAlertPlanner(
            visibilityDelayMs = 3000L,
            visibilityRepeatMs = 60_000L
        )

        planner.evaluate(
            snapshot(
                timestampMs = 1000L,
                faceValid = false,
                dataSufficient = false
            ),
            emptyList()
        )
        val firstAlert = planner.evaluate(
            snapshot(
                timestampMs = 4000L,
                faceValid = false,
                dataSufficient = false
            ),
            emptyList()
        )
        val recovered = planner.evaluate(
            snapshot(
                timestampMs = 5000L,
                faceValid = true
            ),
            emptyList()
        )
        planner.evaluate(
            snapshot(
                timestampMs = 6000L,
                faceValid = false,
                dataSufficient = false
            ),
            emptyList()
        )
        val secondAlert = planner.evaluate(
            snapshot(
                timestampMs = 9000L,
                faceValid = false,
                dataSufficient = false
            ),
            emptyList()
        )

        requireNotNull(firstAlert)
        requireNotNull(recovered)
        assertEquals(
            AlertIntent.FACE_RESTORED,
            recovered.intent
        )
        requireNotNull(secondAlert)
        assertEquals(
            AlertIntent.RESTORE_VISIBILITY,
            secondAlert.intent
        )
    }

    @Test
    fun recoveryTransitionProvidesGuidance() {
        val planner = ContextualAlertPlanner()

        val decision = planner.evaluate(
            snapshot(
                state = FatigueState.RECOVERY,
                stateEvent = "state_recovery",
                score = 0.15
            ),
            emptyList()
        )

        requireNotNull(decision)
        assertEquals(
            AlertIntent.RECOVERY_GUIDANCE,
            decision.intent
        )
    }

    private fun snapshot(
        timestampMs: Long = 10_000L,
        state: FatigueState = FatigueState.NORMAL,
        stateEvent: String? = null,
        score: Double = 0.0,
        reasons: List<String> = emptyList(),
        closureS: Double = 0.0,
        perclos: Double = 0.0,
        faceValid: Boolean = true,
        dataSufficient: Boolean = true,
        alertAcknowledged: Boolean = false
    ): FatigueAlertSnapshot {
        return FatigueAlertSnapshot(
            timestampMs = timestampMs,
            state = state,
            stateEvent = stateEvent,
            score = score,
            reasons = reasons,
            closureS = closureS,
            perclosRatio = perclos,
            yawnsLast5Min = 0,
            nodsLast2Min = 0,
            faceValid = faceValid,
            dataSufficient = dataSufficient,
            alertAcknowledged =
                alertAcknowledged
        )
    }

    private fun record(
        timestampMs: Long,
        eventType: String,
        state: FatigueState,
        reasons: List<String>,
        score: Double
    ): FatigueHistoryRecord {
        return FatigueHistoryRecord(
            timestampMs = timestampMs,
            sessionId = "session",
            subjectId = "subject",
            eventType = eventType,
            detail = eventType.lowercase(),
            state = state,
            score = score,
            reasons = reasons,
            closureS = 0.0,
            perclosRatio = 0.0,
            blinkCount = 0,
            yawnsLast5Min = 0,
            nodsLast2Min = 0,
            gazeAwayCount = 0,
            poseAwayCount = 0,
            faceValid = true
        )
    }
}
