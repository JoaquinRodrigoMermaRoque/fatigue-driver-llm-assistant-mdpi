package com.google.mediapipe.examples.facelandmarker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FatigueHistoryTest {

    @Test
    fun recorderStoresOnlySignificantEvents() {
        val repository = InMemoryHistory()
        val recorder = FatigueEventRecorder(
            repository = repository,
            subjectId = "subject",
            sessionId = "session"
        )

        recorder.record(frame(0L))
        recorder.record(
            frame(
                timestampMs = 1000L,
                yawnEvent = true,
                nodEvent = true
            )
        )

        assertEquals(2, repository.records.size)
        assertEquals(
            listOf("YAWN", "NOD"),
            repository.records.map {
                it.eventType
            }
        )
    }

    @Test
    fun faceLossAndRecoveryAreStoredOnce() {
        val repository = InMemoryHistory()
        val recorder =
            FatigueEventRecorder(repository)

        recorder.record(frame(0L, faceValid = true))
        recorder.record(
            frame(1000L, faceValid = false)
        )
        recorder.record(
            frame(2000L, faceValid = false)
        )
        recorder.record(
            frame(3000L, faceValid = true)
        )
        recorder.record(
            frame(4000L, faceValid = true)
        )

        assertEquals(
            listOf("FACE_LOST", "FACE_RECOVERED"),
            repository.records.map {
                it.eventType
            }
        )
    }

    @Test
    fun transientFaceLossIsNotStored() {
        val repository = InMemoryHistory()
        val recorder = FatigueEventRecorder(
            repository = repository,
            faceTransitionConfirmMs = 1000L
        )

        recorder.record(frame(0L, faceValid = true))
        recorder.record(
            frame(100L, faceValid = false)
        )
        recorder.record(
            frame(500L, faceValid = true)
        )

        assertEquals(
            emptyList<String>(),
            repository.records.map {
                it.eventType
            }
        )
    }

    @Test
    fun stateTransitionPreservesExplanation() {
        val repository = InMemoryHistory()
        val recorder =
            FatigueEventRecorder(repository)

        recorder.record(
            frame(
                timestampMs = 1000L,
                state = FatigueState.CRITICAL,
                stateEvent =
                    "prolonged_eye_closure",
                score = 0.8,
                reasons = listOf(
                    "closure",
                    "perclos"
                )
            )
        )

        val record = repository.records.single()
        assertEquals("STATE_TRANSITION", record.eventType)
        assertEquals(
            "prolonged_eye_closure",
            record.detail
        )
        assertEquals(
            listOf("closure", "perclos"),
            record.reasons
        )
    }

    @Test
    fun acknowledgementIsNotAStateTransition() {
        val repository = InMemoryHistory()
        val recorder =
            FatigueEventRecorder(repository)

        recorder.record(
            frame(
                timestampMs = 1000L,
                state = FatigueState.CRITICAL,
                stateEvent =
                    "critical_acknowledged"
            )
        )

        val record = repository.records.single()
        assertEquals(
            "ACKNOWLEDGEMENT",
            record.eventType
        )
        assertEquals(
            "critical_acknowledged",
            record.detail
        )
    }

    @Test
    fun summaryUsesRequestedTimeHorizon() {
        val records = listOf(
            historyRecord(
                timestampMs = 1000L,
                eventType = "YAWN",
                score = 0.2,
                reasons = listOf("yawn")
            ),
            historyRecord(
                timestampMs = 9000L,
                eventType = "NOD",
                score = 0.6,
                reasons = listOf("nod")
            ),
            historyRecord(
                timestampMs = 10_000L,
                eventType = "STATE_TRANSITION",
                state = FatigueState.CRITICAL,
                score = 0.9,
                reasons = listOf("nod")
            )
        )

        val summary =
            FatigueHistorySummarizer.summarize(
                records = records,
                nowMs = 10_000L,
                horizonMs = 2000L
            )

        assertEquals(2, summary.eventCount)
        assertEquals(1, summary.countsByType["NOD"])
        assertEquals(0.9, summary.maximumScore, 1e-9)
        assertEquals(
            FatigueState.CRITICAL,
            summary.latestState
        )
        assertEquals(1, summary.criticalTransitions)
        assertEquals("nod", summary.mostFrequentReason)
    }

    @Test
    fun emptySummaryIsSafe() {
        val summary =
            FatigueHistorySummarizer.summarize(
                records = emptyList(),
                nowMs = 10_000L,
                horizonMs = 1000L
            )

        assertEquals(0, summary.eventCount)
        assertEquals(0.0, summary.maximumScore, 1e-9)
        assertEquals(
            FatigueState.NORMAL,
            summary.latestState
        )
        assertNull(summary.mostFrequentReason)
    }

    private fun frame(
        timestampMs: Long,
        faceValid: Boolean = true,
        state: FatigueState = FatigueState.NORMAL,
        stateEvent: String? = null,
        score: Double = 0.0,
        reasons: List<String> = emptyList(),
        yawnEvent: Boolean = false,
        nodEvent: Boolean = false
    ): FatigueHistoryFrame {
        return FatigueHistoryFrame(
            timestampMs = timestampMs,
            faceValid = faceValid,
            state = state,
            stateEvent = stateEvent,
            score = score,
            reasons = reasons,
            closureS = 0.0,
            perclosRatio = 0.0,
            blinkCount = 0,
            yawnEvent = yawnEvent,
            yawnsLast5Min =
                if (yawnEvent) 1 else 0,
            nodEvent = nodEvent,
            nodsLast2Min =
                if (nodEvent) 1 else 0,
            gazeAwayEvent = false,
            gazeAwayCount = 0,
            poseAwayEvent = false,
            poseAwayCount = 0
        )
    }

    private fun historyRecord(
        timestampMs: Long,
        eventType: String,
        state: FatigueState = FatigueState.NORMAL,
        score: Double,
        reasons: List<String>
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

    private class InMemoryHistory :
        FatigueHistoryRepository {
        val records =
            mutableListOf<FatigueHistoryRecord>()

        override fun append(
            record: FatigueHistoryRecord
        ) {
            records += record
        }

        override fun recent(
            limit: Int
        ): List<FatigueHistoryRecord> {
            return records.takeLast(limit)
        }

        override fun count(): Long {
            return records.size.toLong()
        }
    }
}
