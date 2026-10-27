/*
 * Local history used to build contextual alerts without sending data
 * outside the device.
 */
package com.google.mediapipe.examples.facelandmarker

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.io.Closeable
import java.util.UUID

data class FatigueHistoryRecord(
    val timestampMs: Long,
    val sessionId: String,
    val subjectId: String,
    val eventType: String,
    val detail: String,
    val state: FatigueState,
    val score: Double,
    val reasons: List<String>,
    val closureS: Double,
    val perclosRatio: Double,
    val blinkCount: Int,
    val yawnsLast5Min: Int,
    val nodsLast2Min: Int,
    val gazeAwayCount: Int,
    val poseAwayCount: Int,
    val faceValid: Boolean
)

data class FatigueHistoryFrame(
    val timestampMs: Long,
    val faceValid: Boolean,
    val state: FatigueState,
    val stateEvent: String?,
    val score: Double,
    val reasons: List<String>,
    val closureS: Double,
    val perclosRatio: Double,
    val blinkCount: Int,
    val yawnEvent: Boolean,
    val yawnsLast5Min: Int,
    val nodEvent: Boolean,
    val nodsLast2Min: Int,
    val gazeAwayEvent: Boolean,
    val gazeAwayCount: Int,
    val poseAwayEvent: Boolean,
    val poseAwayCount: Int
)

data class FatigueHistorySummary(
    val eventCount: Int,
    val countsByType: Map<String, Int>,
    val maximumScore: Double,
    val latestState: FatigueState,
    val criticalTransitions: Int,
    val mostFrequentReason: String?
)

interface FatigueHistoryRepository {
    fun append(record: FatigueHistoryRecord)
    fun recent(limit: Int = 100):
        List<FatigueHistoryRecord>
    fun count(): Long
}

class FatigueEventRecorder(
    private val repository: FatigueHistoryRepository,
    private val subjectId: String = "pilot_001",
    private val sessionId: String =
        UUID.randomUUID().toString(),
    private val faceTransitionConfirmMs: Long = 1000L
) {
    private var confirmedFaceValid: Boolean? = null
    private var candidateFaceValid: Boolean? = null
    private var candidateSinceMs: Long? = null

    init {
        require(faceTransitionConfirmMs >= 0L)
    }

    @Synchronized
    fun resetSessionTracking() {
        confirmedFaceValid = null
        candidateFaceValid = null
        candidateSinceMs = null
    }

    @Synchronized
    fun record(
        frame: FatigueHistoryFrame
    ): List<FatigueHistoryRecord> {
        val events =
            mutableListOf<Pair<String, String>>()

        faceTransitionEvent(frame)?.let {
            events += it
        }

        frame.stateEvent?.let { event ->
            val eventType = if (
                event == "critical_acknowledged"
            ) {
                "ACKNOWLEDGEMENT"
            } else {
                "STATE_TRANSITION"
            }
            events += eventType to event
        }

        if (frame.yawnEvent) {
            events += "YAWN" to "yawn_event"
        }

        if (frame.nodEvent) {
            events += "NOD" to "nod_event"
        }

        if (frame.gazeAwayEvent) {
            events +=
                "GAZE_AWAY" to "gaze_away_event"
        }

        if (frame.poseAwayEvent) {
            events +=
                "POSE_AWAY" to "pose_away_event"
        }

        return events.map { (eventType, detail) ->
            val record = FatigueHistoryRecord(
                timestampMs = frame.timestampMs,
                sessionId = sessionId,
                subjectId = subjectId,
                eventType = eventType,
                detail = detail,
                state = frame.state,
                score = frame.score,
                reasons = frame.reasons,
                closureS = frame.closureS,
                perclosRatio =
                    frame.perclosRatio.takeIf {
                        it.isFinite()
                    } ?: 0.0,
                blinkCount = frame.blinkCount,
                yawnsLast5Min =
                    frame.yawnsLast5Min,
                nodsLast2Min =
                    frame.nodsLast2Min,
                gazeAwayCount =
                    frame.gazeAwayCount,
                poseAwayCount =
                    frame.poseAwayCount,
                faceValid = frame.faceValid
            )
            repository.append(record)
            record
        }
    }

    private fun faceTransitionEvent(
        frame: FatigueHistoryFrame
    ): Pair<String, String>? {
        val confirmed = confirmedFaceValid

        if (confirmed == null) {
            confirmedFaceValid = frame.faceValid
            return null
        }

        if (frame.faceValid == confirmed) {
            candidateFaceValid = null
            candidateSinceMs = null
            return null
        }

        if (candidateFaceValid != frame.faceValid) {
            candidateFaceValid = frame.faceValid
            candidateSinceMs = frame.timestampMs
            return null
        }

        val candidateSince = candidateSinceMs
            ?: frame.timestampMs.also {
                candidateSinceMs = it
            }
        val elapsedMs =
            frame.timestampMs - candidateSince

        if (elapsedMs < 0L) {
            candidateSinceMs = frame.timestampMs
            return null
        }

        if (elapsedMs < faceTransitionConfirmMs) {
            return null
        }

        confirmedFaceValid = frame.faceValid
        candidateFaceValid = null
        candidateSinceMs = null

        return if (frame.faceValid) {
            "FACE_RECOVERED" to "face_recovered"
        } else {
            "FACE_LOST" to "face_lost"
        }
    }
}

object FatigueHistorySummarizer {
    fun summarize(
        records: List<FatigueHistoryRecord>,
        nowMs: Long,
        horizonMs: Long
    ): FatigueHistorySummary {
        require(nowMs >= 0L)
        require(horizonMs > 0L)

        val cutoff = nowMs - horizonMs
        val selected = records
            .filter {
                it.timestampMs in cutoff..nowMs
            }
            .sortedBy { it.timestampMs }
        val counts = selected
            .groupingBy { it.eventType }
            .eachCount()
        val maximumScore = selected
            .maxOfOrNull { it.score } ?: 0.0
        val latestState = selected
            .lastOrNull()
            ?.state ?: FatigueState.NORMAL
        val criticalTransitions = selected.count {
            it.eventType == "STATE_TRANSITION" &&
                it.state == FatigueState.CRITICAL
        }
        val mostFrequentReason = selected
            .flatMap { it.reasons }
            .groupingBy { it }
            .eachCount()
            .maxByOrNull { it.value }
            ?.key

        return FatigueHistorySummary(
            eventCount = selected.size,
            countsByType = counts,
            maximumScore = maximumScore,
            latestState = latestState,
            criticalTransitions =
                criticalTransitions,
            mostFrequentReason =
                mostFrequentReason
        )
    }
}

class SQLiteFatigueHistoryStore(
    context: Context
) : SQLiteOpenHelper(
    context.applicationContext,
    DATABASE_NAME,
    null,
    DATABASE_VERSION
), FatigueHistoryRepository, Closeable {

    override fun onCreate(database: SQLiteDatabase) {
        database.execSQL(
            """
            CREATE TABLE fatigue_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                detail TEXT NOT NULL,
                state TEXT NOT NULL,
                score REAL NOT NULL,
                reasons TEXT NOT NULL,
                closure_s REAL NOT NULL,
                perclos_ratio REAL NOT NULL,
                blink_count INTEGER NOT NULL,
                yawns_5m INTEGER NOT NULL,
                nods_2m INTEGER NOT NULL,
                gaze_away_count INTEGER NOT NULL,
                pose_away_count INTEGER NOT NULL,
                face_valid INTEGER NOT NULL
            )
            """.trimIndent()
        )
        database.execSQL(
            """
            CREATE INDEX fatigue_events_timestamp
            ON fatigue_events(timestamp_ms)
            """.trimIndent()
        )
    }

    override fun onUpgrade(
        database: SQLiteDatabase,
        oldVersion: Int,
        newVersion: Int
    ) {
        if (oldVersion < 1) {
            onCreate(database)
        }

        if (oldVersion < 2) {
            database.execSQL(
                """
                UPDATE fatigue_events
                SET event_type = 'ACKNOWLEDGEMENT'
                WHERE detail = 'critical_acknowledged'
                """.trimIndent()
            )
        }
    }

    @Synchronized
    override fun append(
        record: FatigueHistoryRecord
    ) {
        val values = ContentValues().apply {
            put("timestamp_ms", record.timestampMs)
            put("session_id", record.sessionId)
            put("subject_id", record.subjectId)
            put("event_type", record.eventType)
            put("detail", record.detail)
            put("state", record.state.name)
            put("score", record.score)
            put(
                "reasons",
                record.reasons.joinToString(",")
            )
            put("closure_s", record.closureS)
            put(
                "perclos_ratio",
                record.perclosRatio
            )
            put("blink_count", record.blinkCount)
            put(
                "yawns_5m",
                record.yawnsLast5Min
            )
            put("nods_2m", record.nodsLast2Min)
            put(
                "gaze_away_count",
                record.gazeAwayCount
            )
            put(
                "pose_away_count",
                record.poseAwayCount
            )
            put(
                "face_valid",
                if (record.faceValid) 1 else 0
            )
        }

        writableDatabase.insertOrThrow(
            TABLE_NAME,
            null,
            values
        )
    }

    @Synchronized
    override fun recent(
        limit: Int
    ): List<FatigueHistoryRecord> {
        require(limit > 0)

        val records =
            mutableListOf<FatigueHistoryRecord>()
        val cursor = readableDatabase.query(
            TABLE_NAME,
            null,
            null,
            null,
            null,
            null,
            "timestamp_ms DESC, id DESC",
            limit.toString()
        )

        cursor.use {
            while (it.moveToNext()) {
                records += FatigueHistoryRecord(
                    timestampMs = it.long(
                        "timestamp_ms"
                    ),
                    sessionId =
                        it.string("session_id"),
                    subjectId =
                        it.string("subject_id"),
                    eventType =
                        it.string("event_type"),
                    detail = it.string("detail"),
                    state = FatigueState.valueOf(
                        it.string("state")
                    ),
                    score = it.double("score"),
                    reasons = it.string("reasons")
                        .split(",")
                        .filter { value ->
                            value.isNotBlank()
                        },
                    closureS =
                        it.double("closure_s"),
                    perclosRatio =
                        it.double("perclos_ratio"),
                    blinkCount =
                        it.int("blink_count"),
                    yawnsLast5Min =
                        it.int("yawns_5m"),
                    nodsLast2Min =
                        it.int("nods_2m"),
                    gazeAwayCount =
                        it.int("gaze_away_count"),
                    poseAwayCount =
                        it.int("pose_away_count"),
                    faceValid =
                        it.int("face_valid") == 1
                )
            }
        }

        return records.reversed()
    }

    @Synchronized
    override fun count(): Long {
        readableDatabase.rawQuery(
            "SELECT COUNT(*) FROM $TABLE_NAME",
            null
        ).use { cursor ->
            return if (cursor.moveToFirst()) {
                cursor.getLong(0)
            } else {
                0L
            }
        }
    }

    private fun android.database.Cursor.index(
        column: String
    ): Int {
        return getColumnIndexOrThrow(column)
    }

    private fun android.database.Cursor.string(
        column: String
    ): String {
        return getString(index(column))
    }

    private fun android.database.Cursor.long(
        column: String
    ): Long {
        return getLong(index(column))
    }

    private fun android.database.Cursor.int(
        column: String
    ): Int {
        return getInt(index(column))
    }

    private fun android.database.Cursor.double(
        column: String
    ): Double {
        return getDouble(index(column))
    }

    companion object {
        private const val DATABASE_NAME =
            "fatigue_history.db"
        private const val DATABASE_VERSION = 2
        private const val TABLE_NAME =
            "fatigue_events"
    }
}
