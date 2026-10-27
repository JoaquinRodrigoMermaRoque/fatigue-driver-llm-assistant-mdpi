package com.google.mediapipe.examples.facelandmarker

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.io.Closeable
import java.util.Locale

enum class RecentActivityLevel(
    val wireValue: String,
    val displayLabel: String
) {
    NONE("none", "Ninguna"),
    LIGHT("light", "Ligera"),
    MODERATE("moderate", "Moderada"),
    INTENSE("intense", "Intensa");

    companion object {
        fun fromWireValue(value: String?): RecentActivityLevel? {
            return values().firstOrNull {
                it.wireValue == value
            }
        }
    }
}

data class DriverContextEntry(
    val timestampMs: Long,
    val sleepLastNightHours: Double?,
    val sleepAverage3dHours: Double?,
    val wakeTimestampMs: Long?,
    val subjectiveSleepiness: Int?,
    val workShiftHours: Double?,
    val nightShift: Boolean?,
    val physicalDemand: RecentActivityLevel?,
    val mentalDemand: RecentActivityLevel?
)

data class DriverContextSummary(
    val entryCount7d: Int,
    val latestSleepLastNightHours: Double?,
    val latestSleepAverage3dHours: Double?,
    val latestWakeTimestampMs: Long?,
    val latestSubjectiveSleepiness: Int?,
    val latestWorkShiftHours: Double?,
    val latestNightShift: Boolean?,
    val latestPhysicalDemand: RecentActivityLevel?,
    val latestMentalDemand: RecentActivityLevel?,
    val latestTimestampMs: Long?
) {
    fun toPromptFacts(
        sessionDurationMinutes: Long,
        nowMs: Long
    ): List<String> {
        val facts = mutableListOf<String>()
        facts += "session_min=$sessionDurationMinutes"
        facts += "user_context_entries_7d=$entryCount7d"

        latestSleepLastNightHours?.let {
            facts += "sleep_last_night_h=" +
                String.format(Locale.US, "%.1f", it)
        }
        latestSleepAverage3dHours?.let {
            facts += "sleep_avg_3d_h=" +
                String.format(Locale.US, "%.1f", it)
        }
        latestWakeTimestampMs?.let {
            val hoursAwake =
                ((nowMs - it).coerceAtLeast(0L) / 3_600_000.0)
            facts += "hours_awake=" +
                String.format(Locale.US, "%.1f", hoursAwake)
        }
        latestSubjectiveSleepiness?.let {
            facts += "sleepiness_1_9=$it"
        }
        latestWorkShiftHours?.let {
            facts += "work_shift_h=" +
                String.format(Locale.US, "%.1f", it)
        }
        latestNightShift?.let {
            facts += "night_shift=$it"
        }
        latestPhysicalDemand?.let {
            facts += "physical_demand=${it.wireValue}"
        }
        latestMentalDemand?.let {
            facts += "mental_demand=${it.wireValue}"
        }

        if (entryCount7d == 0) {
            facts += "user_context=none"
        }

        return facts
    }
}

class SQLiteDriverContextStore(
    context: Context
) : SQLiteOpenHelper(
    context.applicationContext,
    DATABASE_NAME,
    null,
    DATABASE_VERSION
), Closeable {

    override fun onCreate(database: SQLiteDatabase) {
        database.execSQL(
            """
            CREATE TABLE driver_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                sleep_last_night_hours REAL,
                sleep_average_3d_hours REAL,
                wake_timestamp_ms INTEGER,
                subjective_sleepiness INTEGER,
                work_shift_hours REAL,
                night_shift INTEGER,
                physical_demand TEXT,
                mental_demand TEXT
            )
            """.trimIndent()
        )
        database.execSQL(
            """
            CREATE INDEX driver_context_timestamp
            ON driver_context(timestamp_ms)
            """.trimIndent()
        )
    }

    override fun onUpgrade(
        database: SQLiteDatabase,
        oldVersion: Int,
        newVersion: Int
    ) {
        if (oldVersion < 2) {
            database.execSQL(
                "ALTER TABLE driver_context " +
                    "ADD COLUMN sleep_last_night_hours REAL"
            )
            database.execSQL(
                "ALTER TABLE driver_context " +
                    "ADD COLUMN sleep_average_3d_hours REAL"
            )
            database.execSQL(
                "ALTER TABLE driver_context " +
                    "ADD COLUMN wake_timestamp_ms INTEGER"
            )
            database.execSQL(
                "ALTER TABLE driver_context " +
                    "ADD COLUMN subjective_sleepiness INTEGER"
            )
            database.execSQL(
                "ALTER TABLE driver_context " +
                    "ADD COLUMN night_shift INTEGER"
            )
            database.execSQL(
                "ALTER TABLE driver_context " +
                    "ADD COLUMN physical_demand TEXT"
            )
            database.execSQL(
                "ALTER TABLE driver_context " +
                    "ADD COLUMN mental_demand TEXT"
            )

            database.execSQL(
                """
                UPDATE driver_context
                SET sleep_last_night_hours = sleep_hours
                WHERE sleep_last_night_hours IS NULL
                """.trimIndent()
            )
            database.execSQL(
                """
                UPDATE driver_context
                SET physical_demand = recent_activity
                WHERE physical_demand IS NULL
                """.trimIndent()
            )
        }
    }

    @Synchronized
    fun append(entry: DriverContextEntry) {
        require(entry.timestampMs > 0L)
        entry.sleepLastNightHours?.let {
            require(it.isFinite() && it in 0.0..24.0)
        }
        entry.sleepAverage3dHours?.let {
            require(it.isFinite() && it in 0.0..24.0)
        }
        entry.wakeTimestampMs?.let {
            require(it > 0L && it <= entry.timestampMs)
        }
        entry.subjectiveSleepiness?.let {
            require(it in 1..9)
        }
        entry.workShiftHours?.let {
            require(it.isFinite() && it in 0.0..24.0)
        }
        require(
            entry.sleepLastNightHours != null ||
                entry.sleepAverage3dHours != null ||
                entry.wakeTimestampMs != null ||
                entry.subjectiveSleepiness != null ||
                entry.workShiftHours != null ||
                entry.nightShift != null ||
                entry.physicalDemand != null ||
                entry.mentalDemand != null
        )

        val values = ContentValues().apply {
            put("timestamp_ms", entry.timestampMs)
            putNullableDouble(
                "sleep_last_night_hours",
                entry.sleepLastNightHours
            )
            putNullableDouble(
                "sleep_average_3d_hours",
                entry.sleepAverage3dHours
            )
            if (entry.wakeTimestampMs != null) {
                put(
                    "wake_timestamp_ms",
                    entry.wakeTimestampMs
                )
            } else {
                putNull("wake_timestamp_ms")
            }
            if (entry.subjectiveSleepiness != null) {
                put(
                    "subjective_sleepiness",
                    entry.subjectiveSleepiness
                )
            } else {
                putNull("subjective_sleepiness")
            }
            putNullableDouble(
                "work_shift_hours",
                entry.workShiftHours
            )
            if (entry.nightShift != null) {
                put(
                    "night_shift",
                    if (entry.nightShift) 1 else 0
                )
            } else {
                putNull("night_shift")
            }
            if (entry.physicalDemand != null) {
                put(
                    "physical_demand",
                    entry.physicalDemand.wireValue
                )
            } else {
                putNull("physical_demand")
            }
            if (entry.mentalDemand != null) {
                put(
                    "mental_demand",
                    entry.mentalDemand.wireValue
                )
            } else {
                putNull("mental_demand")
            }
        }

        writableDatabase.insertOrThrow(
            TABLE_NAME,
            null,
            values
        )
    }

    @Synchronized
    fun summarize(
        nowMs: Long,
        horizonMs: Long = DEFAULT_HORIZON_MS
    ): DriverContextSummary {
        require(nowMs > 0L)
        require(horizonMs > 0L)

        val entries = recent(
            sinceMs = nowMs - horizonMs,
            limit = 64
        )

        return DriverContextSummary(
            entryCount7d = entries.size,
            latestSleepLastNightHours =
                entries.lastOrNull {
                    it.sleepLastNightHours != null
                }?.sleepLastNightHours,
            latestSleepAverage3dHours =
                entries.lastOrNull {
                    it.sleepAverage3dHours != null
                }?.sleepAverage3dHours,
            latestWakeTimestampMs =
                entries.lastOrNull {
                    it.wakeTimestampMs != null
                }?.wakeTimestampMs,
            latestSubjectiveSleepiness =
                entries.lastOrNull {
                    it.subjectiveSleepiness != null
                }?.subjectiveSleepiness,
            latestWorkShiftHours =
                entries.lastOrNull {
                    it.workShiftHours != null
                }?.workShiftHours,
            latestNightShift =
                entries.lastOrNull {
                    it.nightShift != null
                }?.nightShift,
            latestPhysicalDemand =
                entries.lastOrNull {
                    it.physicalDemand != null
                }?.physicalDemand,
            latestMentalDemand =
                entries.lastOrNull {
                    it.mentalDemand != null
                }?.mentalDemand,
            latestTimestampMs =
                entries.lastOrNull()?.timestampMs
        )
    }

    @Synchronized
    fun latest(): DriverContextEntry? {
        return recent(
            sinceMs = 0L,
            limit = 1
        ).lastOrNull()
    }

    @Synchronized
    fun clear() {
        writableDatabase.delete(
            TABLE_NAME,
            null,
            null
        )
    }

    private fun ContentValues.putNullableDouble(
        key: String,
        value: Double?
    ) {
        if (value == null) {
            putNull(key)
        } else {
            put(key, value)
        }
    }

    private fun recent(
        sinceMs: Long,
        limit: Int
    ): List<DriverContextEntry> {
        val result = mutableListOf<DriverContextEntry>()

        readableDatabase.query(
            TABLE_NAME,
            null,
            "timestamp_ms >= ?",
            arrayOf(sinceMs.toString()),
            null,
            null,
            "timestamp_ms DESC, id DESC",
            limit.toString()
        ).use { cursor ->
            val timestampIndex =
                cursor.getColumnIndexOrThrow(
                    "timestamp_ms"
                )
            val sleepLastIndex =
                cursor.getColumnIndexOrThrow(
                    "sleep_last_night_hours"
                )
            val sleepAverageIndex =
                cursor.getColumnIndexOrThrow(
                    "sleep_average_3d_hours"
                )
            val wakeIndex =
                cursor.getColumnIndexOrThrow(
                    "wake_timestamp_ms"
                )
            val sleepinessIndex =
                cursor.getColumnIndexOrThrow(
                    "subjective_sleepiness"
                )
            val shiftIndex =
                cursor.getColumnIndexOrThrow(
                    "work_shift_hours"
                )
            val nightShiftIndex =
                cursor.getColumnIndexOrThrow(
                    "night_shift"
                )
            val physicalIndex =
                cursor.getColumnIndexOrThrow(
                    "physical_demand"
                )
            val mentalIndex =
                cursor.getColumnIndexOrThrow(
                    "mental_demand"
                )

            while (cursor.moveToNext()) {
                result += DriverContextEntry(
                    timestampMs =
                        cursor.getLong(timestampIndex),
                    sleepLastNightHours =
                        cursor.optionalDouble(
                            sleepLastIndex
                        ),
                    sleepAverage3dHours =
                        cursor.optionalDouble(
                            sleepAverageIndex
                        ),
                    wakeTimestampMs =
                        cursor.optionalLong(wakeIndex),
                    subjectiveSleepiness =
                        cursor.optionalInt(
                            sleepinessIndex
                        ),
                    workShiftHours =
                        cursor.optionalDouble(shiftIndex),
                    nightShift =
                        cursor.optionalBoolean(
                            nightShiftIndex
                        ),
                    physicalDemand =
                        cursor.optionalString(
                            physicalIndex
                        )?.let {
                            RecentActivityLevel
                                .fromWireValue(it)
                        },
                    mentalDemand =
                        cursor.optionalString(
                            mentalIndex
                        )?.let {
                            RecentActivityLevel
                                .fromWireValue(it)
                        }
                )
            }
        }

        return result.reversed()
    }

    private fun android.database.Cursor.optionalDouble(
        index: Int
    ): Double? {
        return if (isNull(index)) null else getDouble(index)
    }

    private fun android.database.Cursor.optionalLong(
        index: Int
    ): Long? {
        return if (isNull(index)) null else getLong(index)
    }

    private fun android.database.Cursor.optionalInt(
        index: Int
    ): Int? {
        return if (isNull(index)) null else getInt(index)
    }

    private fun android.database.Cursor.optionalBoolean(
        index: Int
    ): Boolean? {
        return if (isNull(index)) null else getInt(index) == 1
    }

    private fun android.database.Cursor.optionalString(
        index: Int
    ): String? {
        return if (isNull(index)) null else getString(index)
    }

    companion object {
        private const val DATABASE_NAME =
            "driver_context.db"
        private const val DATABASE_VERSION = 2
        private const val TABLE_NAME =
            "driver_context"
        private const val DEFAULT_HORIZON_MS =
            7L * 24L * 60L * 60L * 1000L
    }
}
