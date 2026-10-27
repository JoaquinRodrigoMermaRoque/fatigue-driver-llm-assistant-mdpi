package com.google.mediapipe.examples.facelandmarker

import android.content.Context
import android.util.Log
import java.util.Locale

enum class CalibrationCameraSource {
    INTERNAL,
    USB
}

data class DriverCalibrationProfile(
    val source: CalibrationCameraSource,
    val createdAtEpochMs: Long,
    val yawBaseDeg: Double,
    val pitchBaseDeg: Double,
    val rollBaseDeg: Double,
    val gazeCenterH: Double,
    val gazeCenterV: Double,
    val earOpen: Double,
    val marRest: Double,
    val faceScaleBase: Double,
    val sampleCount: Int
)

class DriverCalibrationStore(
    context: Context
) {
    companion object {
        private const val TAG = "H7_14_CAL"
        private const val PREFS = "driver_calibration_capture_v1"
    }

    private val prefs =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun save(profile: DriverCalibrationProfile) {
        val p = profile.source.name.lowercase(Locale.US)

        prefs.edit()
            .putLong("${p}_created_at", profile.createdAtEpochMs)
            .putString("${p}_yaw_base", profile.yawBaseDeg.toString())
            .putString("${p}_pitch_base", profile.pitchBaseDeg.toString())
            .putString("${p}_roll_base", profile.rollBaseDeg.toString())
            .putString("${p}_gaze_h", profile.gazeCenterH.toString())
            .putString("${p}_gaze_v", profile.gazeCenterV.toString())
            .putString("${p}_ear_open", profile.earOpen.toString())
            .putString("${p}_mar_rest", profile.marRest.toString())
            .putString("${p}_face_scale", profile.faceScaleBase.toString())
            .putInt("${p}_samples", profile.sampleCount)
            .apply()

        Log.i(
            TAG,
            String.format(
                Locale.US,
                "SAVED source=%s samples=%d yaw=%.2f pitch=%.2f roll=%.2f gaze=%.3f/%.3f ear=%.4f mar=%.4f scale=%.4f",
                profile.source.name,
                profile.sampleCount,
                profile.yawBaseDeg,
                profile.pitchBaseDeg,
                profile.rollBaseDeg,
                profile.gazeCenterH,
                profile.gazeCenterV,
                profile.earOpen,
                profile.marRest,
                profile.faceScaleBase
            )
        )
    }

    fun load(source: CalibrationCameraSource): DriverCalibrationProfile? {
        val p = source.name.lowercase(Locale.US)
        if (!prefs.contains("${p}_created_at")) {
            return null
        }

        fun value(name: String): Double? =
            prefs.getString("${p}_${name}", null)?.toDoubleOrNull()

        val yaw = value("yaw_base") ?: return null
        val pitch = value("pitch_base") ?: return null
        val roll = value("roll_base") ?: return null
        val gazeH = value("gaze_h") ?: return null
        val gazeV = value("gaze_v") ?: return null
        val ear = value("ear_open") ?: return null
        val mar = value("mar_rest") ?: return null
        val scale = value("face_scale") ?: return null

        return DriverCalibrationProfile(
            source = source,
            createdAtEpochMs = prefs.getLong("${p}_created_at", 0L),
            yawBaseDeg = yaw,
            pitchBaseDeg = pitch,
            rollBaseDeg = roll,
            gazeCenterH = gazeH,
            gazeCenterV = gazeV,
            earOpen = ear,
            marRest = mar,
            faceScaleBase = scale,
            sampleCount = prefs.getInt("${p}_samples", 0)
        )
    }
}

class NeutralCalibrationSession(
    val source: CalibrationCameraSource,
    private val durationMs: Long = 3_000L,
    private val minSamples: Int = 20
) {
    companion object {
        private const val MAX_GAZE_ASYMMETRY = 0.20
    }

    private data class Sample(
        val timestampMs: Long,
        val yaw: Double,
        val pitch: Double,
        val roll: Double,
        val gazeH: Double,
        val gazeV: Double,
        val ear: Double,
        val mar: Double,
        val scale: Double
    )

    private val samples = mutableListOf<Sample>()
    private var firstTimestampMs: Long? = null

    fun sampleCount(): Int = samples.size

    fun add(
        timestampMs: Long,
        cues: FatigueCueMetrics?,
        pose: HeadPose?
    ): DriverCalibrationProfile? {
        val validCues = cues?.takeIf {
            it.valid &&
                it.ear.isFinite() &&
                it.mar.isFinite() &&
                it.gaze.horizontal.isFinite() &&
                it.gaze.vertical.isFinite() &&
                it.gaze.horizontalAsymmetry.isFinite() &&
                it.gaze.horizontalAsymmetry <= MAX_GAZE_ASYMMETRY
        } ?: return null

        val validPose = pose?.takeIf {
            it.yawDeg.isFinite() &&
                it.pitchDeg.isFinite() &&
                it.rollDeg.isFinite() &&
                it.scale.isFinite() &&
                it.fitError.isFinite() &&
                it.fitError <= 0.10
        } ?: return null

        if (firstTimestampMs == null) {
            firstTimestampMs = timestampMs
        }

        samples += Sample(
            timestampMs = timestampMs,
            yaw = validPose.yawDeg,
            pitch = validPose.pitchDeg,
            roll = validPose.rollDeg,
            gazeH = validCues.gaze.horizontal,
            gazeV = validCues.gaze.vertical,
            ear = validCues.ear,
            mar = validCues.mar,
            scale = validPose.scale
        )

        val elapsedMs = timestampMs - checkNotNull(firstTimestampMs)
        if (elapsedMs < durationMs || samples.size < minSamples) {
            return null
        }

        return DriverCalibrationProfile(
            source = source,
            createdAtEpochMs = System.currentTimeMillis(),
            yawBaseDeg = median(samples.map { it.yaw }),
            pitchBaseDeg = median(samples.map { it.pitch }),
            rollBaseDeg = median(samples.map { it.roll }),
            gazeCenterH = median(samples.map { it.gazeH }),
            gazeCenterV = median(samples.map { it.gazeV }),
            earOpen = median(samples.map { it.ear }),
            marRest = median(samples.map { it.mar }),
            faceScaleBase = median(samples.map { it.scale }),
            sampleCount = samples.size
        )
    }

    private fun median(values: List<Double>): Double {
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        }
    }
}
