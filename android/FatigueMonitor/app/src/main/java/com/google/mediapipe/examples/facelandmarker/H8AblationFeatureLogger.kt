package com.google.mediapipe.examples.facelandmarker

import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.util.Locale
import java.util.UUID

data class H8AblationFrame(
    val timestampMs: Long,
    val source: CalibrationCameraSource,
    val faceDetected: Boolean,
    val validEye: Boolean,
    val validMouth: Boolean,
    val validGaze: Boolean,
    val validPose: Boolean,
    val ear: Double,
    val mar: Double,
    val gazeHRaw: Double,
    val gazeVRaw: Double,
    val gazeHAsymmetry: Double,
    val gazeVAsymmetry: Double,
    val pitchDeg: Double,
    val yawDeg: Double,
    val rollDeg: Double,
    val calibrationProfile: DriverCalibrationProfile?,
    val ablationMode: AblationMode,
    val score: Double,
    val state: FatigueState,
    val stateEvent: String?,
    val closureS: Double,
    val perclos: Double,
    val yawnEvent: Boolean,
    val gazeAwayS: Double,
    val nodEvent: Boolean
)

class H8AblationFeatureLogger(
    outputDir: File,
    private val subjectId: String,
    private val clipId: String,
    requestedRunId: String? = null,
    private val flushEvery: Int = 30
) {
    companion object {
        const val SCHEMA_VERSION = 2

        val HEADER = listOf(
            "schema_version",
            "feature_run_id",
            "subject_id",
            "clip_id",
            "source",
            "frame_index",
            "t_ms",
            "face_detected",
            "valid_eye",
            "valid_mouth",
            "valid_gaze",
            "valid_pose",
            "ear",
            "mar",
            "gaze_h_raw",
            "gaze_v_raw",
            "gaze_h_asymmetry",
            "gaze_v_asymmetry",
            "pitch_deg",
            "yaw_deg",
            "roll_deg",
            "calibration_created_at_ms",
            "calibration_samples",
            "calibration_yaw_base_deg",
            "calibration_pitch_base_deg",
            "calibration_roll_base_deg",
            "calibration_gaze_h",
            "calibration_gaze_v",
            "calibration_ear_open",
            "calibration_mar_rest",
            "calibration_scale",
            "ablation_mode",
            "score",
            "state",
            "state_event",
            "closure_s",
            "perclos",
            "yawn_event",
            "gaze_away_s",
            "nod_event"
        )

        private fun sanitize(value: String): String =
            value.trim()
                .replace(Regex("[^A-Za-z0-9._-]+"), "_")
                .trim('_')
                .take(80)
                .ifBlank { "run" }
    }

    val featureRunId: String =
        requestedRunId
            ?.takeIf { it.isNotBlank() }
            ?.let(::sanitize)
            ?: "H8_${System.currentTimeMillis()}_${UUID.randomUUID().toString().take(8)}"

    val file: File

    private val writer: BufferedWriter
    private var firstTimestampMs: Long? = null
    private var frameIndex = 0L
    private var closed = false

    init {
        require(flushEvery >= 1)
        outputDir.mkdirs()
        require(outputDir.isDirectory) {
            "No se pudo crear el directorio experimental: $outputDir"
        }

        file = File(
            outputDir,
            "h8_features_${featureRunId}.csv"
        )
        require(!file.exists()) {
            "El archivo experimental ya existe: ${file.absolutePath}"
        }

        writer = BufferedWriter(FileWriter(file, false))
        writer.write(HEADER.joinToString(","))
        writer.newLine()
        writer.flush()
    }

    @Synchronized
    fun write(frame: H8AblationFrame) {
        check(!closed) {
            "El logger H8 ya fue cerrado"
        }
        require(frame.timestampMs >= 0L)

        val start = firstTimestampMs ?: frame.timestampMs.also {
            firstTimestampMs = it
        }

        val profile = frame.calibrationProfile
        val values = listOf(
            SCHEMA_VERSION,
            featureRunId,
            subjectId,
            clipId,
            frame.source.name,
            frameIndex,
            frame.timestampMs - start,
            bool(frame.faceDetected),
            bool(frame.validEye),
            bool(frame.validMouth),
            bool(frame.validGaze),
            bool(frame.validPose),
            number(frame.ear),
            number(frame.mar),
            number(frame.gazeHRaw),
            number(frame.gazeVRaw),
            number(frame.gazeHAsymmetry),
            number(frame.gazeVAsymmetry),
            number(frame.pitchDeg),
            number(frame.yawDeg),
            number(frame.rollDeg),
            profile?.createdAtEpochMs ?: "",
            profile?.sampleCount ?: "",
            number(profile?.yawBaseDeg),
            number(profile?.pitchBaseDeg),
            number(profile?.rollBaseDeg),
            number(profile?.gazeCenterH),
            number(profile?.gazeCenterV),
            number(profile?.earOpen),
            number(profile?.marRest),
            number(profile?.faceScaleBase),
            frame.ablationMode.code,
            number(frame.score),
            frame.state.name,
            frame.stateEvent ?: "",
            number(frame.closureS),
            number(frame.perclos),
            bool(frame.yawnEvent),
            number(frame.gazeAwayS),
            bool(frame.nodEvent)
        )

        writer.write(
            values.joinToString(",") {
                csvCell(it.toString())
            }
        )
        writer.newLine()

        frameIndex += 1
        if (frameIndex % flushEvery == 0L) {
            writer.flush()
        }
    }

    @Synchronized
    fun close() {
        if (closed) {
            return
        }

        writer.flush()
        writer.close()
        closed = true
    }

    fun rowsWritten(): Long = frameIndex

    private fun bool(value: Boolean): Int =
        if (value) 1 else 0

    private fun number(value: Double?): String {
        if (value == null || !value.isFinite()) {
            return ""
        }
        return String.format(Locale.US, "%.8f", value)
    }

    private fun csvCell(value: String): String {
        val escaped = value.replace("\"", "\"\"")
        return if (
            escaped.contains(',') ||
            escaped.contains('"') ||
            escaped.contains('\n') ||
            escaped.contains('\r')
        ) {
            "\"$escaped\""
        } else {
            escaped
        }
    }
}
