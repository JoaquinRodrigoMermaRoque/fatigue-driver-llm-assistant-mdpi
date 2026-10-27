/*
 * Temporal analysis of head pose for the Android prototype.
 */
package com.google.mediapipe.examples.facelandmarker

import kotlin.math.exp

data class HeadPoseTemporalMetrics(
    val pitchRelativeDeg: Double,
    val yawRelativeDeg: Double,
    val rollRelativeDeg: Double,
    val direction: String,
    val offCenter: Boolean,
    val awayDurationS: Double,
    val poseAwayCount: Int,
    val poseAwayEvent: Boolean,
    val nodActive: Boolean,
    val nodDurationS: Double,
    val nodCount: Int,
    val nodEvent: Boolean
)

class HeadPoseTemporalAnalyzer(
    val baselinePitchDeg: Double = 12.2,
    val baselineYawDeg: Double = -2.9,
    val baselineRollDeg: Double = 0.1,
    val yawThresholdDeg: Double = 20.0,
    val pitchThresholdDeg: Double = 15.0,
    val rollThresholdDeg: Double = 20.0,
    val minAwayS: Double = 1.50,
    val nodDownThresholdDeg: Double = 15.0,
    val nodReturnThresholdDeg: Double = 8.0,
    val minNodS: Double = 0.30,
    val maxNodS: Double = 2.50,
    val smoothingTauS: Double = 0.15
) {
    private var lastTimestampS: Double? = null
    private var filteredPitch: Double? = null
    private var filteredYaw: Double? = null
    private var filteredRoll: Double? = null
    private var awayStartS: Double? = null
    private var awayReported = false
    private var totalPoseAway = 0
    private var nodStartS: Double? = null
    private var totalNods = 0

    init {
        require(yawThresholdDeg > 0.0)
        require(pitchThresholdDeg > 0.0)
        require(rollThresholdDeg > 0.0)
        require(minAwayS > 0.0)
        require(
            nodReturnThresholdDeg >= 0.0 &&
                nodReturnThresholdDeg <
                nodDownThresholdDeg
        )
        require(
            minNodS > 0.0 &&
                minNodS < maxNodS
        )
        require(smoothingTauS > 0.0)
    }

    @Synchronized
    fun reset() {
        lastTimestampS = null
        filteredPitch = null
        filteredYaw = null
        filteredRoll = null
        awayStartS = null
        awayReported = false
        totalPoseAway = 0
        nodStartS = null
        totalNods = 0
    }

    @Synchronized
    fun update(
        timestampMs: Long,
        pitchDeg: Double,
        yawDeg: Double,
        rollDeg: Double,
        valid: Boolean
    ): HeadPoseTemporalMetrics {
        val timestampS = validateTimestamp(timestampMs)
        val validPose =
            valid &&
                pitchDeg.isFinite() &&
                yawDeg.isFinite() &&
                rollDeg.isFinite()

        if (!validPose) {
            filteredPitch = null
            filteredYaw = null
            filteredRoll = null
            awayStartS = null
            awayReported = false
            nodStartS = null
            lastTimestampS = timestampS
            return invalidMetrics()
        }

        val pitchRelative =
            pitchDeg - baselinePitchDeg
        val yawRelative =
            yawDeg - baselineYawDeg
        val rollRelative =
            rollDeg - baselineRollDeg

        if (
            filteredPitch == null ||
            filteredYaw == null ||
            filteredRoll == null ||
            lastTimestampS == null
        ) {
            filteredPitch = pitchRelative
            filteredYaw = yawRelative
            filteredRoll = rollRelative
        } else {
            val deltaS =
                timestampS - checkNotNull(lastTimestampS)
            val alpha =
                1.0 - exp(-deltaS / smoothingTauS)

            filteredPitch =
                checkNotNull(filteredPitch) +
                    alpha * (
                        pitchRelative -
                            checkNotNull(filteredPitch)
                        )
            filteredYaw =
                checkNotNull(filteredYaw) +
                    alpha * (
                        yawRelative -
                            checkNotNull(filteredYaw)
                        )
            filteredRoll =
                checkNotNull(filteredRoll) +
                    alpha * (
                        rollRelative -
                            checkNotNull(filteredRoll)
                        )
        }

        val pitch = checkNotNull(filteredPitch)
        val yaw = checkNotNull(filteredYaw)
        val roll = checkNotNull(filteredRoll)
        val direction =
            classifyDirection(pitch, yaw, roll)
        val offCenter = direction != "FRONTAL"

        var awayDurationS = 0.0
        var poseAwayEvent = false

        if (offCenter) {
            if (awayStartS == null) {
                awayStartS = timestampS
                awayReported = false
            }

            awayDurationS =
                timestampS - checkNotNull(awayStartS)

            if (
                awayDurationS >= minAwayS &&
                !awayReported
            ) {
                totalPoseAway += 1
                awayReported = true
                poseAwayEvent = true
            }
        } else {
            awayStartS = null
            awayReported = false
        }

        var nodEvent = false
        var nodActive = false
        var nodDurationS = 0.0

        if (nodStartS == null) {
            if (pitch >= nodDownThresholdDeg) {
                nodStartS = timestampS
                nodActive = true
            }
        } else {
            nodDurationS =
                timestampS - checkNotNull(nodStartS)
            nodActive = true

            if (pitch <= nodReturnThresholdDeg) {
                if (nodDurationS in minNodS..maxNodS) {
                    totalNods += 1
                    nodEvent = true
                }

                nodStartS = null
                nodActive = false
                nodDurationS = 0.0
            }
        }

        lastTimestampS = timestampS

        return HeadPoseTemporalMetrics(
            pitchRelativeDeg = pitch,
            yawRelativeDeg = yaw,
            rollRelativeDeg = roll,
            direction = direction,
            offCenter = offCenter,
            awayDurationS = awayDurationS,
            poseAwayCount = totalPoseAway,
            poseAwayEvent = poseAwayEvent,
            nodActive = nodActive,
            nodDurationS = nodDurationS,
            nodCount = totalNods,
            nodEvent = nodEvent
        )
    }

    private fun validateTimestamp(
        timestampMs: Long
    ): Double {
        require(timestampMs >= 0L) {
            "timestampMs debe ser no negativo."
        }

        val timestampS = timestampMs / 1000.0
        val previous = lastTimestampS

        require(previous == null || timestampS > previous) {
            "Los timestamps deben ser estrictamente crecientes."
        }

        return timestampS
    }

    private fun classifyDirection(
        pitch: Double,
        yaw: Double,
        roll: Double
    ): String {
        val directions = mutableListOf<String>()

        // La cámara frontal se analiza reflejada.
        when {
            yaw < -yawThresholdDeg ->
                directions += "IZQUIERDA"
            yaw > yawThresholdDeg ->
                directions += "DERECHA"
        }

        when {
            pitch > pitchThresholdDeg ->
                directions += "ABAJO"
            pitch < -pitchThresholdDeg ->
                directions += "ARRIBA"
        }

        when {
            roll > rollThresholdDeg ->
                directions += "INCLINADA_IZQUIERDA"
            roll < -rollThresholdDeg ->
                directions += "INCLINADA_DERECHA"
        }

        return if (directions.isEmpty()) {
            "FRONTAL"
        } else {
            directions.joinToString("_")
        }
    }

    private fun invalidMetrics():
        HeadPoseTemporalMetrics {
        return HeadPoseTemporalMetrics(
            pitchRelativeDeg = Double.NaN,
            yawRelativeDeg = Double.NaN,
            rollRelativeDeg = Double.NaN,
            direction = "NO_VALIDA",
            offCenter = false,
            awayDurationS = 0.0,
            poseAwayCount = totalPoseAway,
            poseAwayEvent = false,
            nodActive = false,
            nodDurationS = 0.0,
            nodCount = totalNods,
            nodEvent = false
        )
    }
}
