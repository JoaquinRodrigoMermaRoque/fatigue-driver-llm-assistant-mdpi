/*
 * Temporal fatigue indicators for the Android prototype.
 *
 * Ported from desktop/src/temporal.py so the desktop and mobile
 * implementations use the same event semantics.
 */
package com.google.mediapipe.examples.facelandmarker

import java.util.ArrayDeque
import kotlin.math.exp

data class EyeTemporalMetrics(
    val perclos: Double,
    val validRatio: Double,
    val closureDurationS: Double,
    val blinkCount: Int,
    val blinkEvent: Boolean,
    val windowDurationS: Double,
    val validDurationS: Double
)

private data class EyeSample(
    val timestampS: Double,
    val valid: Boolean,
    val closed: Boolean
)

class EyeTemporalAnalyzer(
    val windowS: Double = 30.0,
    val earThreshold: Double = 0.14,
    val minBlinkS: Double = 0.08,
    val maxBlinkS: Double = 0.80
) {
    private val samples = ArrayDeque<EyeSample>()
    private var lastTimestampS: Double? = null
    private var closureStartS: Double? = null
    private var previousClosed = false
    private var totalBlinks = 0

    init {
        require(windowS > 0.0) {
            "windowS debe ser mayor que cero."
        }
        require(earThreshold > 0.0) {
            "earThreshold debe ser mayor que cero."
        }
        require(minBlinkS >= 0.0) {
            "minBlinkS no puede ser negativo."
        }
        require(maxBlinkS > minBlinkS) {
            "maxBlinkS debe ser mayor que minBlinkS."
        }
    }

    @Synchronized
    fun reset() {
        samples.clear()
        lastTimestampS = null
        closureStartS = null
        previousClosed = false
        totalBlinks = 0
    }

    @Synchronized
    fun update(
        timestampMs: Long,
        ear: Double,
        valid: Boolean
    ): EyeTemporalMetrics {
        val timestampS = validateTimestamp(timestampMs)
        val validEar = valid && ear.isFinite()
        var closed = validEar && ear < earThreshold
        var blinkEvent = false

        if (validEar) {
            if (closed && !previousClosed) {
                closureStartS = timestampS
            } else if (!closed && previousClosed) {
                closureStartS?.let { start ->
                    val durationS = timestampS - start
                    if (durationS in minBlinkS..maxBlinkS) {
                        totalBlinks += 1
                        blinkEvent = true
                    }
                }
                closureStartS = null
            }
            previousClosed = closed
        } else {
            // La pérdida de rostro cancela el cierre en curso.
            closureStartS = null
            previousClosed = false
            closed = false
        }

        samples.addLast(
            EyeSample(
                timestampS = timestampS,
                valid = validEar,
                closed = closed
            )
        )
        lastTimestampS = timestampS
        discardOldSamples(timestampS)

        val window = calculateWindow(timestampS)
        val closureDurationS =
            if (validEar && closed) {
                closureStartS?.let { timestampS - it } ?: 0.0
            } else {
                0.0
            }

        return EyeTemporalMetrics(
            perclos = window.perclos,
            validRatio = window.validRatio,
            closureDurationS = closureDurationS,
            blinkCount = totalBlinks,
            blinkEvent = blinkEvent,
            windowDurationS = window.windowDurationS,
            validDurationS = window.validDurationS
        )
    }

    private fun validateTimestamp(timestampMs: Long): Double {
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

    private fun discardOldSamples(currentTimestampS: Double) {
        val cutoffS = currentTimestampS - windowS

        while (samples.size >= 2) {
            val iterator = samples.iterator()
            iterator.next()
            val second = iterator.next()

            if (second.timestampS <= cutoffS) {
                samples.removeFirst()
            } else {
                break
            }
        }
    }

    private fun calculateWindow(
        currentTimestampS: Double
    ): WindowMetrics {
        if (samples.size < 2) {
            val sample = samples.peekLast()
            return WindowMetrics(
                perclos = Double.NaN,
                validRatio = if (sample.valid) 1.0 else 0.0,
                windowDurationS = 0.0,
                validDurationS = 0.0
            )
        }

        val sampleList = samples.toList()
        val cutoffS = currentTimestampS - windowS
        val windowStartS = maxOf(
            cutoffS,
            sampleList.first().timestampS
        )
        val windowDurationS =
            currentTimestampS - windowStartS

        var validDurationS = 0.0
        var closedDurationS = 0.0

        sampleList.forEachIndexed { index, sample ->
            val segmentStartS = maxOf(
                sample.timestampS,
                windowStartS
            )
            val segmentEndS =
                if (index + 1 < sampleList.size) {
                    minOf(
                        sampleList[index + 1].timestampS,
                        currentTimestampS
                    )
                } else {
                    currentTimestampS
                }
            val segmentDurationS =
                segmentEndS - segmentStartS

            if (segmentDurationS > 0.0 && sample.valid) {
                validDurationS += segmentDurationS
                if (sample.closed) {
                    closedDurationS += segmentDurationS
                }
            }
        }

        val perclos =
            if (validDurationS > 0.0) {
                closedDurationS / validDurationS
            } else {
                Double.NaN
            }
        val validRatio =
            if (windowDurationS > 0.0) {
                validDurationS / windowDurationS
            } else {
                0.0
            }

        return WindowMetrics(
            perclos = perclos,
            validRatio = validRatio,
            windowDurationS = windowDurationS,
            validDurationS = validDurationS
        )
    }

    private data class WindowMetrics(
        val perclos: Double,
        val validRatio: Double,
        val windowDurationS: Double,
        val validDurationS: Double
    )
}

data class MouthTemporalMetrics(
    val mouthOpen: Boolean,
    val openingDurationS: Double,
    val yawnCount: Int,
    val yawnEvent: Boolean
)

class MouthTemporalAnalyzer(
    val marThreshold: Double = 0.27,
    val minYawnS: Double = 1.50
) {
    private var lastTimestampS: Double? = null
    private var openingStartS: Double? = null
    private var yawnReported = false
    private var totalYawns = 0

    init {
        require(marThreshold > 0.0) {
            "marThreshold debe ser mayor que cero."
        }
        require(minYawnS > 0.0) {
            "minYawnS debe ser mayor que cero."
        }
    }

    @Synchronized
    fun reset() {
        lastTimestampS = null
        openingStartS = null
        yawnReported = false
        totalYawns = 0
    }

    @Synchronized
    fun update(
        timestampMs: Long,
        mar: Double,
        valid: Boolean
    ): MouthTemporalMetrics {
        val timestampS = validateTimestamp(timestampMs)
        val validMar = valid && mar.isFinite()
        var mouthOpen =
            validMar && mar >= marThreshold
        var openingDurationS = 0.0
        var yawnEvent = false

        if (mouthOpen) {
            if (openingStartS == null) {
                openingStartS = timestampS
                yawnReported = false
            }

            openingDurationS =
                timestampS - checkNotNull(openingStartS)

            if (
                openingDurationS >= minYawnS &&
                !yawnReported
            ) {
                totalYawns += 1
                yawnReported = true
                yawnEvent = true
            }
        } else {
            openingStartS = null
            yawnReported = false
        }

        if (!validMar) {
            mouthOpen = false
            openingDurationS = 0.0
        }

        lastTimestampS = timestampS

        return MouthTemporalMetrics(
            mouthOpen = mouthOpen,
            openingDurationS = openingDurationS,
            yawnCount = totalYawns,
            yawnEvent = yawnEvent
        )
    }

    private fun validateTimestamp(timestampMs: Long): Double {
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
}

data class GazeTemporalMetrics(
    val horizontal: Double,
    val vertical: Double,
    val direction: String,
    val offCenter: Boolean,
    val awayDurationS: Double,
    val distractionActive: Boolean,
    val gazeAwayCount: Int,
    val gazeAwayEvent: Boolean
)

class GazeTemporalAnalyzer(
    val horizontalLow: Double = 0.37,
    val horizontalHigh: Double = 0.62,
    val verticalLow: Double = 0.38,
    val verticalHigh: Double = 0.42,
    val verticalEnabled: Boolean = false,
    val minAwayS: Double = 1.50,
    val smoothingTauS: Double = 0.15
) {
    private var lastTimestampS: Double? = null
    private var filteredHorizontal: Double? = null
    private var filteredVertical: Double? = null
    private var awayStartS: Double? = null
    private var awayReported = false
    private var totalGazeAway = 0

    init {
        require(horizontalLow < horizontalHigh) {
            "Los umbrales horizontales no son válidos."
        }
        require(verticalLow < verticalHigh) {
            "Los umbrales verticales no son válidos."
        }
        require(minAwayS > 0.0) {
            "minAwayS debe ser mayor que cero."
        }
        require(smoothingTauS > 0.0) {
            "smoothingTauS debe ser mayor que cero."
        }
    }

    @Synchronized
    fun reset() {
        lastTimestampS = null
        filteredHorizontal = null
        filteredVertical = null
        awayStartS = null
        awayReported = false
        totalGazeAway = 0
    }

    @Synchronized
    fun update(
        timestampMs: Long,
        horizontal: Double,
        vertical: Double,
        valid: Boolean
    ): GazeTemporalMetrics {
        val timestampS = validateTimestamp(timestampMs)
        val validGaze =
            valid &&
                horizontal.isFinite() &&
                vertical.isFinite()

        if (!validGaze) {
            filteredHorizontal = null
            filteredVertical = null
            awayStartS = null
            awayReported = false
            lastTimestampS = timestampS

            return invalidMetrics()
        }

        if (
            filteredHorizontal == null ||
            filteredVertical == null ||
            lastTimestampS == null
        ) {
            filteredHorizontal = horizontal
            filteredVertical = vertical
        } else {
            val deltaS =
                timestampS - checkNotNull(lastTimestampS)
            val alpha =
                1.0 - exp(-deltaS / smoothingTauS)

            filteredHorizontal =
                checkNotNull(filteredHorizontal) +
                    alpha * (
                        horizontal -
                            checkNotNull(filteredHorizontal)
                        )
            filteredVertical =
                checkNotNull(filteredVertical) +
                    alpha * (
                        vertical -
                            checkNotNull(filteredVertical)
                        )
        }

        val filteredH = checkNotNull(filteredHorizontal)
        val filteredV = checkNotNull(filteredVertical)
        val direction = classifyDirection(
            filteredH,
            filteredV
        )
        val offCenter = direction != "CENTRO"
        var awayDurationS = 0.0
        var distractionActive = false
        var gazeAwayEvent = false

        if (offCenter) {
            if (awayStartS == null) {
                awayStartS = timestampS
                awayReported = false
            }

            awayDurationS =
                timestampS - checkNotNull(awayStartS)
            distractionActive =
                awayDurationS >= minAwayS

            if (distractionActive && !awayReported) {
                totalGazeAway += 1
                awayReported = true
                gazeAwayEvent = true
            }
        } else {
            awayStartS = null
            awayReported = false
        }

        lastTimestampS = timestampS

        return GazeTemporalMetrics(
            horizontal = filteredH,
            vertical = filteredV,
            direction = direction,
            offCenter = offCenter,
            awayDurationS = awayDurationS,
            distractionActive = distractionActive,
            gazeAwayCount = totalGazeAway,
            gazeAwayEvent = gazeAwayEvent
        )
    }

    private fun validateTimestamp(timestampMs: Long): Double {
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
        horizontal: Double,
        vertical: Double
    ): String {
        // La entrada frontal se refleja antes de MediaPipe.
        val horizontalDirection =
            when {
                horizontal < horizontalLow -> "IZQUIERDA"
                horizontal > horizontalHigh -> "DERECHA"
                else -> ""
            }

        if (!verticalEnabled) {
            return horizontalDirection.ifEmpty {
                "CENTRO"
            }
        }

        val verticalDirection =
            when {
                vertical < verticalLow -> "ARRIBA"
                vertical > verticalHigh -> "ABAJO"
                else -> ""
            }

        return when {
            horizontalDirection.isNotEmpty() &&
                verticalDirection.isNotEmpty() ->
                horizontalDirection + "_" +
                    verticalDirection
            horizontalDirection.isNotEmpty() ->
                horizontalDirection
            verticalDirection.isNotEmpty() ->
                verticalDirection
            else -> "CENTRO"
        }
    }

    private fun invalidMetrics(): GazeTemporalMetrics {
        return GazeTemporalMetrics(
            horizontal = Double.NaN,
            vertical = Double.NaN,
            direction = "NO_VALIDA",
            offCenter = false,
            awayDurationS = 0.0,
            distractionActive = false,
            gazeAwayCount = totalGazeAway,
            gazeAwayEvent = false
        )
    }
}
