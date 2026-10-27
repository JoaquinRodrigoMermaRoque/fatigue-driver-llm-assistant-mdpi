package com.google.mediapipe.examples.facelandmarker

import android.os.SystemClock
import android.util.Log
import java.util.Locale

/**
 * Lightweight, non-overlapping telemetry windows for publication-oriented
 * MediaPipe performance measurements.
 *
 * Logs only one summary every [windowSize] callbacks to avoid distorting the
 * camera pipeline with per-frame logging.
 */
class MediaPipePerformanceTelemetry(
    private val windowSize: Int = 30
) {
    companion object {
        const val TAG = "H7_9_PERF"
    }

    private data class Sample(
        val endToEndMs: Long,
        val preprocessingMs: Long,
        val mediaPipeMs: Long,
        val callbackAtMs: Long
    )

    private val samples = ArrayList<Sample>(windowSize)
    private var activeDelegate: Int? = null
    private var windowIndex = 0L

    init {
        require(windowSize >= 5)
    }

    @Synchronized
    fun reset(delegate: Int? = null) {
        samples.clear()
        activeDelegate = delegate
        windowIndex = 0L

        Log.i(
            TAG,
            "RESET delegate=" +
                FaceLandmarkerHelper.delegateName(delegate)
        )
    }

    @Synchronized
    fun record(
        delegate: Int?,
        endToEndMs: Long,
        preprocessingMs: Long,
        mediaPipeMs: Long,
        callbackAtMs: Long = SystemClock.elapsedRealtime()
    ) {
        if (delegate != activeDelegate) {
            reset(delegate)
        }

        samples += Sample(
            endToEndMs = endToEndMs.coerceAtLeast(0L),
            preprocessingMs =
                preprocessingMs.coerceAtLeast(0L),
            mediaPipeMs = mediaPipeMs.coerceAtLeast(0L),
            callbackAtMs = callbackAtMs
        )

        if (samples.size >= windowSize) {
            logWindow()
            samples.clear()
        }
    }

    private fun logWindow() {
        if (samples.isEmpty()) {
            return
        }

        windowIndex += 1L

        val endToEnd = samples.map { it.endToEndMs }
        val preprocessing =
            samples.map { it.preprocessingMs }
        val mediaPipe = samples.map { it.mediaPipeMs }

        val firstCallback = samples.first().callbackAtMs
        val lastCallback = samples.last().callbackAtMs
        val callbackDurationMs =
            (lastCallback - firstCallback)
                .coerceAtLeast(0L)
        val callbackFps =
            if (samples.size > 1 && callbackDurationMs > 0L) {
                (samples.size - 1) * 1000.0 /
                    callbackDurationMs.toDouble()
            } else {
                0.0
            }

        Log.i(
            TAG,
            String.format(
                Locale.US,
                "delegate=%s,window=%d,samples=%d," +
                    "window_start_ms=%d,window_end_ms=%d," +
                    "e2e_mean_ms=%.2f,e2e_p50_ms=%d," +
                    "e2e_p95_ms=%d,e2e_max_ms=%d," +
                    "pre_mean_ms=%.2f,pre_p50_ms=%d," +
                    "pre_p95_ms=%d,pre_max_ms=%d," +
                    "mp_mean_ms=%.2f,mp_p50_ms=%d," +
                    "mp_p95_ms=%d,mp_max_ms=%d," +
                    "callback_fps=%.2f",
                FaceLandmarkerHelper.delegateName(
                    activeDelegate
                ),
                windowIndex,
                samples.size,
                firstCallback,
                lastCallback,
                mean(endToEnd),
                percentile(endToEnd, 0.50),
                percentile(endToEnd, 0.95),
                endToEnd.maxOrNull() ?: 0L,
                mean(preprocessing),
                percentile(preprocessing, 0.50),
                percentile(preprocessing, 0.95),
                preprocessing.maxOrNull() ?: 0L,
                mean(mediaPipe),
                percentile(mediaPipe, 0.50),
                percentile(mediaPipe, 0.95),
                mediaPipe.maxOrNull() ?: 0L,
                callbackFps
            )
        )
    }

    private fun mean(values: List<Long>): Double {
        if (values.isEmpty()) {
            return 0.0
        }

        return values.sumOf { it.toDouble() } /
            values.size.toDouble()
    }

    private fun percentile(
        values: List<Long>,
        probability: Double
    ): Long {
        if (values.isEmpty()) {
            return 0L
        }

        val sorted = values.sorted()
        val index = kotlin.math.ceil(
            probability * sorted.size
        ).toInt()
            .coerceIn(1, sorted.size) - 1

        return sorted[index]
    }
}
