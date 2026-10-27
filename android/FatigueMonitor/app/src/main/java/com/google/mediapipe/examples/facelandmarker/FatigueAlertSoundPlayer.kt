package com.google.mediapipe.examples.facelandmarker

import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Handler
import android.os.Looper
import android.util.Log

class FatigueAlertSoundPlayer {
    companion object {
        private const val TAG = "FatigueAlertSound"
        const val MODERATE_DELAY_MS = 650L
        const val CRITICAL_DELAY_MS = 1_250L
    }

    private val mainHandler =
        Handler(Looper.getMainLooper())

    private var moderateTone: ToneGenerator? =
        runCatching {
            ToneGenerator(
                AudioManager.STREAM_ALARM,
                80
            )
        }.getOrNull()

    private var criticalTone: ToneGenerator? =
        runCatching {
            ToneGenerator(
                AudioManager.STREAM_ALARM,
                100
            )
        }.getOrNull()

    fun playModerate() {
        mainHandler.post {
            val started = moderateTone?.startTone(
                ToneGenerator.TONE_PROP_BEEP2,
                MODERATE_DELAY_MS.toInt()
            ) == true

            Log.i(
                TAG,
                "MODERATE started=$started"
            )
        }
    }

    fun playCritical() {
        mainHandler.post {
            val started = criticalTone?.startTone(
                ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD,
                CRITICAL_DELAY_MS.toInt()
            ) == true

            Log.i(
                TAG,
                "CRITICAL started=$started"
            )
        }
    }

    fun stop() {
        mainHandler.post {
            moderateTone?.stopTone()
            criticalTone?.stopTone()
        }
    }

    fun release() {
        mainHandler.post {
            moderateTone?.release()
            criticalTone?.release()
            moderateTone = null
            criticalTone = null
        }
    }
}
