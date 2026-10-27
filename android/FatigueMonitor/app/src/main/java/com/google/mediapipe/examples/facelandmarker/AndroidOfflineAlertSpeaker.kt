/*
 * Android offline text-to-speech adapter for contextual safety alerts.
 */
package com.google.mediapipe.examples.facelandmarker

import android.content.Context
import android.media.AudioAttributes
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.ArrayDeque
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong

data class OfflineVoiceOption(
    val name: String,
    val language: String,
    val country: String,
    val quality: Int,
    val latency: Int,
    val networkRequired: Boolean
)

object OfflineVoiceSelector {
    private val countryPriority = mapOf(
        "PE" to 4,
        "MX" to 3,
        "US" to 2,
        "ES" to 1
    )

    fun select(
        options: Collection<OfflineVoiceOption>
    ): OfflineVoiceOption? {
        return options
            .asSequence()
            .filter {
                !it.networkRequired &&
                    it.language.equals(
                        "es",
                        ignoreCase = true
                    )
            }
            .sortedWith(
                compareByDescending<OfflineVoiceOption> {
                    countryPriority[
                        it.country.uppercase()
                    ] ?: 0
                }.thenByDescending {
                    it.quality
                }.thenBy {
                    it.latency
                }.thenBy {
                    it.name
                }
            )
            .firstOrNull()
    }
}

enum class OfflineSpeechState {
    INITIALIZING,
    READY,
    UNAVAILABLE,
    CLOSED
}

private data class PendingSpeech(
    val text: String,
    val interrupt: Boolean
)

class AndroidOfflineAlertSpeaker(
    context: Context
) : TextToSpeech.OnInitListener {

    companion object {
        private const val TAG = "OfflineVoice"
        private const val MAX_PENDING = 4
    }

    private val applicationContext =
        context.applicationContext
    private val mainHandler =
        Handler(Looper.getMainLooper())
    private val utteranceCounter =
        AtomicLong(0L)
    private val pending =
        ArrayDeque<PendingSpeech>()

    @Volatile
    var state = OfflineSpeechState.INITIALIZING
        private set

    @Volatile
    var selectedVoiceName: String? = null
        private set

    @Volatile
    var completedCount = 0
        private set

    @Volatile
    var failedCount = 0
        private set

    @Volatile
    var cancelledCount = 0
        private set

    private var engine: TextToSpeech? = null

    init {
        mainHandler.post {
            synchronized(this) {
                if (
                    state ==
                    OfflineSpeechState.INITIALIZING
                ) {
                    engine = TextToSpeech(
                        applicationContext,
                        this
                    )
                }
            }
        }
    }

    @Synchronized
    override fun onInit(status: Int) {
        if (
            state == OfflineSpeechState.CLOSED
        ) {
            engine?.shutdown()
            engine = null
            return
        }

        if (status != TextToSpeech.SUCCESS) {
            markUnavailable(
                "No se pudo inicializar Android TTS."
            )
            return
        }

        val currentEngine = engine
        if (currentEngine == null) {
            markUnavailable(
                "Android TTS no entregó un motor."
            )
            return
        }

        val voices = currentEngine.voices
            .orEmpty()
        val selectedOption =
            OfflineVoiceSelector.select(
                voices.map { voice ->
                    OfflineVoiceOption(
                        name = voice.name,
                        language =
                            voice.locale.language,
                        country =
                            voice.locale.country,
                        quality = voice.quality,
                        latency = voice.latency,
                        networkRequired =
                            voice
                                .isNetworkConnectionRequired
                    )
                }
            )

        if (selectedOption == null) {
            markUnavailable(
                "No hay una voz española offline instalada."
            )
            return
        }

        val selectedVoice = voices.firstOrNull {
            it.name == selectedOption.name
        }

        if (
            selectedVoice == null ||
            currentEngine.setVoice(
                selectedVoice
            ) == TextToSpeech.ERROR
        ) {
            markUnavailable(
                "No se pudo seleccionar la voz offline."
            )
            return
        }

        currentEngine.setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(
                    AudioAttributes
                        .USAGE_ASSISTANCE_NAVIGATION_GUIDANCE
                )
                .setContentType(
                    AudioAttributes.CONTENT_TYPE_SPEECH
                )
                .build()
        )
        currentEngine.setSpeechRate(1.05f)
        currentEngine.setPitch(1.0f)
        currentEngine.setOnUtteranceProgressListener(
            progressListener()
        )

        selectedVoiceName = selectedVoice.name
        state = OfflineSpeechState.READY

        Log.i(
            TAG,
            "Voz offline lista: " +
                selectedVoice.name +
                " | locale=" +
                selectedVoice.locale.toLanguageTag()
        )

        val queued = pending.toList()
        pending.clear()
        queued.forEach {
            dispatch(it.text, it.interrupt)
        }
    }

    @Synchronized
    fun speak(
        text: String,
        interrupt: Boolean
    ): Boolean {
        val normalized = text.trim()
        if (normalized.isEmpty()) {
            return false
        }

        return when (state) {
            OfflineSpeechState.INITIALIZING -> {
                enqueue(normalized, interrupt)
                true
            }
            OfflineSpeechState.READY -> {
                dispatch(normalized, interrupt)
                true
            }
            OfflineSpeechState.UNAVAILABLE,
            OfflineSpeechState.CLOSED -> false
        }
    }

    @Synchronized
    fun stop() {
        if (pending.isNotEmpty()) {
            cancelledCount += pending.size
            pending.clear()
        }

        mainHandler.post {
            if (
                engine?.isSpeaking == true
            ) {
                cancelledCount += 1
            }
            engine?.stop()
        }
    }

    @Synchronized
    fun shutdown() {
        if (
            state == OfflineSpeechState.CLOSED
        ) {
            return
        }

        pending.clear()
        state = OfflineSpeechState.CLOSED
        selectedVoiceName = null

        mainHandler.post {
            engine?.stop()
            engine?.shutdown()
            engine = null
            Log.i(TAG, "Emisor offline cerrado.")
        }
    }

    private fun enqueue(
        text: String,
        interrupt: Boolean
    ) {
        if (interrupt) {
            cancelledCount += pending.size
            pending.clear()
        }

        while (pending.size >= MAX_PENDING) {
            pending.removeFirst()
            cancelledCount += 1
        }

        pending.addLast(
            PendingSpeech(text, interrupt)
        )
    }

    private fun dispatch(
        text: String,
        interrupt: Boolean
    ) {
        val utteranceId =
            "fatigue-alert-" +
                utteranceCounter.incrementAndGet()
        val queueMode = if (interrupt) {
            TextToSpeech.QUEUE_FLUSH
        } else {
            TextToSpeech.QUEUE_ADD
        }

        mainHandler.post {
            val currentEngine = engine
            if (
                state != OfflineSpeechState.READY ||
                currentEngine == null
            ) {
                failedCount += 1
                Log.e(
                    TAG,
                    "Alerta descartada: voz no disponible."
                )
                return@post
            }

            val parameters = Bundle().apply {
                putFloat(
                    TextToSpeech.Engine
                        .KEY_PARAM_VOLUME,
                    1.0f
                )
            }
            val result = currentEngine.speak(
                text,
                queueMode,
                parameters,
                utteranceId
            )

            if (result == TextToSpeech.ERROR) {
                failedCount += 1
                Log.e(
                    TAG,
                    "Android TTS rechazó $utteranceId"
                )
            } else {
                Log.i(
                    TAG,
                    "Encolado $utteranceId" +
                        " | interrumpe=$interrupt"
                )
            }
        }
    }

    private fun progressListener():
        UtteranceProgressListener {
        return object :
            UtteranceProgressListener() {
            override fun onStart(
                utteranceId: String
            ) {
                Log.i(
                    TAG,
                    "Inicio $utteranceId"
                )
            }

            override fun onDone(
                utteranceId: String
            ) {
                completedCount += 1
                Log.i(
                    TAG,
                    "Completado $utteranceId"
                )
            }

            @Deprecated(
                "Use onError(String, Int)"
            )
            override fun onError(
                utteranceId: String
            ) {
                failedCount += 1
                Log.e(
                    TAG,
                    "Error $utteranceId"
                )
            }

            override fun onError(
                utteranceId: String,
                errorCode: Int
            ) {
                failedCount += 1
                Log.e(
                    TAG,
                    "Error $utteranceId" +
                        " | código=$errorCode"
                )
            }

            override fun onStop(
                utteranceId: String,
                interrupted: Boolean
            ) {
                if (interrupted) {
                    cancelledCount += 1
                }
                Log.i(
                    TAG,
                    "Detenido $utteranceId" +
                        " | interrumpido=$interrupted"
                )
            }
        }
    }

    private fun markUnavailable(
        reason: String
    ) {
        state = OfflineSpeechState.UNAVAILABLE
        pending.clear()
        engine?.shutdown()
        engine = null
        Log.e(TAG, reason)
    }
}
