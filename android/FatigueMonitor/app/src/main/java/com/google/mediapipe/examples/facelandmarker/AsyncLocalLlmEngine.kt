package com.google.mediapipe.examples.facelandmarker

import android.content.Context
import android.os.Process
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class AsyncLocalLlmEngine(
    context: Context
) {
    private data class PendingGeneration(
        val decision: ContextualAlertDecision,
        val onGenerated: (SafeContextualMessage) -> Unit
    )
    companion object {
        private const val TAG = "H7_11_ASYNC"
        private const val MODEL_NAME =
            "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
    }

    private val appContext = context.applicationContext
    private val executor =
        Executors.newSingleThreadExecutor { task ->
            Thread(
                {
                    Process.setThreadPriority(
                        Process.THREAD_PRIORITY_BACKGROUND
                    )
                    task.run()
                },
                "fatigue-llm-async"
            )
        }
    private val generationInFlight =
        AtomicBoolean(false)
    private val timeoutExecutor =
        Executors.newSingleThreadScheduledExecutor { task ->
            Thread(task, "fatigue-llm-timeout")
        }
    private val pendingLock = Any()

    @Volatile
    private var currentUrgency =
        AlertUrgency.NONE

    private var pendingGeneration:
        PendingGeneration? = null

    @Volatile
    private var ready = false

    @Volatile
    private var closed = false

    fun initialize() {
        if (closed) return

        executor.execute {
            if (ready || closed) return@execute

            val externalDir =
                appContext.getExternalFilesDir(null)
            val modelFile = externalDir?.let {
                File(File(it, "models"), MODEL_NAME)
            }

            if (modelFile == null || !modelFile.isFile) {
                Log.e(
                    TAG,
                    "INIT_ERROR model_missing path=" +
                        (modelFile?.absolutePath ?: "null")
                )
                return@execute
            }

            val startedMs = SystemClock.elapsedRealtime()
            val result =
                LocalLlmBridge.initializeModel(
                    modelFile.absolutePath
                )
            val elapsedMs =
                SystemClock.elapsedRealtime() - startedMs

            ready = result.startsWith(
                "H7.10.5 INIT_OK"
            )

            Log.i(
                TAG,
                "INIT ready=$ready elapsed_ms=$elapsedMs result=$result"
            )
        }
    }

    fun submit(
        decision: ContextualAlertDecision,
        onGenerated: (SafeContextualMessage) -> Unit
    ): Boolean {
        if (closed || !ready) {
            Log.i(
                TAG,
                "SKIP ready=$ready closed=$closed " +
                    "in_flight=${generationInFlight.get()} " +
                    "trigger=${decision.trigger}"
            )
            return false
        }

        if (!generationInFlight.compareAndSet(false, true)) {
            val queued = synchronized(pendingLock) {
                if (
                    decision.urgency.ordinal >
                    currentUrgency.ordinal
                ) {
                    val existing = pendingGeneration
                    if (
                        existing == null ||
                        decision.urgency.ordinal >=
                        existing.decision.urgency.ordinal
                    ) {
                        pendingGeneration =
                            PendingGeneration(
                                decision,
                                onGenerated
                            )
                    }
                    true
                } else {
                    false
                }
            }

            if (queued) {
                Log.i(
                    TAG,
                    "PREEMPT current_urgency=" +
                        currentUrgency +
                        " next_urgency=" +
                        decision.urgency +
                        " trigger=" +
                        decision.trigger
                )
                LocalLlmBridge.cancelGeneration()
                return true
            }

            Log.i(
                TAG,
                "SKIP ready=$ready closed=$closed " +
                    "in_flight=true " +
                    "trigger=${decision.trigger}"
            )
            return false
        }

        currentUrgency = decision.urgency
        val submittedMs = SystemClock.elapsedRealtime()
        Log.i(
            TAG,
            "SUBMIT trigger=${decision.trigger} " +
                "urgency=${decision.urgency}"
        )

        executor.execute {
            try {
                if (closed) return@execute

                val request =
                    ContextualLanguagePromptBuilder.build(
                        decision
                    )
                Log.i(
                    TAG,
                    "PROMPT trigger=${decision.trigger} " +
                        "chars=${request.prompt.length} " +
                        "max_new=${request.maxOutputTokens}"
                )

                val timeoutFuture =
                    timeoutExecutor.schedule(
                        {
                            if (
                                !closed &&
                                generationInFlight.get()
                            ) {
                                Log.w(
                                    TAG,
                                    "TIMEOUT trigger=" +
                                        decision.trigger +
                                        " after_ms=" +
                                        request.timeoutMs
                                )
                                LocalLlmBridge
                                    .cancelGeneration()
                            }
                        },
                        request.timeoutMs,
                        TimeUnit.MILLISECONDS
                    )

                val nativeResult =
                    try {
                        LocalLlmBridge.generate(
                            request.prompt,
                            request.maxOutputTokens
                        )
                    } finally {
                        timeoutFuture.cancel(false)
                    }

                Log.i(
                    TAG,
                    "NATIVE trigger=${decision.trigger} " +
                        nativeResult
                            .substringBefore(" response=")
                            .take(300)
                )

                val candidate =
                    extractGeneratedText(nativeResult)
                Log.i(
                    TAG,
                    "LLM_CANDIDATE trigger=${decision.trigger} text=" +
                        (candidate ?: "<null>")
                            .replace(Regex("[\\r\\n\\t]+"), " ")
                            .take(300)
                )

                val safeMessage =
                    SafeContextualMessageComposer.compose(
                        decision = decision,
                        candidate = candidate
                    )
                val elapsedMs =
                    SystemClock.elapsedRealtime() -
                        submittedMs

                Log.i(
                    TAG,
                    "RESULT status=${safeMessage.status} " +
                        "elapsed_ms=$elapsedMs " +
                        "trigger=${decision.trigger}"
                )

                if (
                    !closed &&
                    safeMessage.shouldSpeakContextual
                ) {
                    onGenerated(safeMessage)
                }
            } finally {
                generationInFlight.set(false)
                currentUrgency = AlertUrgency.NONE

                val pending =
                    synchronized(pendingLock) {
                        pendingGeneration.also {
                            pendingGeneration = null
                        }
                    }

                if (pending != null && !closed) {
                    Log.i(
                        TAG,
                        "DRAIN_PENDING trigger=" +
                            pending.decision.trigger +
                            " urgency=" +
                            pending.decision.urgency
                    )
                    submit(
                        pending.decision,
                        pending.onGenerated
                    )
                }
            }
        }

        return true
    }

    fun isGenerationInFlight(): Boolean {
        return generationInFlight.get()
    }

    fun cancelGeneration(
        reason: String = "external"
    ) {
        if (!closed) {
            synchronized(pendingLock) {
                pendingGeneration = null
            }
            Log.i(
                TAG,
                "CANCEL_REQUEST reason=$reason " +
                    "in_flight=${generationInFlight.get()} " +
                    "current_urgency=$currentUrgency"
            )
            LocalLlmBridge.cancelGeneration()
        }
    }

    fun close() {
        if (closed) return
        closed = true
        synchronized(pendingLock) {
            pendingGeneration = null
        }
        LocalLlmBridge.cancelGeneration()
        timeoutExecutor.shutdownNow()

        executor.execute {
            val result = LocalLlmBridge.releaseModel()
            ready = false
            Log.i(TAG, "CLOSE $result")
        }
        executor.shutdown()
    }

    private fun extractGeneratedText(
        nativeResult: String
    ): String? {
        val marker = " response="
        val hasGeneratedText =
            nativeResult.startsWith(
                "H7.10.5 GENERATE_OK"
            ) ||
                nativeResult.startsWith(
                    "H7.10.5 GENERATE_PARTIAL"
                )

        if (!hasGeneratedText) {
            Log.w(
                TAG,
                "GENERATE_NATIVE_ERROR $nativeResult"
            )
            return null
        }

        val index = nativeResult.indexOf(marker)
        if (index < 0) return null

        return nativeResult
            .substring(index + marker.length)
            .trim()
            .takeIf { it.isNotEmpty() }
    }
}
