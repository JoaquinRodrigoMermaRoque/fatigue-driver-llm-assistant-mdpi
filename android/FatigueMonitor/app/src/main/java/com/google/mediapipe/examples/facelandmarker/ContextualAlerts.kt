/*
 * Deterministic context and safety policy for local contextual alerts.
 *
 * The fatigue engine remains the only component allowed to decide the
 * operational state. A future local language model may paraphrase the
 * language context, but it must preserve its urgency and mandatory action.
 */
package com.google.mediapipe.examples.facelandmarker

import java.util.Locale

enum class AlertUrgency {
    NONE,
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL
}

enum class AlertIntent {
    NONE,
    MONITOR_ATTENTION,
    REST_SOON,
    STOP_SAFELY,
    RECOVERY_GUIDANCE,
    NORMAL_STATUS,
    FACE_RESTORED,
    RESTORE_VISIBILITY
}

data class FatigueAlertSnapshot(
    val timestampMs: Long,
    val state: FatigueState,
    val stateEvent: String?,
    val score: Double,
    val reasons: List<String>,
    val closureS: Double,
    val perclosRatio: Double,
    val yawnsLast5Min: Int,
    val nodsLast2Min: Int,
    val faceValid: Boolean,
    val dataSufficient: Boolean,
    val alertAcknowledged: Boolean
)

data class AlertLanguageContext(
    val locale: String,
    val state: String,
    val urgency: String,
    val intent: String,
    val currentFacts: List<String>,
    val historicalFacts: List<String>,
    val mandatoryAction: String,
    val fallbackMessage: String
)

data class ContextualAlertDecision(
    val urgency: AlertUrgency,
    val intent: AlertIntent,
    val trigger: String,
    val history: FatigueHistorySummary,
    val languageContext: AlertLanguageContext
)

class ContextualAlertPlanner(
    private val historyHorizonMs: Long =
        30L * 60L * 1000L,
    private val criticalRepeatMs: Long = 30_000L,
    private val visibilityDelayMs: Long = 3_000L,
    private val visibilityRepeatMs: Long = 60_000L
) {
    private val lastEmissionByKey =
        mutableMapOf<String, Long>()
    private var faceMissingSinceMs: Long? = null

    init {
        require(historyHorizonMs > 0L)
        require(criticalRepeatMs > 0L)
        require(visibilityDelayMs >= 0L)
        require(visibilityRepeatMs > 0L)
    }

    @Synchronized
    fun resetSessionTracking() {
        lastEmissionByKey.clear()
        faceMissingSinceMs = null
    }

    @Synchronized
    fun evaluate(
        snapshot: FatigueAlertSnapshot,
        historyRecords: List<FatigueHistoryRecord>
    ): ContextualAlertDecision? {
        validateSnapshot(snapshot)

        if (!snapshot.faceValid) {
            return evaluateVisibility(
                snapshot,
                historyRecords
            )
        }

        if (faceMissingSinceMs != null) {
            val visibilityAlertWasEmitted =
                lastEmissionByKey.containsKey("visibility")
            faceMissingSinceMs = null
            lastEmissionByKey.remove("visibility")

            if (visibilityAlertWasEmitted) {
                return buildDecision(
                    snapshot = snapshot,
                    historyRecords = historyRecords,
                    urgency = AlertUrgency.LOW,
                    intent = AlertIntent.FACE_RESTORED,
                    trigger = "face_visible_again"
                )
            }
        }

        if (!snapshot.dataSufficient) {
            return null
        }

        val stateEvent = snapshot.stateEvent

        return when {
            snapshot.state == FatigueState.CRITICAL &&
                !snapshot.alertAcknowledged -> {
                val isTransition =
                    stateEvent == "state_critical" ||
                        stateEvent ==
                        "prolonged_eye_closure"

                emitIfAllowed(
                    key = "critical",
                    timestampMs = snapshot.timestampMs,
                    cooldownMs = criticalRepeatMs,
                    force = isTransition
                ) {
                    buildDecision(
                        snapshot = snapshot,
                        historyRecords = historyRecords,
                        urgency = AlertUrgency.CRITICAL,
                        intent = AlertIntent.STOP_SAFELY,
                        trigger = stateEvent
                            ?: "critical_repeat"
                    )
                }
            }

            stateEvent == "state_fatigue" -> {
                buildDecision(
                    snapshot,
                    historyRecords,
                    AlertUrgency.HIGH,
                    AlertIntent.REST_SOON,
                    stateEvent
                )
            }

            stateEvent == "state_suspected" -> {
                buildDecision(
                    snapshot,
                    historyRecords,
                    AlertUrgency.MEDIUM,
                    AlertIntent.MONITOR_ATTENTION,
                    stateEvent
                )
            }

            stateEvent == "state_recovery" -> {
                buildDecision(
                    snapshot,
                    historyRecords,
                    AlertUrgency.LOW,
                    AlertIntent.RECOVERY_GUIDANCE,
                    stateEvent
                )
            }

            stateEvent == "state_normal" -> {
                buildDecision(
                    snapshot,
                    historyRecords,
                    AlertUrgency.LOW,
                    AlertIntent.NORMAL_STATUS,
                    stateEvent
                )
            }

            else -> null
        }
    }

    private fun evaluateVisibility(
        snapshot: FatigueAlertSnapshot,
        historyRecords: List<FatigueHistoryRecord>
    ): ContextualAlertDecision? {
        val missingSince =
            faceMissingSinceMs ?: snapshot.timestampMs
                .also { faceMissingSinceMs = it }

        if (
            snapshot.timestampMs - missingSince <
            visibilityDelayMs
        ) {
            return null
        }

        return emitIfAllowed(
            key = "visibility",
            timestampMs = snapshot.timestampMs,
            cooldownMs = visibilityRepeatMs
        ) {
            buildDecision(
                snapshot = snapshot,
                historyRecords = historyRecords,
                urgency = AlertUrgency.MEDIUM,
                intent = AlertIntent.RESTORE_VISIBILITY,
                trigger = "face_not_visible"
            )
        }
    }

    private fun emitIfAllowed(
        key: String,
        timestampMs: Long,
        cooldownMs: Long,
        force: Boolean = false,
        block: () -> ContextualAlertDecision
    ): ContextualAlertDecision? {
        val previous = lastEmissionByKey[key]

        if (
            !force &&
            previous != null &&
            timestampMs - previous in 0 until cooldownMs
        ) {
            return null
        }

        lastEmissionByKey[key] = timestampMs
        return block()
    }

    private fun buildDecision(
        snapshot: FatigueAlertSnapshot,
        historyRecords: List<FatigueHistoryRecord>,
        urgency: AlertUrgency,
        intent: AlertIntent,
        trigger: String
    ): ContextualAlertDecision {
        val history = FatigueHistorySummarizer.summarize(
            records = historyRecords,
            nowMs = snapshot.timestampMs,
            horizonMs = historyHorizonMs
        )
        val currentFacts =
            buildCurrentFacts(snapshot)
        val historicalFacts =
            buildHistoricalFacts(history)
        val mandatoryAction =
            mandatoryAction(intent)
        val messageReasons =
            resolveMessageReasons(
                snapshot,
                historyRecords
            )
        val fallback = fallbackMessage(
            snapshot,
            history,
            intent,
            mandatoryAction,
            messageReasons
        )

        return ContextualAlertDecision(
            urgency = urgency,
            intent = intent,
            trigger = trigger,
            history = history,
            languageContext = AlertLanguageContext(
                locale = "es-PE",
                state = snapshot.state.name,
                urgency = urgency.name,
                intent = intent.name,
                currentFacts = currentFacts,
                historicalFacts = historicalFacts,
                mandatoryAction = mandatoryAction,
                fallbackMessage = fallback
            )
        )
    }

    private fun buildCurrentFacts(
        snapshot: FatigueAlertSnapshot
    ): List<String> {
        val facts = mutableListOf<String>()
        facts += "state=" + snapshot.state.name
        facts += "score=" + format(snapshot.score)
        facts += "closure_s=" +
            format(snapshot.closureS)
        facts += "perclos=" +
            format(snapshot.perclosRatio)
        facts += "yawns_5m=" +
            snapshot.yawnsLast5Min
        facts += "nods_2m=" +
            snapshot.nodsLast2Min
        facts += if (snapshot.reasons.isEmpty()) {
            "reasons=none"
        } else {
            "reasons=" +
                snapshot.reasons.joinToString(",")
        }
        return facts
    }

    private fun buildHistoricalFacts(
        history: FatigueHistorySummary
    ): List<String> {
        val facts = mutableListOf<String>()
        facts += "events_30m=" +
            history.eventCount
        facts += "critical_transitions_30m=" +
            history.criticalTransitions
        facts += "max_score_30m=" +
            format(history.maximumScore)

        history.mostFrequentReason?.let {
            facts += "main_reason_30m=$it"
        }

        listOf(
            "YAWN",
            "NOD",
            "GAZE_AWAY",
            "POSE_AWAY"
        ).forEach { eventType ->
            facts += eventType.lowercase() +
                "_30m=" +
                (history.countsByType[eventType] ?: 0)
        }

        return facts
    }

    private fun mandatoryAction(
        intent: AlertIntent
    ): String {
        return when (intent) {
            AlertIntent.MONITOR_ATTENTION ->
                "Mantén la mirada al frente y presta atención a cómo evolucionan las señales."
            AlertIntent.REST_SOON ->
                "Busca un lugar seguro para estacionarte y hacer una pausa en cuanto puedas."
            AlertIntent.STOP_SAFELY ->
                "Detente cuando sea seguro. Busca un lugar seguro para estacionarte y descansa antes de continuar."
            AlertIntent.RECOVERY_GUIDANCE ->
                "Ya te estás estabilizando. Mantén la mirada al frente y continúa con atención."
            AlertIntent.NORMAL_STATUS ->
                "Ya te estás estabilizando. Tus señales volvieron a valores normales. Mantén la mirada al frente."
            AlertIntent.FACE_RESTORED -> ""
            AlertIntent.RESTORE_VISIBILITY ->
                "Vuelve la mirada al frente y mantén tu rostro visible para la cámara."
            AlertIntent.NONE -> ""
        }
    }

    private fun resolveMessageReasons(
        snapshot: FatigueAlertSnapshot,
        historyRecords: List<FatigueHistoryRecord>
    ): List<String> {
        if (snapshot.reasons.isNotEmpty()) {
            return snapshot.reasons
        }

        val cutoff =
            snapshot.timestampMs - historyHorizonMs

        return historyRecords
            .filter {
                it.timestampMs in
                    cutoff..snapshot.timestampMs &&
                    it.eventType ==
                    "STATE_TRANSITION" &&
                    it.state ==
                    FatigueState.CRITICAL
            }
            .maxByOrNull { it.timestampMs }
            ?.reasons
            .orEmpty()
    }

    private fun fallbackMessage(
        snapshot: FatigueAlertSnapshot,
        history: FatigueHistorySummary,
        intent: AlertIntent,
        mandatoryAction: String,
        messageReasons: List<String>
    ): String {
        if (intent == AlertIntent.RESTORE_VISIBILITY) {
            return "Joaquín, no puedo verte bien. " +
                mandatoryAction
        }

        if (intent == AlertIntent.FACE_RESTORED) {
            val restoredMessages = listOf(
                "Oh, así está bien. Ya te veo.",
                "Ya te veo bien, Joaquín.",
                "Excelente, ahora sí te veo bien."
            )
            val index =
                (snapshot.timestampMs % restoredMessages.size)
                    .toInt()
            return restoredMessages[index]
        }

        val evidence = messageReasons
            .take(2)
            .map { reasonInSpanish(it) }
        val evidenceText = if (evidence.isEmpty()) {
            "cambios compatibles con fatiga"
        } else {
            evidence.joinToString(" y ")
        }
        val recurrence = if (
            history.criticalTransitions > 1
        ) {
            " Ya se registraron " +
                history.criticalTransitions +
                " episodios críticos en los últimos treinta minutos."
        } else {
            ""
        }

        return when (intent) {
            AlertIntent.MONITOR_ATTENTION ->
                "Joaquín, cuidado. Estoy detectando señales iniciales de fatiga: " +
                    "$evidenceText. Tu nivel está comenzando a subir. " +
                    mandatoryAction
            AlertIntent.REST_SOON ->
                "Joaquín, cuidado. No parece que las señales hayan bajado: " +
                    "$evidenceText. Tu nivel de fatiga está aumentando. " +
                    mandatoryAction
            AlertIntent.STOP_SAFELY ->
                "Joaquín, detecté señales fuertes de fatiga: " +
                    "$evidenceText.$recurrence " +
                    mandatoryAction
            AlertIntent.RECOVERY_GUIDANCE ->
                "Joaquín, ya te estás estabilizando. Las señales están bajando. " +
                    mandatoryAction
            AlertIntent.NORMAL_STATUS ->
                "Joaquín, ya te estás estabilizando. " +
                    mandatoryAction
            AlertIntent.FACE_RESTORED ->
                error("Handled above")
            AlertIntent.RESTORE_VISIBILITY ->
                error("Handled above")
            AlertIntent.NONE -> ""
        }
    }

    private fun reasonInSpanish(
        reason: String
    ): String {
        return when (reason) {
            "closure" ->
                "un cierre ocular prolongado"
            "perclos" ->
                "un porcentaje elevado de cierre ocular"
            "yawn" ->
                "bostezos repetidos"
            "gaze" ->
                "una desviación sostenida de la mirada"
            "nod" ->
                "cabeceos recientes"
            else ->
                "una señal de atención reducida"
        }
    }

    private fun validateSnapshot(
        snapshot: FatigueAlertSnapshot
    ) {
        require(snapshot.timestampMs >= 0L)
        require(
            snapshot.score.isFinite() &&
                snapshot.score in 0.0..1.0
        )
        require(
            snapshot.closureS.isFinite() &&
                snapshot.closureS >= 0.0
        )
        require(
            snapshot.perclosRatio.isFinite() &&
                snapshot.perclosRatio >= 0.0
        )
        require(snapshot.yawnsLast5Min >= 0)
        require(snapshot.nodsLast2Min >= 0)
    }

    private fun format(value: Double): String {
        return String.format(
            Locale.US,
            "%.3f",
            value
        )
    }
}
