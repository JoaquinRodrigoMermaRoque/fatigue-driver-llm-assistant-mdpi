/*
 * Safe language layer for the optional on-device LLM.
 *
 * The deterministic fatigue engine and ContextualAlertPlanner remain the
 * authority for state, urgency and mandatory action. The model may only
 * create a short contextual introduction.
 */
package com.google.mediapipe.examples.facelandmarker

import java.text.Normalizer

enum class ContextualMessageStatus {
    GENERATED,
    FALLBACK_CONTEXTUAL,
    FALLBACK_EMPTY,
    FALLBACK_TOO_SHORT,
    FALLBACK_TOO_LONG,
    FALLBACK_UNSAFE,
    FALLBACK_FORMAT
}

data class SafeContextualMessage(
    val text: String,
    val status: ContextualMessageStatus,
    val generatedIntroduction: String?
) {
    val generatedByLocalModel: Boolean
        get() = status == ContextualMessageStatus.GENERATED

    val usedLocalModel: Boolean
        get() = shouldSpeakContextual

    val shouldSpeakContextual: Boolean
        get() =
            status == ContextualMessageStatus.GENERATED ||
                status ==
                    ContextualMessageStatus.FALLBACK_CONTEXTUAL
}

data class LocalLanguageRequest(
    val prompt: String,
    val maxOutputTokens: Int = 16,
    val temperature: Float = 0.35f,
    val topP: Float = 0.80f,
    val timeoutMs: Long = 25_000L
) {
    init {
        require(prompt.isNotBlank())
        require(maxOutputTokens in 8..96)
        require(temperature in 0.0f..1.0f)
        require(topP in 0.1f..1.0f)
        require(timeoutMs in 250L..30_000L)
    }
}

/**
 * Runtime boundary implemented later by the llama.cpp Android bridge.
 */
fun interface LocalAlertLanguageModel {
    fun generate(request: LocalLanguageRequest): String
}

object ContextualLanguagePromptBuilder {
    private const val DRIVER_NAME = "Joaquín"

    fun build(
        decision: ContextualAlertDecision
    ): LocalLanguageRequest {
        val context = decision.languageContext
        val current = context.currentFacts.associateFactValues()
        val history = context.historicalFacts.associateFactValues()

        val signalFact =
            when (decision.trigger) {
                "prolonged_eye_closure" ->
                    "senal=cierre ocular"
                "gaze_away" ->
                    "senal=mirada desviada"
                "yawn" ->
                    "senal=bostezo"
                "nod" ->
                    "senal=cabeceo"
                else -> {
                    val reasons =
                        current["reasons"].orEmpty()
                    when {
                        "closure" in reasons ||
                            "perclos" in reasons ->
                            "senal=cierre ocular"
                        "yawn" in reasons ->
                            "senal=bostezo"
                        "gaze" in reasons ->
                            "senal=mirada desviada"
                        "nod" in reasons ->
                            "senal=cabeceo"
                        else ->
                            "senal=alerta actual"
                    }
                }
            }

        val userFact =
            history["sleep_last_night_h"]
                ?.let {
                    "contexto=reportaste $it horas de sueno"
                }
                ?: history["sleepiness_1_9"]
                    ?.let {
                        "contexto=reportaste somnolencia $it de 9"
                    }
                ?: history["hours_awake"]
                    ?.let {
                        "contexto=llevas $it horas despierto"
                    }
                ?: history["sleep_avg_3d_h"]
                    ?.let {
                        "contexto=reportaste promedio de sueno $it horas"
                    }
                ?: history["work_shift_h"]
                    ?.let {
                        "contexto=reportaste jornada de $it horas"
                    }
                ?: history["physical_demand"]
                    ?.let {
                        "contexto=demanda fisica reportada $it"
                    }
                ?: history["mental_demand"]
                    ?.let {
                        "contexto=demanda mental reportada $it"
                    }

        val historyFact =
            history["critical_transitions_30m"]
                ?.takeIf { it != "0" }
                ?.let { "criticos=$it" }
                ?: history["events_30m"]
                    ?.takeIf { it != "0" }
                    ?.let { "eventos=$it" }

        val contextFact = userFact ?: historyFact
        val dataFacts =
            listOfNotNull(
                signalFact,
                contextFact
            ).joinToString(";")

        val prompt = """
            /no_think
            8-14 palabras. Empieza "$DRIVER_NAME,".
            Menciona obligatoriamente la senal y el contexto si aparece.
            Sin inventar ni aconsejar.
            $dataFacts
        """.trimIndent()

        return LocalLanguageRequest(prompt = prompt)
    }

    private fun List<String>.associateFactValues():
        Map<String, String> {
        return mapNotNull { fact ->
            val separator = fact.indexOf('=')
            if (
                separator <= 0 ||
                separator >= fact.lastIndex
            ) {
                null
            } else {
                fact.substring(0, separator) to
                    fact.substring(separator + 1)
            }
        }.toMap()
    }
}

/**
 * Rejects unsafe or malformed model output and appends the deterministic
 * mandatory action verbatim. A rejected candidate always returns the
 * planner's original fallback.
 */
object SafeContextualMessageComposer {
    private const val MIN_WORDS = 8
    private const val MAX_WORDS = 16
    private const val MAX_INTRODUCTION_CHARS = 220

    private val forbiddenFragments = listOf(
        "continúa conduciendo",
        "continua conduciendo",
        "sigue conduciendo",
        "puedes conducir",
        "puede conducir",
        "no te detengas",
        "no es necesario detener",
        "ignora la alerta",
        "no hay riesgo",
        "todo está bien",
        "no estás fatigado",
        "no estas fatigado",
        "diagnóstico",
        "diagnostico",
        "enfermedad",
        "medicamento",
        "pastilla",
        "alcohol",
        "cafeína",
        "cafeina",
        "energizante",
        "responsable",
        "irresponsable",
        "culpa",
        "condición física",
        "condicion fisica",
        "problemas futuros",
        "entendiendo la situación",
        "entendiendo la situacion",
        "mantener la calma",
        "mantén la calma",
        "manten la calma",
        "seguir las indicaciones",
        "fatiga grave",
        "estado grave",
        "detenerse",
        "detente",
        "descansar",
        "descansa",
        "lugar seguro",
        "antes de continuar",
        "http://",
        "https://"
    )

    fun compose(
        decision: ContextualAlertDecision,
        candidate: String?
    ): SafeContextualMessage {
        if (candidate.isNullOrBlank()) {
            return fallback(
                decision,
                ContextualMessageStatus.FALLBACK_EMPTY
            )
        }

        val introduction = sanitize(candidate)

        if (introduction.isBlank()) {
            return fallback(
                decision,
                ContextualMessageStatus.FALLBACK_EMPTY
            )
        }

        if (!hasSafeFormat(introduction)) {
            return fallback(
                decision,
                ContextualMessageStatus.FALLBACK_FORMAT
            )
        }

        if (
            hasUnsafeContent(introduction)
        ) {
            return fallback(
                decision,
                ContextualMessageStatus.FALLBACK_UNSAFE
            )
        }

        if (!usesOnlyKnownNumbers(decision, introduction)) {
            return fallback(
                decision,
                ContextualMessageStatus.FALLBACK_FORMAT
            )
        }

        if (!mentionsObservedSignal(decision, introduction)) {
            return fallback(
                decision,
                ContextualMessageStatus.FALLBACK_UNSAFE
            )
        }

        if (!mentionsRequiredUserContext(decision, introduction)) {
            return fallback(
                decision,
                ContextualMessageStatus.FALLBACK_FORMAT
            )
        }

        val words = introduction
            .split(Regex("\\s+"))
            .filter { it.isNotBlank() }

        if (words.size < MIN_WORDS) {
            return fallback(
                decision,
                ContextualMessageStatus.FALLBACK_TOO_SHORT
            )
        }

        if (
            words.size > MAX_WORDS ||
            introduction.length > MAX_INTRODUCTION_CHARS
        ) {
            return fallback(
                decision,
                ContextualMessageStatus.FALLBACK_TOO_LONG
            )
        }

        if (!startsWithDriverName(introduction)) {
            return fallback(
                decision,
                ContextualMessageStatus.FALLBACK_FORMAT
            )
        }

        val punctuated = when {
            introduction.last() in charArrayOf('.', '!', '?') ->
                introduction
            else -> "$introduction."
        }
        val mandatoryAction = decision.languageContext
            .mandatoryAction
            .trim()
        val finalText = listOf(
            punctuated,
            mandatoryAction
        )
            .filter { it.isNotBlank() }
            .joinToString(" ")

        return SafeContextualMessage(
            text = finalText,
            status = ContextualMessageStatus.GENERATED,
            generatedIntroduction = punctuated
        )
    }

    private fun sanitize(candidate: String): String {
        return candidate
            .replace(
                Regex(
                    "<think>.*?</think>",
                    setOf(
                        RegexOption.IGNORE_CASE,
                        RegexOption.DOT_MATCHES_ALL
                    )
                ),
                " "
            )
            .replace(
                Regex(
                    "^\\s*(respuesta|introducción|introduccion)\\s*:\\s*",
                    RegexOption.IGNORE_CASE
                ),
                ""
            )
            .replace(
                Regex(
                    "\\[end of text]",
                    RegexOption.IGNORE_CASE
                ),
                " "
            )
            .replace(Regex("[\\r\\n\\t]+"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
            .trim('"', '\'', '“', '”')
            .trim()
    }

    private fun startsWithDriverName(
        introduction: String
    ): Boolean {
        return normalizeForComparison(introduction)
            .startsWith("joaquin ")
    }

    private fun hasUnsafeContent(
        introduction: String
    ): Boolean {
        val normalized = normalizeForComparison(introduction)

        if (
            forbiddenFragments.any {
                normalized.contains(
                    normalizeForComparison(it)
                )
            }
        ) {
            return true
        }

        return false
    }

    private fun mentionsObservedSignal(
        decision: ContextualAlertDecision,
        introduction: String
    ): Boolean {
        val normalized = normalizeForComparison(introduction)
        val requiredFragments = mutableListOf<String>()

        when (decision.trigger) {
            "prolonged_eye_closure" ->
                requiredFragments += listOf(
                    "cierre ocular",
                    "ojos",
                    "parpados"
                )
            "face_not_visible" ->
                requiredFragments += listOf(
                    "rostro",
                    "camara",
                    "visibilidad"
                )
            "gaze_away" ->
                requiredFragments += listOf(
                    "mirada",
                    "vista"
                )
            "yawn" ->
                requiredFragments += listOf(
                    "bostezo",
                    "bostezos"
                )
            "nod" ->
                requiredFragments += listOf(
                    "cabeceo",
                    "cabeceos"
                )
        }

        val reasons = decision.languageContext
            .currentFacts
            .firstOrNull {
                it.startsWith("reasons=")
            }
            ?.substringAfter("=")
            ?.split(",")
            ?.map { it.trim() }
            .orEmpty()

        if ("closure" in reasons) {
            requiredFragments += listOf(
                "cierre ocular",
                "ojos",
                "parpados"
            )
        }
        if ("perclos" in reasons) {
            requiredFragments += listOf(
                "cierre ocular",
                "perclos"
            )
        }
        if ("yawn" in reasons) {
            requiredFragments += listOf(
                "bostezo",
                "bostezos"
            )
        }
        if ("gaze" in reasons) {
            requiredFragments += listOf(
                "mirada",
                "vista"
            )
        }
        if ("nod" in reasons) {
            requiredFragments += listOf(
                "cabeceo",
                "cabeceos"
            )
        }

        if (requiredFragments.isEmpty()) {
            return true
        }

        return requiredFragments.distinct().any {
            normalized.contains(
                normalizeForComparison(it)
            )
        }
    }

    private fun mentionsRequiredUserContext(
        decision: ContextualAlertDecision,
        introduction: String
    ): Boolean {
        val facts =
            decision.languageContext.historicalFacts
                .associateFactValues()
        val normalized =
            normalizeForComparison(introduction)

        fun containsKnownNumber(
            raw: String?
        ): Boolean {
            val expected =
                raw?.replace(',', '.')?.toDoubleOrNull()
                    ?: return false
            val numberRegex =
                Regex("""\d+(?:[\.,]\d+)?""")
            return numberRegex
                .findAll(introduction)
                .mapNotNull {
                    it.value
                        .replace(',', '.')
                        .toDoubleOrNull()
                }
                .any {
                    kotlin.math.abs(it - expected) < 0.0001
                }
        }

        facts["sleep_last_night_h"]?.let { value ->
            return containsKnownNumber(value) &&
                (
                    normalized.contains("sueno") ||
                    normalized.contains("dormi") ||
                    normalized.contains("dormiste") ||
                    normalized.contains("horas de descanso")
                )
        }

        facts["sleepiness_1_9"]?.let { value ->
            return containsKnownNumber(value) &&
                normalized.contains("somnolencia")
        }

        facts["hours_awake"]?.let { value ->
            return containsKnownNumber(value) &&
                normalized.contains("despiert")
        }

        facts["sleep_avg_3d_h"]?.let { value ->
            return containsKnownNumber(value) &&
                normalized.contains("sueno")
        }

        facts["work_shift_h"]?.let { value ->
            return containsKnownNumber(value) &&
                (
                    normalized.contains("jornada") ||
                    normalized.contains("turno")
                )
        }

        facts["physical_demand"]?.let { value ->
            return normalized.contains("fisic") &&
                normalized.contains(
                    normalizeForComparison(value)
                )
        }

        facts["mental_demand"]?.let { value ->
            return normalized.contains("mental") &&
                normalized.contains(
                    normalizeForComparison(value)
                )
        }

        return true
    }

    private fun List<String>.associateFactValues():
        Map<String, String> {
        return mapNotNull { fact ->
            val separator = fact.indexOf('=')
            if (
                separator <= 0 ||
                separator >= fact.lastIndex
            ) {
                null
            } else {
                fact.substring(0, separator) to
                    fact.substring(separator + 1)
            }
        }.toMap()
    }

    private fun normalizeForComparison(
        value: String
    ): String {
        return Normalizer
            .normalize(
                value,
                Normalizer.Form.NFD
            )
            .replace(Regex("\\p{M}+"), "")
            .lowercase()
            .replace(Regex("[^a-z0-9]+"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    private fun hasSafeFormat(
        introduction: String
    ): Boolean {
        if (
            introduction.contains("~~~") ||
            introduction.contains("<|") ||
            introduction.contains("|>")
        ) {
            return false
        }

        val sentenceTerminators =
            introduction.count {
                it == '.' || it == '!' || it == '?'
            }

        return sentenceTerminators <= 1
    }

    private fun usesOnlyKnownNumbers(
        decision: ContextualAlertDecision,
        introduction: String
    ): Boolean {
        val numberRegex =
            Regex("""\d+(?:[\.,]\d+)?""")
        val candidateNumbers =
            numberRegex.findAll(introduction)
                .mapNotNull {
                    it.value
                        .replace(',', '.')
                        .toDoubleOrNull()
                }
                .toList()

        if (candidateNumbers.isEmpty()) {
            return true
        }

        val knownNumbers =
            (
                decision.languageContext.currentFacts +
                    decision.languageContext.historicalFacts
                )
                .flatMap { fact ->
                    val value = fact.substringAfter(
                        '=',
                        missingDelimiterValue = ""
                    )
                    numberRegex.findAll(value)
                        .mapNotNull {
                            it.value
                                .replace(',', '.')
                                .toDoubleOrNull()
                        }
                        .toList()
                }

        return candidateNumbers.all { candidate ->
            knownNumbers.any { known ->
                kotlin.math.abs(candidate - known) <
                    0.0001
            }
        }
    }

    private fun fallback(
        decision: ContextualAlertDecision,
        status: ContextualMessageStatus
    ): SafeContextualMessage {
        val contextualFallback =
            buildContextualFallback(decision)

        if (contextualFallback != null) {
            return SafeContextualMessage(
                text = contextualFallback,
                status =
                    ContextualMessageStatus
                        .FALLBACK_CONTEXTUAL,
                generatedIntroduction = null
            )
        }

        return SafeContextualMessage(
            text = decision.languageContext
                .fallbackMessage,
            status = status,
            generatedIntroduction = null
        )
    }

    private fun buildContextualFallback(
        decision: ContextualAlertDecision
    ): String? {
        val facts =
            decision.languageContext.historicalFacts
                .associateFactValues()

        val contextPhrase =
            facts["sleep_last_night_h"]
                ?.let {
                    "reportaste $it horas de sueño"
                }
                ?: facts["sleepiness_1_9"]
                    ?.let {
                        "reportaste somnolencia $it de 9"
                    }
                ?: facts["hours_awake"]
                    ?.let {
                        "llevas $it horas despierto"
                    }
                ?: facts["sleep_avg_3d_h"]
                    ?.let {
                        "reportaste un promedio de sueño de $it horas"
                    }
                ?: facts["work_shift_h"]
                    ?.let {
                        "reportaste una jornada de $it horas"
                    }
                ?: facts["physical_demand"]
                    ?.let {
                        "reportaste demanda física $it"
                    }
                ?: facts["mental_demand"]
                    ?.let {
                        "reportaste demanda mental $it"
                    }
                ?: return null

        val signalPhrase =
            when (decision.trigger) {
                "prolonged_eye_closure" ->
                    "detecté un cierre ocular prolongado"
                "gaze_away" ->
                    "detecté una desviación sostenida de la mirada"
                "yawn" ->
                    "detecté un bostezo"
                "nod" ->
                    "detecté un cabeceo"
                else ->
                    "detecté señales de fatiga"
            }

        val mandatoryAction =
            decision.languageContext
                .mandatoryAction
                .trim()

        return "Joaquín, $signalPhrase; $contextPhrase. " +
            mandatoryAction
    }
}
