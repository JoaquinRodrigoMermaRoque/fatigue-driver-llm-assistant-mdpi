package com.google.mediapipe.examples.facelandmarker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SafeContextualLanguageTest {

    @Test
    fun promptContainsFactsAndSafetyBoundary() {
        val request = ContextualLanguagePromptBuilder
            .build(decision())

        assertTrue(request.prompt.contains("/no_think"))
        assertTrue(
            request.prompt.contains(
                "senal=cierre ocular"
            )
        )
        assertTrue(
            request.prompt.contains(
                "criticos=2"
            )
        )
        assertTrue(
            request.prompt.contains(
                "Sin inventar"
            )
        )
        assertEquals(16, request.maxOutputTokens)
        assertEquals(25_000L, request.timeoutMs)
        assertTrue(request.prompt.length < 220)
    }

    @Test
    fun validIntroductionKeepsMandatoryActionVerbatim() {
        val decision = decision()
        val result = SafeContextualMessageComposer.compose(
            decision,
            "Joaquín, el cierre ocular prolongado volvió a presentarse " +
                "junto con otras señales recientes"
        )

        assertEquals(
            ContextualMessageStatus.GENERATED,
            result.status
        )
        assertTrue(result.usedLocalModel)
        assertEquals(
            decision.languageContext.mandatoryAction,
            result.text.substringAfter(". ")
        )
    }

    @Test
    fun shortValidContextualIntroductionIsAccepted() {
        val decision = decision()
        val result = SafeContextualMessageComposer.compose(
            decision,
            "Joaquín, el cierre ocular prolongado volvió a presentarse otra vez"
        )

        assertEquals(
            ContextualMessageStatus.GENERATED,
            result.status
        )
        assertTrue(result.usedLocalModel)
    }

    @Test
    fun emptyCandidateUsesDeterministicFallback() {
        val decision = decision()
        val result = SafeContextualMessageComposer.compose(
            decision,
            "  "
        )

        assertEquals(
            ContextualMessageStatus.FALLBACK_EMPTY,
            result.status
        )
        assertEquals(
            decision.languageContext.fallbackMessage,
            result.text
        )
        assertFalse(result.usedLocalModel)
    }

    @Test
    fun contradictoryAdviceIsRejected() {
        val decision = decision()
        val result = SafeContextualMessageComposer.compose(
            decision,
            "Todo está bien, puedes conducir con normalidad"
        )

        assertEquals(
            ContextualMessageStatus.FALLBACK_UNSAFE,
            result.status
        )
        assertEquals(
            decision.languageContext.fallbackMessage,
            result.text
        )
    }

    @Test
    fun medicalDiagnosisIsRejected() {
        val result = SafeContextualMessageComposer.compose(
            decision(),
            "Este patrón confirma un diagnóstico de una enfermedad"
        )

        assertEquals(
            ContextualMessageStatus.FALLBACK_UNSAFE,
            result.status
        )
    }

    @Test
    fun inventedNumericClaimIsRejected() {
        val result = SafeContextualMessageComposer.compose(
            decision(),
            "Detecté exactamente 14 episodios graves durante este trayecto"
        )

        assertEquals(
            ContextualMessageStatus.FALLBACK_FORMAT,
            result.status
        )
    }

    @Test
    fun multipleSentencesAreRejected() {
        val result = SafeContextualMessageComposer.compose(
            decision(),
            "Tu atención ha disminuido. La situación empeoró."
        )

        assertEquals(
            ContextualMessageStatus.FALLBACK_FORMAT,
            result.status
        )
    }

    @Test
    fun hiddenThinkingIsRemovedBeforeValidation() {
        val result = SafeContextualMessageComposer.compose(
            decision(),
            "<think>Debo razonar primero.</think> " +
                "Joaquín, el cierre ocular prolongado volvió a presentarse " +
                "junto con otras señales recientes"
        )

        assertEquals(
            ContextualMessageStatus.GENERATED,
            result.status
        )
        assertFalse(result.text.contains("<think>"))
        assertFalse(result.text.contains("razonar"))
    }

    @Test
    fun observedMoralizingOutputFromSmallModelIsRejected() {
        val result = SafeContextualMessageComposer.compose(
            decision(),
            "La situación requiere que el conductor se sienta " +
                "responsable por su condición física y priorice " +
                "la recuperación para evitar futuros problemas"
        )

        assertEquals(
            ContextualMessageStatus.FALLBACK_UNSAFE,
            result.status
        )
    }

    @Test
    fun observedMalformedShortOutputIsRejected() {
        val result = SafeContextualMessageComposer.compose(
            decision(),
            "El cierre ocular ha repetido y el estado esgrave"
        )

        assertEquals(
            ContextualMessageStatus.FALLBACK_FORMAT,
            result.status
        )
    }

    @Test
    fun observedRepeatedMandatoryActionIsRejected() {
        val result = SafeContextualMessageComposer.compose(
            decision(),
            "El cierre ocular prolongado se ha repetido, " +
                "indicando un estado de fatiga grave que requiere " +
                "detenerse en un lugar seguro y descansar antes " +
                "de continuar"
        )

        assertEquals(
            ContextualMessageStatus.FALLBACK_UNSAFE,
            result.status
        )
    }

    @Test
    fun reservedActionFragmentsAreRejectedWithinLengthLimit() {
        val result = SafeContextualMessageComposer.compose(
            decision(),
            "El cierre ocular volvió a aparecer; conviene buscar " +
                "un lugar seguro antes de continuar"
        )

        assertEquals(
            ContextualMessageStatus.FALLBACK_UNSAFE,
            result.status
        )
    }

    @Test
    fun genericIntroductionWithoutObservedSignalIsRejected() {
        val result = SafeContextualMessageComposer.compose(
            decision(),
            "Las señales recientes volvieron a aparecer y requieren " +
                "que mantengas especial atención en este momento"
        )

        assertEquals(
            ContextualMessageStatus.FALLBACK_UNSAFE,
            result.status
        )
    }

    @Test
    fun runtimeEndMarkerIsRemoved() {
        val result = SafeContextualMessageComposer.compose(
            decision(),
            "Joaquín, el cierre ocular prolongado volvió a " +
                "presentarse junto con otras señales recientes. " +
                "[end of text]"
        )

        assertEquals(
            ContextualMessageStatus.GENERATED,
            result.status
        )
        assertFalse(
            result.text.contains(
                "[end of text]",
                ignoreCase = true
            )
        )
    }

    @Test
    fun excessiveOutputUsesFallback() {
        val longCandidate = (
            listOf("cierre", "ocular") +
                (1..29).map { "señal" }
            )
            .joinToString(" ")
        val result = SafeContextualMessageComposer.compose(
            decision(),
            longCandidate
        )

        assertEquals(
            ContextualMessageStatus.FALLBACK_TOO_LONG,
            result.status
        )
    }

    private fun decision(): ContextualAlertDecision {
        val mandatoryAction =
            "Detenerse en un lugar seguro y descansar antes de continuar."
        val fallback =
            "Alerta crítica. Detecté un cierre ocular prolongado. " +
                mandatoryAction

        return ContextualAlertDecision(
            urgency = AlertUrgency.CRITICAL,
            intent = AlertIntent.STOP_SAFELY,
            trigger = "prolonged_eye_closure",
            history = FatigueHistorySummary(
                eventCount = 8,
                countsByType = mapOf(
                    "YAWN" to 2,
                    "NOD" to 1
                ),
                criticalTransitions = 2,
                maximumScore = 0.85,
                latestState = FatigueState.CRITICAL,
                mostFrequentReason = "closure"
            ),
            languageContext = AlertLanguageContext(
                locale = "es-PE",
                state = "CRITICAL",
                urgency = "CRITICAL",
                intent = "STOP_SAFELY",
                currentFacts = listOf(
                    "state=CRITICAL",
                    "score=0.850",
                    "closure_s=2.100",
                    "perclos=0.350",
                    "reasons=closure,perclos"
                ),
                historicalFacts = listOf(
                    "events_30m=8",
                    "critical_transitions_30m=2",
                    "yawn_30m=2",
                    "nod_30m=1"
                ),
                mandatoryAction = mandatoryAction,
                fallbackMessage = fallback
            )
        )
    }
}
