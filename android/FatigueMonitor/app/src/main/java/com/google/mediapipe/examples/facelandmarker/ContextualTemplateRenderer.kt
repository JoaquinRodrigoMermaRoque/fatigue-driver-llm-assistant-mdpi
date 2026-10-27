package com.google.mediapipe.examples.facelandmarker

/**
 * Closed-vocabulary output that a small local LLM may select.
 *
 * The model does not write user-facing prose. It only chooses identifiers.
 * User-facing text is rendered locally from reviewed templates.
 */
enum class ContextTemplateId(val wireValue: String) {
    REPEATED_PRIMARY_WITH_RECENT_EVENTS(
        "repeated_primary_with_recent_events"
    ),
    RECENT_EVENTS_CHANGED_FROM_HISTORY(
        "recent_events_changed_from_history"
    ),
    RECOVERY_AFTER_CRITICAL(
        "recovery_after_critical"
    )
}

enum class ContextPrimarySignal(val wireValue: String) {
    EYE_CLOSURE("eye_closure"),
    YAWNS("yawns"),
    RECOVERY("recovery")
}

enum class ContextSecondarySignal(val wireValue: String) {
    NONE("none"),
    YAWNS_AND_NOD("yawns_and_nod"),
    GAZE_AWAY("gaze_away")
}

data class StructuredContextSelection(
    val template: ContextTemplateId,
    val primary: ContextPrimarySignal,
    val secondary: ContextSecondarySignal
)

enum class ContextStyleVariant(val wireValue: String) {
    A("a"),
    B("b"),
    C("c")
}

object ContextualTemplateRenderer {

    /**
     * Renders only combinations that have been explicitly reviewed.
     * Unsupported combinations return null and must fall back to the
     * deterministic alert path.
     */
    fun render(
        selection: StructuredContextSelection,
        displayName: String? = null,
        locale: String = "es-PE",
        variant: ContextStyleVariant = ContextStyleVariant.A
    ): String? {
        if (locale != "es-PE") {
            return null
        }

        val body = when (selection) {
            StructuredContextSelection(
                ContextTemplateId.REPEATED_PRIMARY_WITH_RECENT_EVENTS,
                ContextPrimarySignal.EYE_CLOSURE,
                ContextSecondarySignal.YAWNS_AND_NOD
            ) -> when (variant) {
                ContextStyleVariant.A ->
                    "el cierre ocular prolongado volvió a presentarse; " +
                        "además, se registraron bostezos y un cabeceo " +
                        "recientemente."
                ContextStyleVariant.B ->
                    "volvió a registrarse un cierre ocular prolongado, " +
                        "junto con bostezos y un cabeceo en los eventos " +
                        "recientes."
                ContextStyleVariant.C ->
                    "se repitió el cierre ocular prolongado y también se " +
                        "registraron bostezos y un cabeceo recientemente."
            }

            StructuredContextSelection(
                ContextTemplateId.RECENT_EVENTS_CHANGED_FROM_HISTORY,
                ContextPrimarySignal.YAWNS,
                ContextSecondarySignal.GAZE_AWAY
            ) -> when (variant) {
                ContextStyleVariant.A ->
                    "los bostezos se han repetido y también apareció una " +
                        "desviación prolongada de la mirada, un cambio " +
                        "respecto a las señales leves previas."
                ContextStyleVariant.B ->
                    "ahora se registran bostezos repetidos y una desviación " +
                        "prolongada de la mirada, a diferencia de las " +
                        "señales leves previas."
                ContextStyleVariant.C ->
                    "los eventos recientes incluyen bostezos repetidos y " +
                        "una desviación prolongada de la mirada, distintos " +
                        "al historial leve previo."
            }

            StructuredContextSelection(
                ContextTemplateId.RECOVERY_AFTER_CRITICAL,
                ContextPrimarySignal.RECOVERY,
                ContextSecondarySignal.NONE
            ) -> when (variant) {
                ContextStyleVariant.A ->
                    "las señales se mantienen en recuperación después de " +
                        "un episodio crítico previo, sin nuevos bostezos " +
                        "ni cabeceos."
                ContextStyleVariant.B ->
                    "se observa una recuperación sostenida tras el episodio " +
                        "crítico previo, sin nuevos bostezos ni cabeceos."
                ContextStyleVariant.C ->
                    "la recuperación continúa después del episodio crítico " +
                        "anterior y no se han registrado nuevos bostezos " +
                        "ni cabeceos."
            }

            else -> return null
        }

        val safeName = displayName
            ?.trim()
            ?.takeIf { it.isNotEmpty() }
            ?.take(40)

        return if (safeName == null) {
            body.replaceFirstChar { it.uppercase() }
        } else {
            "$safeName, $body"
        }
    }

    fun parseWireSelection(
        template: String,
        primary: String,
        secondary: String
    ): StructuredContextSelection? {
        val parsedTemplate = ContextTemplateId.values()
            .firstOrNull { it.wireValue == template }
            ?: return null
        val parsedPrimary = ContextPrimarySignal.values()
            .firstOrNull { it.wireValue == primary }
            ?: return null
        val parsedSecondary = ContextSecondarySignal.values()
            .firstOrNull { it.wireValue == secondary }
            ?: return null

        return StructuredContextSelection(
            template = parsedTemplate,
            primary = parsedPrimary,
            secondary = parsedSecondary
        )
    }
}
