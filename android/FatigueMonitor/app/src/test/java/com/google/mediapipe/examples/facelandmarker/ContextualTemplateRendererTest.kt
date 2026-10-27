package com.google.mediapipe.examples.facelandmarker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ContextualTemplateRendererTest {

    @Test
    fun rendersRepeatedEyeClosureWithRecentEvents() {
        val result = ContextualTemplateRenderer.render(
            selection = StructuredContextSelection(
                template =
                    ContextTemplateId.REPEATED_PRIMARY_WITH_RECENT_EVENTS,
                primary = ContextPrimarySignal.EYE_CLOSURE,
                secondary = ContextSecondarySignal.YAWNS_AND_NOD
            ),
            displayName = "Joaquín"
        )

        assertEquals(
            "Joaquín, el cierre ocular prolongado volvió a presentarse; " +
                "además, se registraron bostezos y un cabeceo " +
                "recientemente.",
            result
        )
    }

    @Test
    fun rendersChangedHistoryScenario() {
        val result = ContextualTemplateRenderer.render(
            selection = StructuredContextSelection(
                template =
                    ContextTemplateId.RECENT_EVENTS_CHANGED_FROM_HISTORY,
                primary = ContextPrimarySignal.YAWNS,
                secondary = ContextSecondarySignal.GAZE_AWAY
            ),
            displayName = "Joaquín"
        )

        assertEquals(
            "Joaquín, los bostezos se han repetido y también apareció una " +
                "desviación prolongada de la mirada, un cambio respecto " +
                "a las señales leves previas.",
            result
        )
    }

    @Test
    fun rendersRecoveryAfterCriticalScenario() {
        val result = ContextualTemplateRenderer.render(
            selection = StructuredContextSelection(
                template =
                    ContextTemplateId.RECOVERY_AFTER_CRITICAL,
                primary = ContextPrimarySignal.RECOVERY,
                secondary = ContextSecondarySignal.NONE
            ),
            displayName = "Joaquín"
        )

        assertEquals(
            "Joaquín, las señales se mantienen en recuperación después de " +
                "un episodio crítico previo, sin nuevos bostezos ni " +
                "cabeceos.",
            result
        )
    }

    @Test
    fun rendersApprovedStyleVariantWithoutChangingSemantics() {
        val selection = StructuredContextSelection(
            template =
                ContextTemplateId.RECOVERY_AFTER_CRITICAL,
            primary = ContextPrimarySignal.RECOVERY,
            secondary = ContextSecondarySignal.NONE
        )

        val result = ContextualTemplateRenderer.render(
            selection = selection,
            displayName = "Joaquín",
            variant = ContextStyleVariant.B
        )

        assertEquals(
            "Joaquín, se observa una recuperación sostenida tras el episodio " +
                "crítico previo, sin nuevos bostezos ni cabeceos.",
            result
        )
    }

    @Test
    fun rejectsUnsupportedCombination() {
        val result = ContextualTemplateRenderer.render(
            selection = StructuredContextSelection(
                template =
                    ContextTemplateId.REPEATED_PRIMARY_WITH_RECENT_EVENTS,
                primary = ContextPrimarySignal.RECOVERY,
                secondary = ContextSecondarySignal.GAZE_AWAY
            ),
            displayName = "Joaquín"
        )

        assertNull(result)
    }

    @Test
    fun parsesOnlyKnownWireValues() {
        val parsed = ContextualTemplateRenderer.parseWireSelection(
            template = "repeated_primary_with_recent_events",
            primary = "eye_closure",
            secondary = "yawns_and_nod"
        )

        assertEquals(
            StructuredContextSelection(
                template =
                    ContextTemplateId.REPEATED_PRIMARY_WITH_RECENT_EVENTS,
                primary = ContextPrimarySignal.EYE_CLOSURE,
                secondary = ContextSecondarySignal.YAWNS_AND_NOD
            ),
            parsed
        )

        assertNull(
            ContextualTemplateRenderer.parseWireSelection(
                template = "invented_template",
                primary = "eye_closure",
                secondary = "yawns_and_nod"
            )
        )
    }
}
