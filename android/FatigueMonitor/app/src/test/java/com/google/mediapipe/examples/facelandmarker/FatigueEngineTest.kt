package com.google.mediapipe.examples.facelandmarker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FatigueEngineTest {

    @Test
    fun rampClampsAndInterpolates() {
        assertEquals(
            0.0,
            FatigueFusion.ramp(0.1, 0.2, 0.4),
            1e-9
        )
        assertEquals(
            0.5,
            FatigueFusion.ramp(0.3, 0.2, 0.4),
            1e-9
        )
        assertEquals(
            1.0,
            FatigueFusion.ramp(0.5, 0.2, 0.4),
            1e-9
        )
    }

    @Test
    fun closureEvidenceUsesConfiguredWeight() {
        val result = FatigueFusion().evaluate(
            closureS = 2.0,
            perclosRatio = 0.0,
            yawnsLast5Min = 0,
            gazeAwayS = 0.0,
            nodsLast2Min = 0
        )

        assertEquals(1.0, result.evidence.closure, 1e-9)
        assertEquals(0.30, result.score, 1e-9)
        assertEquals(listOf("closure"), result.reasons)
    }

    @Test
    fun ablationAUsesOnlyClosureAndRenormalizesScore() {
        val ignoredCues = FatigueFusion(
            mode = AblationMode.A_EAR
        ).evaluate(
            closureS = 0.0,
            perclosRatio = 0.40,
            yawnsLast5Min = 3,
            gazeAwayS = 5.0,
            nodsLast2Min = 3
        )

        assertEquals(0.0, ignoredCues.score, 1e-9)
        assertTrue(ignoredCues.reasons.isEmpty())

        val closureOnly = FatigueFusion(
            mode = AblationMode.A_EAR
        ).evaluate(
            closureS = 2.0,
            perclosRatio = 0.0,
            yawnsLast5Min = 0,
            gazeAwayS = 0.0,
            nodsLast2Min = 0
        )

        assertEquals(1.0, closureOnly.score, 1e-9)
        assertEquals(
            listOf("closure"),
            closureOnly.reasons
        )
    }

    @Test
    fun ablationModesAddCuesIncrementally() {
        assertEquals(
            setOf("closure"),
            AblationMode.A_EAR.enabledCues
        )
        assertEquals(
            setOf("closure", "yawn"),
            AblationMode.B_EAR_MAR.enabledCues
        )
        assertEquals(
            setOf(
                "closure",
                "yawn",
                "perclos"
            ),
            AblationMode
                .C_EAR_MAR_PERCLOS
                .enabledCues
        )
        assertEquals(
            setOf(
                "closure",
                "yawn",
                "perclos",
                "gaze"
            ),
            AblationMode.D_PLUS_GAZE.enabledCues
        )
        assertEquals(
            setOf(
                "closure",
                "yawn",
                "perclos",
                "gaze",
                "nod"
            ),
            AblationMode.E_FULL.enabledCues
        )
    }

    @Test
    fun ablationAvailabilityUsesOnlyEnabledCues() {
        val engine = FatigueEngine(
            ablationMode = AblationMode.A_EAR
        )

        val result = engine.update(
            timestampMs = 1000L,
            closureS = 0.0,
            perclosRatio = 0.0,
            gazeAwayS = Double.NaN,
            yawnEvent = false,
            nodEvent = false,
            faceValid = true,
            eyeValid = true,
            mouthValid = false,
            gazeValid = false,
            poseValid = false
        )

        assertTrue(result.dataSufficient)
        assertEquals(
            1.0,
            result.availableWeightRatio,
            1e-12
        )
        assertEquals(
            AblationMode.A_EAR,
            result.ablationMode
        )
    }

    @Test
    fun ablationModeParserAcceptsCodesAndDefaultsToFull() {
        assertEquals(
            AblationMode.A_EAR,
            AblationMode.fromCode("A")
        )
        assertEquals(
            AblationMode.B_EAR_MAR,
            AblationMode.fromCode(" b ")
        )
        assertEquals(
            AblationMode.C_EAR_MAR_PERCLOS,
            AblationMode.fromCode("C")
        )
        assertEquals(
            AblationMode.D_PLUS_GAZE,
            AblationMode.fromCode("d")
        )
        assertEquals(
            AblationMode.E_FULL,
            AblationMode.fromCode("E")
        )
        assertEquals(
            AblationMode.E_FULL,
            AblationMode.fromCode(null)
        )
        assertEquals(
            AblationMode.E_FULL,
            AblationMode.fromCode("invalid")
        )
    }

    @Test
    fun sustainedScoreEntersSuspected() {
        val machine = FatigueStateMachine()

        machine.update(0L, 0.30, 0.0)
        val result =
            machine.update(5000L, 0.30, 0.0)

        assertEquals(
            FatigueState.SUSPECTED,
            result.state
        )
        assertEquals(
            "state_suspected",
            result.event
        )
    }

    @Test
    fun invalidFrameCancelsPendingTransition() {
        val machine = FatigueStateMachine()

        machine.update(0L, 0.30, 0.0)
        machine.update(1000L, 0.30, 0.0)
        machine.invalidate(1100L)
        val result =
            machine.update(2000L, 0.30, 0.0)

        assertEquals(FatigueState.NORMAL, result.state)
        assertNull(result.event)
    }

    @Test
    fun prolongedClosureEntersCriticalImmediately() {
        val machine = FatigueStateMachine()

        val result =
            machine.update(0L, 0.0, 3.0)

        assertEquals(
            FatigueState.CRITICAL,
            result.state
        )
        assertEquals(
            "prolonged_eye_closure",
            result.event
        )
    }

    @Test
    fun criticalRequiresAcknowledgementAndRecoveryTime() {
        val machine = FatigueStateMachine()

        machine.update(0L, 0.0, 3.0)
        machine.update(
            100L,
            0.30,
            0.0,
            acknowledged = true
        )
        val before =
            machine.update(30_000L, 0.30, 0.0)
        val recovered =
            machine.update(30_100L, 0.30, 0.0)

        assertEquals(
            FatigueState.CRITICAL,
            before.state
        )
        assertEquals(
            FatigueState.RECOVERY,
            recovered.state
        )
    }

    @Test
    fun invalidFaceCannotAddEventsOrChangeState() {
        val engine = FatigueEngine()

        val result = engine.update(
            timestampMs = 1000L,
            closureS = Double.NaN,
            perclosRatio = Double.NaN,
            gazeAwayS = Double.NaN,
            yawnEvent = true,
            nodEvent = true,
            faceValid = false,
            eyeValid = false,
            mouthValid = false,
            gazeValid = false,
            poseValid = false
        )

        assertEquals(FatigueState.NORMAL, result.state)
        assertEquals(0, result.yawnsLast5Min)
        assertEquals(0, result.nodsLast2Min)
        assertFalse(result.dataSufficient)
    }

    @Test
    fun invalidGazeRetainsEnoughAvailability() {
        val engine = FatigueEngine()

        val result = engine.update(
            timestampMs = 1000L,
            closureS = 0.0,
            perclosRatio = 0.0,
            gazeAwayS = Double.NaN,
            yawnEvent = false,
            nodEvent = false,
            faceValid = true,
            eyeValid = true,
            mouthValid = true,
            gazeValid = false,
            poseValid = true
        )

        assertTrue(result.dataSufficient)
        assertEquals(
            0.90,
            result.availableWeightRatio,
            1e-12
        )
    }
}
