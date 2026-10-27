/*
 * Explainable fatigue fusion and temporal state machine.
 *
 * This is a direct Kotlin port of the deterministic desktop logic.
 * States are operational alert levels, not medical diagnoses.
 */
package com.google.mediapipe.examples.facelandmarker

import java.util.ArrayDeque

enum class AblationMode(
    val code: String,
    val enabledCues: Set<String>
) {
    A_EAR(
        "A",
        setOf("closure")
    ),
    B_EAR_MAR(
        "B",
        setOf(
            "closure",
            "yawn"
        )
    ),
    C_EAR_MAR_PERCLOS(
        "C",
        setOf(
            "closure",
            "yawn",
            "perclos"
        )
    ),
    D_PLUS_GAZE(
        "D",
        setOf(
            "closure",
            "yawn",
            "perclos",
            "gaze"
        )
    ),
    E_FULL(
        "E",
        setOf(
            "closure",
            "yawn",
            "perclos",
            "gaze",
            "nod"
        )
    );

    companion object {
        fun fromCode(raw: String?): AblationMode {
            val normalized =
                raw?.trim()?.uppercase()
            return values().firstOrNull {
                it.code == normalized
            } ?: E_FULL
        }
    }
}

data class FusionEvidence(
    val closure: Double,
    val perclos: Double,
    val yawn: Double,
    val gaze: Double,
    val nod: Double
) {
    fun asMap(): Map<String, Double> {
        return mapOf(
            "closure" to closure,
            "perclos" to perclos,
            "yawn" to yawn,
            "gaze" to gaze,
            "nod" to nod
        )
    }
}

data class FusionResult(
    val score: Double,
    val evidence: FusionEvidence,
    val reasons: List<String>
)

class FatigueFusion(
    val weights: Map<String, Double> =
        DEFAULT_WEIGHTS,
    val mode: AblationMode =
        AblationMode.E_FULL
) {
    init {
        require(CUE_NAMES.all { it in weights })
        require(weights.values.all { it >= 0.0 })
        require(weights.values.sum() > 0.0)
        require(
            mode.enabledCues.isNotEmpty() &&
                mode.enabledCues.all {
                    it in CUE_NAMES
                }
        )
        require(
            mode.enabledCues.sumOf {
                checkNotNull(weights[it])
            } > 0.0
        )
    }

    fun evaluate(
        closureS: Double,
        perclosRatio: Double,
        yawnsLast5Min: Int,
        gazeAwayS: Double,
        nodsLast2Min: Int
    ): FusionResult {
        val evidence = FusionEvidence(
            closure = ramp(closureS, 0.40, 2.00),
            perclos = ramp(
                perclosRatio,
                0.15,
                0.40
            ),
            yawn = ramp(
                yawnsLast5Min.toDouble(),
                0.0,
                3.0
            ),
            gaze = ramp(
                gazeAwayS,
                1.50,
                5.00
            ),
            nod = ramp(
                nodsLast2Min.toDouble(),
                0.0,
                3.0
            )
        )

        val values = evidence.asMap()
        val activeCues = CUE_NAMES.filter {
            it in mode.enabledCues
        }
        val denominator = activeCues.sumOf {
            checkNotNull(weights[it])
        }
        val numerator = activeCues.sumOf {
            checkNotNull(weights[it]) *
                checkNotNull(values[it])
        }
        val score =
            (numerator / denominator)
                .coerceIn(0.0, 1.0)
        val reasons = activeCues
            .filter { checkNotNull(values[it]) > 0.0 }
            .sortedByDescending {
                checkNotNull(weights[it]) *
                    checkNotNull(values[it])
            }

        return FusionResult(
            score = score,
            evidence = evidence,
            reasons = reasons
        )
    }

    companion object {
        val CUE_NAMES = listOf(
            "closure",
            "perclos",
            "yawn",
            "gaze",
            "nod"
        )

        val DEFAULT_WEIGHTS = mapOf(
            "closure" to 0.30,
            "perclos" to 0.25,
            "yawn" to 0.15,
            "gaze" to 0.10,
            "nod" to 0.20
        )

        fun ramp(
            value: Double,
            low: Double,
            high: Double
        ): Double {
            require(value.isFinite())
            require(low.isFinite())
            require(high.isFinite())
            require(high > low)

            return when {
                value <= low -> 0.0
                value >= high -> 1.0
                else -> (value - low) / (high - low)
            }
        }
    }
}

class RollingEventCounter(
    windowS: Double
) {
    private val windowMs =
        (windowS * 1000.0).toLong()
    private val eventTimestamps =
        ArrayDeque<Long>()
    private var lastTimestampMs: Long? = null
    private var previousEvent = false

    init {
        require(windowS.isFinite() && windowS > 0.0)
    }

    fun reset() {
        eventTimestamps.clear()
        lastTimestampMs = null
        previousEvent = false
    }

    fun update(
        timestampMs: Long,
        event: Boolean,
        valid: Boolean
    ): Int {
        validateTimestamp(timestampMs)
        val cutoff = timestampMs - windowMs

        while (
            eventTimestamps.isNotEmpty() &&
            eventTimestamps.peekFirst() < cutoff
        ) {
            eventTimestamps.removeFirst()
        }

        if (!valid) {
            previousEvent = false
            return eventTimestamps.size
        }

        if (event && !previousEvent) {
            eventTimestamps.addLast(timestampMs)
        }

        previousEvent = event
        return eventTimestamps.size
    }

    private fun validateTimestamp(timestampMs: Long) {
        require(timestampMs >= 0L)
        val previous = lastTimestampMs
        require(previous == null || timestampMs >= previous)
        lastTimestampMs = timestampMs
    }
}

data class RecentEventMetrics(
    val yawnsLast5Min: Int,
    val nodsLast2Min: Int
)

class RecentEventWindows {
    private val yawns =
        RollingEventCounter(300.0)
    private val nods =
        RollingEventCounter(120.0)

    fun reset() {
        yawns.reset()
        nods.reset()
    }

    fun update(
        timestampMs: Long,
        yawnEvent: Boolean,
        nodEvent: Boolean,
        yawnValid: Boolean,
        nodValid: Boolean
    ): RecentEventMetrics {
        return RecentEventMetrics(
            yawnsLast5Min = yawns.update(
                timestampMs,
                yawnEvent,
                yawnValid
            ),
            nodsLast2Min = nods.update(
                timestampMs,
                nodEvent,
                nodValid
            )
        )
    }
}

enum class FatigueState(
    val visibleLabel: String
) {
    NORMAL("NORMAL"),
    SUSPECTED("FATIGA LEVE"),
    FATIGUE("FATIGA MODERADA"),
    CRITICAL("FATIGA GRAVE"),
    RECOVERY("RECUPERACIÓN")
}

data class StateResult(
    val state: FatigueState,
    val event: String?
)

class FatigueStateMachine {
    var state = FatigueState.NORMAL
        private set

    private var candidateKey: String? = null
    private var candidateSinceMs: Long? = null
    private var lastTimestampMs: Long? = null
    private var criticalAcknowledged = false

    val alertAcknowledged: Boolean
        get() = (
            state == FatigueState.CRITICAL &&
                criticalAcknowledged
            )

    val criticalRecoveryElapsedS: Double
        get() {
            if (
                !alertAcknowledged ||
                candidateKey !=
                "critical_to_recovery"
            ) {
                return 0.0
            }

            val start = candidateSinceMs
                ?: return 0.0
            val last = lastTimestampMs
                ?: return 0.0

            return maxOf(
                0.0,
                (last - start) / 1000.0
            )
        }

    fun reset() {
        state = FatigueState.NORMAL
        candidateKey = null
        candidateSinceMs = null
        lastTimestampMs = null
        criticalAcknowledged = false
    }

    fun invalidate(timestampMs: Long): StateResult {
        validateTimestamp(timestampMs)
        clearCandidate()
        return StateResult(state, null)
    }

    fun update(
        timestampMs: Long,
        score: Double,
        closureS: Double,
        acknowledged: Boolean = false
    ): StateResult {
        validateTimestamp(timestampMs)
        require(
            score.isFinite() &&
                score in 0.0..1.0
        )
        require(
            closureS.isFinite() &&
                closureS >= 0.0
        )

        if (closureS >= 3.0) {
            if (state != FatigueState.CRITICAL) {
                return enter(
                    FatigueState.CRITICAL,
                    "prolonged_eye_closure"
                )
            }

            criticalAcknowledged = false
            clearCandidate()
            return StateResult(state, null)
        }

        if (
            state != FatigueState.NORMAL &&
            score < 0.20
        ) {
            return persistTransition(
                "any_to_normal",
                FatigueState.NORMAL,
                5_000L,
                timestampMs,
                "state_normal"
            )
        }

        if (state == FatigueState.CRITICAL) {
            var acknowledgementEvent: String? = null

            if (
                acknowledged &&
                !criticalAcknowledged
            ) {
                acknowledgementEvent =
                    "critical_acknowledged"
            }

            if (acknowledged) {
                criticalAcknowledged = true
            }

            if (!criticalAcknowledged) {
                clearCandidate()
                return StateResult(state, null)
            }

            if (score < 0.40) {
                val transition = persistTransition(
                    "critical_to_recovery",
                    FatigueState.RECOVERY,
                    30_000L,
                    timestampMs,
                    "state_recovery"
                )

                return if (transition.event != null) {
                    transition
                } else {
                    StateResult(
                        state,
                        acknowledgementEvent
                    )
                }
            }

            clearCandidate()
            return StateResult(
                state,
                acknowledgementEvent
            )
        }

        if (
            score >= 0.75 &&
            state == FatigueState.FATIGUE
        ) {
            return persistTransition(
                "to_critical",
                FatigueState.CRITICAL,
                10_000L,
                timestampMs,
                "state_critical"
            )
        }

        if (
            score >= 0.35 &&
            state == FatigueState.SUSPECTED
        ) {
            return persistTransition(
                "to_fatigue",
                FatigueState.FATIGUE,
                3_000L,
                timestampMs,
                "state_fatigue"
            )
        }

        if (
            score >= 0.20 &&
            state in setOf(
                FatigueState.NORMAL,
                FatigueState.RECOVERY
            )
        ) {
            return persistTransition(
                "to_suspected",
                FatigueState.SUSPECTED,
                1_500L,
                timestampMs,
                "state_suspected"
            )
        }

        if (
            state == FatigueState.SUSPECTED &&
            score < 0.20
        ) {
            return persistTransition(
                "suspected_to_normal",
                FatigueState.NORMAL,
                10_000L,
                timestampMs,
                "state_normal"
            )
        }

        if (
            state == FatigueState.FATIGUE &&
            score < 0.35
        ) {
            return persistTransition(
                "fatigue_to_recovery",
                FatigueState.RECOVERY,
                15_000L,
                timestampMs,
                "state_recovery"
            )
        }

        if (
            state == FatigueState.RECOVERY &&
            score < 0.20
        ) {
            return persistTransition(
                "recovery_to_normal",
                FatigueState.NORMAL,
                10_000L,
                timestampMs,
                "state_normal"
            )
        }

        clearCandidate()
        return StateResult(state, null)
    }

    private fun validateTimestamp(
        timestampMs: Long
    ) {
        require(timestampMs >= 0L)
        val previous = lastTimestampMs
        require(previous == null || timestampMs >= previous)
        lastTimestampMs = timestampMs
    }

    private fun clearCandidate() {
        candidateKey = null
        candidateSinceMs = null
    }

    private fun candidateElapsed(
        key: String,
        dwellMs: Long,
        timestampMs: Long
    ): Boolean {
        if (candidateKey != key) {
            candidateKey = key
            candidateSinceMs = timestampMs
            return dwellMs == 0L
        }

        return timestampMs -
            checkNotNull(candidateSinceMs) >= dwellMs
    }

    private fun enter(
        target: FatigueState,
        event: String
    ): StateResult {
        if (state == target) {
            clearCandidate()
            return StateResult(state, null)
        }

        state = target
        clearCandidate()

        if (
            target == FatigueState.CRITICAL ||
            target == FatigueState.NORMAL
        ) {
            criticalAcknowledged = false
        }

        return StateResult(state, event)
    }

    private fun persistTransition(
        key: String,
        target: FatigueState,
        dwellMs: Long,
        timestampMs: Long,
        event: String
    ): StateResult {
        return if (
            candidateElapsed(
                key,
                dwellMs,
                timestampMs
            )
        ) {
            enter(target, event)
        } else {
            StateResult(state, null)
        }
    }
}

data class FatigueEngineMetrics(
    val score: Double,
    val evidence: FusionEvidence,
    val reasons: List<String>,
    val state: FatigueState,
    val stateEvent: String?,
    val yawnsLast5Min: Int,
    val nodsLast2Min: Int,
    val dataSufficient: Boolean,
    val availableWeightRatio: Double,
    val ablationMode: AblationMode,
    val alertAcknowledged: Boolean,
    val criticalRecoveryElapsedS: Double
)

class FatigueEngine(
    private val minimumAvailableWeightRatio:
        Double = 0.55,
    private val ablationMode:
        AblationMode = AblationMode.E_FULL
) {
    private val fusion = FatigueFusion(
        mode = ablationMode
    )
    private val events = RecentEventWindows()
    private val stateMachine =
        FatigueStateMachine()

    init {
        require(
            minimumAvailableWeightRatio
                in 0.0..1.0
        )
    }

    @Synchronized
    fun reset() {
        events.reset()
        stateMachine.reset()
    }

    @Synchronized
    fun update(
        timestampMs: Long,
        closureS: Double,
        perclosRatio: Double,
        gazeAwayS: Double,
        yawnEvent: Boolean,
        nodEvent: Boolean,
        faceValid: Boolean,
        eyeValid: Boolean,
        mouthValid: Boolean,
        gazeValid: Boolean,
        poseValid: Boolean,
        acknowledged: Boolean = false
    ): FatigueEngineMetrics {
        require(timestampMs >= 0L)

        val recent = events.update(
            timestampMs = timestampMs,
            yawnEvent = yawnEvent,
            nodEvent = nodEvent,
            yawnValid = faceValid && mouthValid,
            nodValid = faceValid && poseValid
        )

        val safeClosure = safeNonnegative(
            closureS,
            faceValid && eyeValid
        )
        val safePerclos = safeNonnegative(
            perclosRatio,
            faceValid && eyeValid
        )
        val safeGaze = safeNonnegative(
            gazeAwayS,
            faceValid && gazeValid
        )

        val fusionResult = fusion.evaluate(
            closureS = safeClosure,
            perclosRatio = safePerclos,
            yawnsLast5Min =
                recent.yawnsLast5Min,
            gazeAwayS = safeGaze,
            nodsLast2Min =
                recent.nodsLast2Min
        )

        val availability = availability(
            faceValid = faceValid,
            eyeValid = eyeValid,
            mouthValid = mouthValid,
            gazeValid = gazeValid,
            poseValid = poseValid
        )
        val sufficient =
            faceValid &&
                eyeValid &&
                availability >=
                minimumAvailableWeightRatio

        val stateResult =
            if (sufficient) {
                stateMachine.update(
                    timestampMs,
                    fusionResult.score,
                    safeClosure,
                    acknowledged
                )
            } else {
                stateMachine.invalidate(timestampMs)
            }

        if (stateResult.event == "state_normal") {
            events.reset()
        }

        return FatigueEngineMetrics(
            score = fusionResult.score,
            evidence = fusionResult.evidence,
            reasons = fusionResult.reasons,
            state = stateResult.state,
            stateEvent = stateResult.event,
            yawnsLast5Min =
                recent.yawnsLast5Min,
            nodsLast2Min =
                recent.nodsLast2Min,
            dataSufficient = sufficient,
            availableWeightRatio = availability,
            ablationMode = ablationMode,
            alertAcknowledged =
                stateMachine.alertAcknowledged,
            criticalRecoveryElapsedS =
                stateMachine
                    .criticalRecoveryElapsedS
        )
    }

    private fun safeNonnegative(
        value: Double,
        valid: Boolean
    ): Double {
        if (!valid) {
            return 0.0
        }

        require(
            value.isFinite() &&
                value >= 0.0
        )
        return value
    }

    private fun availability(
        faceValid: Boolean,
        eyeValid: Boolean,
        mouthValid: Boolean,
        gazeValid: Boolean,
        poseValid: Boolean
    ): Double {
        val available = mapOf(
            "closure" to
                (faceValid && eyeValid),
            "perclos" to
                (faceValid && eyeValid),
            "yawn" to
                (faceValid && mouthValid),
            "gaze" to
                (faceValid && gazeValid),
            "nod" to
                (faceValid && poseValid)
        )
        val activeCues =
            FatigueFusion.CUE_NAMES.filter {
                it in ablationMode.enabledCues
            }

        val denominator =
            activeCues.sumOf {
                checkNotNull(
                    FatigueFusion
                        .DEFAULT_WEIGHTS[it]
                )
            }
        val numerator =
            activeCues
                .filter {
                    checkNotNull(available[it])
                }
                .sumOf {
                    checkNotNull(
                        FatigueFusion
                            .DEFAULT_WEIGHTS[it]
                    )
                }

        return (
            numerator / denominator
            ).let {
                // Avoid 0.9000000000000001 in
                // exported/debug values.
                kotlin.math.round(it * 1e12) / 1e12
            }
    }
}
