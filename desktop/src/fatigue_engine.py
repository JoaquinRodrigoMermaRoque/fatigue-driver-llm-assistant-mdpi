"""Coordinador de eventos, fusión y máquina temporal de estados."""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

try:
    # Importación cuando se ejecuta como paquete: python -m src.check_cues
    from .fusion import (
        CUE_NAMES,
        DEFAULT_WEIGHTS,
        FatigueFusion,
        FusionEvidence,
    )
    from .recent_events import RecentEventWindows
    from .state_machine import (
        FatigueStateMachine,
        State,
        VISIBLE_LABELS,
    )
except ImportError:
    # Compatibilidad con la ejecución usada en el prototipo:
    # python src/check_cues.py
    from fusion import (
        CUE_NAMES,
        DEFAULT_WEIGHTS,
        FatigueFusion,
        FusionEvidence,
    )
    from recent_events import RecentEventWindows
    from state_machine import (
        FatigueStateMachine,
        State,
        VISIBLE_LABELS,
    )


@dataclass(frozen=True)
class FatigueEngineMetrics:
    """Salida completa y explicable del motor multimodal."""

    score: float
    evidence: FusionEvidence
    reasons: tuple[str, ...]
    state: State
    state_label: str
    state_event: str | None
    yawns_last_5_min: int
    nods_last_2_min: int
    data_sufficient: bool
    available_weight_ratio: float
    alert_acknowledged: bool
    critical_recovery_elapsed_s: float
    critical_recovery_required_s: float


class FatigueEngine:
    """Ejecuta toda la lógica de H5 sin depender de cámara o interfaz."""

    def __init__(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        active_cues: Sequence[str] = CUE_NAMES,
        minimum_available_weight_ratio: float = 0.55,
    ):
        self.fusion = FatigueFusion(
            weights=DEFAULT_WEIGHTS if weights is None else weights,
            active_cues=active_cues,
        )
        self.events = RecentEventWindows()
        self.state_machine = FatigueStateMachine()

        minimum = float(minimum_available_weight_ratio)

        if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
            raise ValueError(
                "minimum_available_weight_ratio debe estar entre 0 y 1"
            )

        self.minimum_available_weight_ratio = minimum

    def reset(self) -> None:
        self.events.reset()
        self.state_machine.reset()

    @staticmethod
    def _safe_nonnegative(
        value: float,
        *,
        name: str,
        valid: bool,
    ) -> float:
        if not valid:
            return 0.0

        number = float(value)

        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name} debe ser finito y no negativo")

        return number

    def _availability(
        self,
        *,
        face_valid: bool,
        eye_valid: bool,
        gaze_valid: bool,
    ) -> tuple[float, bool]:
        """Calcula qué proporción del peso puede evaluarse con seguridad."""

        available = {
            "closure": bool(face_valid and eye_valid),
            "perclos": bool(face_valid and eye_valid),
            # Bostezos y cabeceos representan eventos históricos válidos.
            "yawn": bool(face_valid),
            "nod": bool(face_valid),
            "gaze": bool(face_valid and gaze_valid),
        }

        denominator = sum(
            self.fusion.weights[name]
            for name in self.fusion.active_cues
        )
        numerator = sum(
            self.fusion.weights[name]
            for name in self.fusion.active_cues
            if available[name]
        )

        ratio = numerator / denominator
        sufficient = (
            bool(face_valid)
            and bool(eye_valid)
            and ratio >= self.minimum_available_weight_ratio
        )

        return ratio, sufficient

    def update(
        self,
        *,
        timestamp_s: float,
        closure_s: float,
        perclos_ratio: float,
        gaze_away_s: float,
        yawn_event: bool,
        nod_event: bool,
        face_valid: bool,
        eye_valid: bool,
        mouth_valid: bool,
        gaze_valid: bool,
        pose_valid: bool,
        acknowledged: bool = False,
    ) -> FatigueEngineMetrics:
        timestamp = float(timestamp_s)

        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(
                "timestamp_s debe ser finito y no negativo"
            )

        face_valid = bool(face_valid)
        eye_valid = bool(eye_valid)
        mouth_valid = bool(mouth_valid)
        gaze_valid = bool(gaze_valid)
        pose_valid = bool(pose_valid)

        recent = self.events.update(
            timestamp_s=timestamp,
            yawn_event=bool(yawn_event),
            nod_event=bool(nod_event),
            yawn_valid=face_valid and mouth_valid,
            nod_valid=face_valid and pose_valid,
        )

        safe_closure = self._safe_nonnegative(
            closure_s,
            name="closure_s",
            valid=face_valid and eye_valid,
        )
        safe_perclos = self._safe_nonnegative(
            perclos_ratio,
            name="perclos_ratio",
            valid=face_valid and eye_valid,
        )
        safe_gaze = self._safe_nonnegative(
            gaze_away_s,
            name="gaze_away_s",
            valid=face_valid and gaze_valid,
        )

        fusion_result = self.fusion.evaluate(
            max_closure_s=safe_closure,
            perclos_ratio=safe_perclos,
            yawns_last_5_min=recent.yawns_last_5_min,
            gaze_away_s=safe_gaze,
            nods_last_2_min=recent.nods_last_2_min,
        )

        availability, sufficient = self._availability(
            face_valid=face_valid,
            eye_valid=eye_valid,
            gaze_valid=gaze_valid,
        )

        timestamp_ms = int(round(timestamp * 1000.0))

        if sufficient:
            state, state_event = self.state_machine.update(
                timestamp_ms,
                fusion_result.score,
                safe_closure,
                acknowledged=acknowledged,
            )
        else:
            state, state_event = self.state_machine.invalidate(
                timestamp_ms
            )

        return FatigueEngineMetrics(
            score=fusion_result.score,
            evidence=fusion_result.evidence,
            reasons=fusion_result.reasons,
            state=state,
            state_label=VISIBLE_LABELS[state],
            state_event=state_event,
            yawns_last_5_min=recent.yawns_last_5_min,
            nods_last_2_min=recent.nods_last_2_min,
            data_sufficient=sufficient,
            available_weight_ratio=availability,
            alert_acknowledged=(
                self.state_machine.alert_acknowledged
            ),
            critical_recovery_elapsed_s=(
                self.state_machine.critical_recovery_elapsed_s
            ),
            critical_recovery_required_s=30.0,
        )
