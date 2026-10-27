"""Fusión reproducible de indicadores faciales de fatiga.

Este módulo no decide todavía el estado temporal de alerta. Convierte cada
indicador en evidencia normalizada y calcula un puntaje comparable.
"""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


CUE_NAMES = ("closure", "perclos", "yawn", "gaze", "nod")

DEFAULT_WEIGHTS = {
    "closure": 0.30,
    "perclos": 0.25,
    "yawn": 0.15,
    "gaze": 0.10,
    "nod": 0.20,
}


@dataclass(frozen=True)
class FusionEvidence:
    """Evidencias individuales, todas limitadas al intervalo 0-1."""

    closure: float
    perclos: float
    yawn: float
    gaze: float
    nod: float

    def as_dict(self) -> dict[str, float]:
        return {
            "closure": self.closure,
            "perclos": self.perclos,
            "yawn": self.yawn,
            "gaze": self.gaze,
            "nod": self.nod,
        }


@dataclass(frozen=True)
class FusionResult:
    """Resultado instantáneo de la fusión, previo a la máquina de estados."""

    score: float
    evidence: FusionEvidence
    active_cues: tuple[str, ...]
    reasons: tuple[str, ...]


def _finite(value: float, name: str) -> float:
    """Convierte a float y rechaza valores no finitos."""

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{name} debe ser un número finito")

    return number


def ramp(value: float, low: float, high: float) -> float:
    """Convierte linealmente un valor al intervalo cerrado 0-1."""

    value = _finite(value, "value")
    low = _finite(low, "low")
    high = _finite(high, "high")

    if high <= low:
        raise ValueError("high debe ser mayor que low")

    if value <= low:
        return 0.0

    if value >= high:
        return 1.0

    return (value - low) / (high - low)


def compute_evidence(
    *,
    max_closure_s: float,
    perclos_ratio: float,
    yawns_last_5_min: int,
    gaze_away_s: float,
    nods_last_2_min: int,
) -> FusionEvidence:
    """Convierte los indicadores actuales en evidencias normalizadas.

    ``perclos_ratio`` se expresa entre 0 y 1. Por ejemplo, 25 % se pasa
    como 0.25, no como 25.
    """

    return FusionEvidence(
        closure=ramp(max_closure_s, 0.40, 2.00),
        perclos=ramp(perclos_ratio, 0.15, 0.40),
        yawn=ramp(yawns_last_5_min, 0.0, 3.0),
        gaze=ramp(gaze_away_s, 1.50, 5.00),
        nod=ramp(nods_last_2_min, 0.0, 3.0),
    )


def normalized_score(
    evidence: Mapping[str, float],
    weights: Mapping[str, float],
    active_cues: Sequence[str],
) -> float:
    """Calcula el puntaje usando solamente el peso de señales activas."""

    active = tuple(active_cues)

    if not active:
        raise ValueError("Debe existir al menos una señal activa")

    unknown = sorted(set(active) - set(CUE_NAMES))

    if unknown:
        raise ValueError(f"Señales desconocidas: {unknown}")

    missing = [
        name
        for name in active
        if name not in evidence or name not in weights
    ]

    if missing:
        raise ValueError(f"Faltan evidencias o pesos: {missing}")

    denominator = 0.0
    numerator = 0.0

    for name in active:
        cue_value = _finite(evidence[name], f"evidence[{name}]")
        cue_weight = _finite(weights[name], f"weights[{name}]")

        if not 0.0 <= cue_value <= 1.0:
            raise ValueError(
                f"La evidencia {name} debe estar entre 0 y 1"
            )

        if cue_weight < 0.0:
            raise ValueError(f"El peso {name} no puede ser negativo")

        numerator += cue_weight * cue_value
        denominator += cue_weight

    if denominator <= 0.0:
        raise ValueError("La suma de pesos activos debe ser positiva")

    return max(0.0, min(1.0, numerator / denominator))


class FatigueFusion:
    """Calcula evidencia y puntaje para un conjunto de señales activas."""

    def __init__(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        active_cues: Sequence[str] = CUE_NAMES,
    ):
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.active_cues = tuple(active_cues)

        # Valida la configuración al construir el objeto.
        normalized_score(
            {name: 0.0 for name in CUE_NAMES},
            self.weights,
            self.active_cues,
        )

    def evaluate(
        self,
        *,
        max_closure_s: float,
        perclos_ratio: float,
        yawns_last_5_min: int,
        gaze_away_s: float,
        nods_last_2_min: int,
    ) -> FusionResult:
        evidence = compute_evidence(
            max_closure_s=max_closure_s,
            perclos_ratio=perclos_ratio,
            yawns_last_5_min=yawns_last_5_min,
            gaze_away_s=gaze_away_s,
            nods_last_2_min=nods_last_2_min,
        )

        evidence_dict = evidence.as_dict()
        score = normalized_score(
            evidence_dict,
            self.weights,
            self.active_cues,
        )

        reasons = tuple(
            sorted(
                (
                    name
                    for name in self.active_cues
                    if evidence_dict[name] > 0.0
                ),
                key=lambda name: (
                    self.weights[name] * evidence_dict[name]
                ),
                reverse=True,
            )
        )

        return FusionResult(
            score=score,
            evidence=evidence,
            active_cues=self.active_cues,
            reasons=reasons,
        )
