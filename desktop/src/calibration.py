"""Cálculo, validación y persistencia de calibración personal."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable


CALIBRATION_SCHEMA_VERSION = 1
MIN_VALID_SAMPLES = 30
MIN_GAZE_SEPARATION = 0.02

PHASES = (
    "neutral",
    "eyes_closed",
    "mouth_open",
    "gaze_left",
    "gaze_right",
    "gaze_up",
    "gaze_down",
)


class CalibrationError(ValueError):
    """Indica que las muestras no permiten una calibración confiable."""


def _finite_values(values: Iterable[float]) -> list[float]:
    result = []

    for value in values:
        number = float(value)

        if math.isfinite(number):
            result.append(number)

    return result


def _percentile(values: Iterable[float], percentage: float) -> float:
    """Percentil lineal sin depender de bibliotecas adicionales."""

    ordered = sorted(_finite_values(values))

    if not ordered:
        raise CalibrationError("No existen valores finitos")

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentage / 100.0
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))

    if lower_index == upper_index:
        return ordered[lower_index]

    fraction = position - lower_index

    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


@dataclass
class CalibrationDataset:
    """Acumula frames válidos y totales para cada fase guiada."""

    records: dict[str, list[dict[str, float]]] = field(
        default_factory=lambda: {phase: [] for phase in PHASES}
    )
    total_frames: dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in PHASES}
    )

    def add(
        self,
        phase: str,
        *,
        valid: bool,
        **measurements: float,
    ) -> None:
        if phase not in self.records:
            raise KeyError(f"Fase de calibración desconocida: {phase}")

        self.total_frames[phase] += 1

        if not valid:
            return

        finite_measurements = {
            name: float(value)
            for name, value in measurements.items()
            if math.isfinite(float(value))
        }
        self.records[phase].append(finite_measurements)

    def values(self, phase: str, measurement: str) -> list[float]:
        return [
            record[measurement]
            for record in self.records[phase]
            if measurement in record
        ]

    def sample_count(self, phase: str) -> int:
        return len(self.records[phase])

    def reset_phase(self, phase: str) -> None:
        """Elimina un intento fallido antes de repetir una fase."""

        if phase not in self.records:
            raise KeyError(f"Fase de calibración desconocida: {phase}")

        self.records[phase].clear()
        self.total_frames[phase] = 0

    def valid_ratio(self, phase: str) -> float:
        total = self.total_frames[phase]

        if total == 0:
            return 0.0

        return self.sample_count(phase) / total


@dataclass(frozen=True)
class CalibrationProfile:
    """Parámetros personales consumibles por el detector en tiempo real."""

    schema_version: int
    subject_id: str
    created_at_utc: str
    glasses: bool
    sample_counts: dict[str, int]
    valid_ratios: dict[str, float]
    ear_open_median: float
    ear_closed_median: float
    ear_threshold: float
    mar_neutral_median: float
    mar_open_median: float
    mar_threshold: float
    gaze_center_h: float
    gaze_center_v: float
    gaze_horizontal_low: float
    gaze_horizontal_high: float
    gaze_vertical_low: float
    gaze_vertical_high: float
    pose_baseline_pitch_deg: float
    pose_baseline_yaw_deg: float
    pose_baseline_roll_deg: float
    pose_pitch_threshold_deg: float
    pose_yaw_threshold_deg: float
    pose_roll_threshold_deg: float

    def save(self, path: str | Path) -> None:
        """Guarda el perfil sin dejar un JSON parcial."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                asdict(self),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))

        if data.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise CalibrationError(
                "Versión de perfil de calibración no compatible"
            )

        return cls(**data)


def _require_samples(
    dataset: CalibrationDataset,
    phase: str,
    measurement: str,
    minimum: int,
) -> list[float]:
    values = _finite_values(dataset.values(phase, measurement))

    if len(values) < minimum:
        raise CalibrationError(
            f"{phase}.{measurement}: se requieren al menos "
            f"{minimum} muestras válidas y solo hay {len(values)}"
        )

    return values


def build_calibration_profile(
    *,
    subject_id: str,
    glasses: bool,
    dataset: CalibrationDataset,
    minimum_samples: int = MIN_VALID_SAMPLES,
) -> CalibrationProfile:
    """Calcula umbrales robustos a partir de las siete fases."""

    if not subject_id.strip():
        raise CalibrationError("subject_id no puede estar vacío")

    ear_open = _require_samples(
        dataset,
        "neutral",
        "ear",
        minimum_samples,
    )
    ear_closed = _require_samples(
        dataset,
        "eyes_closed",
        "ear",
        minimum_samples,
    )
    mar_neutral = _require_samples(
        dataset,
        "neutral",
        "mar",
        minimum_samples,
    )
    mar_open = _require_samples(
        dataset,
        "mouth_open",
        "mar",
        minimum_samples,
    )

    ear_open_boundary = _percentile(ear_open, 10.0)
    ear_closed_boundary = _percentile(ear_closed, 90.0)

    if ear_closed_boundary >= ear_open_boundary:
        raise CalibrationError(
            "EAR abierto y cerrado no tienen separación suficiente"
        )

    ear_threshold = (
        ear_open_boundary + ear_closed_boundary
    ) / 2.0

    mar_neutral_boundary = _percentile(mar_neutral, 90.0)
    mar_open_boundary = _percentile(mar_open, 10.0)

    if mar_open_boundary <= mar_neutral_boundary:
        raise CalibrationError(
            "MAR neutral y abierto no tienen separación suficiente"
        )

    mar_threshold = (
        mar_neutral_boundary + mar_open_boundary
    ) / 2.0

    center_h_values = _require_samples(
        dataset,
        "neutral",
        "gaze_h",
        minimum_samples,
    )
    center_v_values = _require_samples(
        dataset,
        "neutral",
        "gaze_v",
        minimum_samples,
    )
    gaze_left_values = _require_samples(
        dataset,
        "gaze_left",
        "gaze_h",
        minimum_samples,
    )
    gaze_right_values = _require_samples(
        dataset,
        "gaze_right",
        "gaze_h",
        minimum_samples,
    )
    gaze_up_values = _require_samples(
        dataset,
        "gaze_up",
        "gaze_v",
        minimum_samples,
    )
    gaze_down_values = _require_samples(
        dataset,
        "gaze_down",
        "gaze_v",
        minimum_samples,
    )

    center_h = float(median(center_h_values))
    center_v = float(median(center_v_values))
    horizontal_extremes = sorted(
        (
            float(median(gaze_left_values)),
            float(median(gaze_right_values)),
        )
    )
    vertical_extremes = sorted(
        (
            float(median(gaze_up_values)),
            float(median(gaze_down_values)),
        )
    )

    if not (
        horizontal_extremes[0]
        + MIN_GAZE_SEPARATION
        < center_h
        < horizontal_extremes[1]
        - MIN_GAZE_SEPARATION
    ):
        raise CalibrationError(
            "Los extremos horizontales no rodean claramente al centro"
        )

    if not (
        vertical_extremes[0]
        + MIN_GAZE_SEPARATION
        < center_v
        < vertical_extremes[1]
        - MIN_GAZE_SEPARATION
    ):
        raise CalibrationError(
            "Los extremos verticales no rodean claramente al centro"
        )

    gaze_horizontal_low = (
        horizontal_extremes[0] + center_h
    ) / 2.0
    gaze_horizontal_high = (
        center_h + horizontal_extremes[1]
    ) / 2.0
    gaze_vertical_low = (
        vertical_extremes[0] + center_v
    ) / 2.0
    gaze_vertical_high = (
        center_v + vertical_extremes[1]
    ) / 2.0

    pose_pitch = _require_samples(
        dataset,
        "neutral",
        "pitch_deg",
        minimum_samples,
    )
    pose_yaw = _require_samples(
        dataset,
        "neutral",
        "yaw_deg",
        minimum_samples,
    )
    pose_roll = _require_samples(
        dataset,
        "neutral",
        "roll_deg",
        minimum_samples,
    )

    return CalibrationProfile(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        subject_id=subject_id.strip(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        glasses=bool(glasses),
        sample_counts={
            phase: dataset.sample_count(phase)
            for phase in PHASES
        },
        valid_ratios={
            phase: dataset.valid_ratio(phase)
            for phase in PHASES
        },
        ear_open_median=float(median(ear_open)),
        ear_closed_median=float(median(ear_closed)),
        ear_threshold=ear_threshold,
        mar_neutral_median=float(median(mar_neutral)),
        mar_open_median=float(median(mar_open)),
        mar_threshold=mar_threshold,
        gaze_center_h=center_h,
        gaze_center_v=center_v,
        gaze_horizontal_low=gaze_horizontal_low,
        gaze_horizontal_high=gaze_horizontal_high,
        gaze_vertical_low=gaze_vertical_low,
        gaze_vertical_high=gaze_vertical_high,
        pose_baseline_pitch_deg=float(median(pose_pitch)),
        pose_baseline_yaw_deg=float(median(pose_yaw)),
        pose_baseline_roll_deg=float(median(pose_roll)),
        # Los ángulos límite son criterios operativos; por ahora solo
        # personalizamos el cero frontal de cada usuario.
        pose_pitch_threshold_deg=15.0,
        pose_yaw_threshold_deg=20.0,
        pose_roll_threshold_deg=20.0,
    )
