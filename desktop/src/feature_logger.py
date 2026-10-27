"""Exportación incremental de indicadores faciales para la ablación."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


RAW_FEATURE_FIELDS = (
    "feature_run_id",
    "subject_id",
    "clip_id",
    "source",
    "config_id",
    "frame_index",
    "t_ms",
    "frame_width_px",
    "frame_height_px",
    "face_detected",
    "landmark_count",
    "face_width_px",
    "valid_landmarks",
    "valid_eye",
    "valid_mouth",
    "valid_gaze",
    "valid_pose",
    "valid_all",
    "invalid_reason",
    "ear",
    "mar",
    "perclos",
    "perclos_window_s",
    "closure_s",
    "blink_event",
    "blink_count",
    "mouth_open",
    "opening_s",
    "yawn_event",
    "yawn_count",
    "gaze_h_raw",
    "gaze_v_raw",
    "gaze_h_asymmetry",
    "gaze_v_asymmetry",
    "gaze_h",
    "gaze_v",
    "gaze_direction",
    "gaze_away_s",
    "gaze_away_event",
    "gaze_away_count",
    "pitch_deg",
    "yaw_deg",
    "roll_deg",
    "pose_scale",
    "pose_fit_error",
    "pitch_relative_deg",
    "yaw_relative_deg",
    "roll_relative_deg",
    "pose_direction",
    "pose_away_s",
    "pose_away_event",
    "pose_away_count",
    "nod_active",
    "nod_s",
    "nod_event",
    "nod_count",
    "valid_ratio",
    "face_latency_ms",
    "fusion_score",
    "evidence_closure",
    "evidence_perclos",
    "evidence_yawn",
    "evidence_gaze",
    "evidence_nod",
    "fusion_reasons",
    "fatigue_state",
    "fatigue_state_label",
    "fatigue_state_event",
    "yawns_last_5_min",
    "nods_last_2_min",
    "fusion_data_sufficient",
    "fusion_available_weight_ratio",
)


class RawFeatureLogger:
    """Añade filas compatibles a un único archivo ``raw_features.csv``."""

    def __init__(
        self,
        path: str | Path,
        *,
        subject_id: str,
        clip_id: str,
        source: str,
        config_id: str = "h3_provisional",
        flush_every: int = 30,
    ) -> None:
        if flush_every < 1:
            raise ValueError("flush_every debe ser mayor o igual que 1")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.subject_id = str(subject_id)
        self.clip_id = str(clip_id)
        self.source = str(source)
        self.config_id = str(config_id)
        self.flush_every = int(flush_every)
        self.feature_run_id = self._new_run_id()
        self.frame_index = 0
        self.first_timestamp_ms: int | None = None
        self.closed = False

        self._validate_existing_header()

        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._file = self.path.open(
            "a",
            encoding="utf-8",
            newline="",
        )
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=RAW_FEATURE_FIELDS,
            extrasaction="raise",
        )

        if new_file:
            self._writer.writeheader()
            self._file.flush()

    @staticmethod
    def _new_run_id() -> str:
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{now}_{uuid4().hex[:8]}"

    def _validate_existing_header(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return

        with self.path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as existing_file:
            reader = csv.reader(existing_file)
            existing_header = tuple(next(reader, ()))

        if existing_header != RAW_FEATURE_FIELDS:
            raise RuntimeError(
                "El encabezado del CSV existente no coincide con "
                "la versión actual. Renombra el archivo anterior "
                "antes de iniciar una nueva captura."
            )

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if isinstance(value, bool):
            return int(value)

        if value is None:
            return ""

        return value

    def write(self, *, timestamp_ms: int, **values: Any) -> None:
        """Escribe una fila y calcula un tiempo relativo por ejecución."""

        if self.closed:
            raise RuntimeError("No se puede escribir en un logger cerrado")

        timestamp_ms = int(timestamp_ms)

        if self.first_timestamp_ms is None:
            self.first_timestamp_ms = timestamp_ms

        unexpected = set(values) - set(RAW_FEATURE_FIELDS)

        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise KeyError(f"Columnas no reconocidas: {names}")

        row = {field: "" for field in RAW_FEATURE_FIELDS}
        row.update(
            {
                "feature_run_id": self.feature_run_id,
                "subject_id": self.subject_id,
                "clip_id": self.clip_id,
                "source": self.source,
                "config_id": self.config_id,
                "frame_index": self.frame_index,
                "t_ms": timestamp_ms - self.first_timestamp_ms,
            }
        )
        row.update(values)
        row = {
            key: self._csv_value(value)
            for key, value in row.items()
        }

        self._writer.writerow(row)
        self.frame_index += 1

        if self.frame_index % self.flush_every == 0:
            self._file.flush()

    def close(self) -> None:
        if self.closed:
            return

        self._file.flush()
        self._file.close()
        self.closed = True

    def __enter__(self) -> "RawFeatureLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
