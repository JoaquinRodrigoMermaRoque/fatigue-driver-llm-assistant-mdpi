"""Reproduce las configuraciones de ablación A-F sobre los mismos frames."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from statistics import fmean
from uuid import uuid4

try:
    from .ablation_config import AblationConfig, load_ablation
    from .calibration import CalibrationProfile
    from .fatigue_engine import FatigueEngine
    from .temporal import (
        EyeTemporalAnalyzer,
        GazeTemporalAnalyzer,
        HeadPoseTemporalAnalyzer,
        MouthTemporalAnalyzer,
    )
except ImportError:
    from ablation_config import AblationConfig, load_ablation
    from calibration import CalibrationProfile
    from fatigue_engine import FatigueEngine
    from temporal import (
        EyeTemporalAnalyzer,
        GazeTemporalAnalyzer,
        HeadPoseTemporalAnalyzer,
        MouthTemporalAnalyzer,
    )


MAX_GAZE_ASYMMETRY = 0.20
ACKNOWLEDGEMENT_POLICY = "automatic_after_critical"

REQUIRED_FEATURE_FIELDS = {
    "feature_run_id",
    "subject_id",
    "clip_id",
    "frame_index",
    "t_ms",
    "face_detected",
    "valid_eye",
    "valid_mouth",
    "valid_pose",
    "ear",
    "mar",
    "gaze_h_raw",
    "gaze_v_raw",
    "gaze_h_asymmetry",
    "gaze_v_asymmetry",
    "pitch_deg",
    "yaw_deg",
    "roll_deg",
}

PREDICTION_FIELDS = (
    "experiment_id",
    "feature_run_id",
    "subject_id",
    "clip_id",
    "frame_index",
    "t_ms",
    "config_id",
    "config_sha256",
    "schema_version",
    "active_cues",
    "calibration",
    "gaze_source",
    "acknowledgement_policy",
    "data_sufficient",
    "available_weight_ratio",
    "score",
    "state",
    "state_label",
    "state_event",
    "evidence_closure",
    "evidence_yawn",
    "evidence_perclos",
    "evidence_gaze",
    "evidence_nod",
    "closure_s",
    "perclos",
    "yawn_event",
    "yawns_last_5_min",
    "gaze_away_s",
    "nod_event",
    "nods_last_2_min",
)


@dataclass(frozen=True)
class ReplaySettings:
    name: str
    ear_threshold: float
    mar_threshold: float
    gaze_min_ear: float
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta las ablaciones A-F sobre una misma ejecución "
            "de raw_features.csv"
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("experiments/features/raw_features.csv"),
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path("config/ablations.json"),
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=list("ABCDEF"),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Perfil requerido por configuraciones personales como F",
    )
    parser.add_argument(
        "--run-id",
        help="Run de características; por defecto utiliza el último",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/predictions/ablation_predictions.csv"
        ),
    )
    return parser.parse_args()


def float_value(row: dict[str, str], key: str) -> float:
    text = row.get(key, "").strip()

    if not text:
        return float("nan")

    try:
        return float(text)
    except ValueError:
        return float("nan")


def int_value(row: dict[str, str], key: str) -> int:
    value = float_value(row, key)

    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"{key} debe contener un entero: {value!r}")

    return int(value)


def bool_value(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "si",
    }


def load_feature_run(
    path: str | Path,
    requested_run_id: str | None = None,
) -> tuple[str, list[dict[str, str]]]:
    feature_path = Path(path)

    if not feature_path.is_file():
        raise FileNotFoundError(f"No se encontró el CSV: {feature_path}")

    with feature_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_FEATURE_FIELDS - fields

        if missing:
            raise RuntimeError(
                "Faltan columnas crudas requeridas: "
                + ", ".join(sorted(missing))
            )

        rows = list(reader)

    if not rows:
        raise RuntimeError("El CSV no contiene frames")

    run_id = requested_run_id or rows[-1]["feature_run_id"]
    selected = [
        row
        for row in rows
        if row["feature_run_id"] == run_id
    ]

    if not selected:
        raise RuntimeError(f"No existen filas para el run: {run_id}")

    selected.sort(key=lambda row: int_value(row, "frame_index"))

    subjects = {row["subject_id"] for row in selected}
    clips = {row["clip_id"] for row in selected}

    if len(subjects) != 1 or len(clips) != 1:
        raise RuntimeError(
            "Un feature_run_id debe pertenecer a un único sujeto y clip"
        )

    previous_frame = -1
    previous_t_ms = -1

    for row in selected:
        frame_index = int_value(row, "frame_index")
        t_ms = int_value(row, "t_ms")

        if frame_index <= previous_frame:
            raise RuntimeError("frame_index debe ser único y creciente")

        if t_ms < previous_t_ms:
            raise RuntimeError("t_ms debe ser monotónico")

        previous_frame = frame_index
        previous_t_ms = t_ms

    return run_id, selected


def global_settings() -> ReplaySettings:
    return ReplaySettings(
        name="global",
        ear_threshold=0.21,
        mar_threshold=0.75,
        gaze_min_ear=0.15,
        gaze_horizontal_low=0.36,
        gaze_horizontal_high=0.62,
        gaze_vertical_low=0.40,
        gaze_vertical_high=0.44,
        pose_baseline_pitch_deg=6.55,
        pose_baseline_yaw_deg=1.25,
        pose_baseline_roll_deg=0.75,
        pose_pitch_threshold_deg=15.0,
        pose_yaw_threshold_deg=20.0,
        pose_roll_threshold_deg=20.0,
    )


def personal_settings(profile: CalibrationProfile) -> ReplaySettings:
    return ReplaySettings(
        name=f"personal:{profile.subject_id}",
        ear_threshold=profile.ear_threshold,
        mar_threshold=profile.mar_threshold,
        gaze_min_ear=profile.ear_threshold,
        gaze_horizontal_low=profile.gaze_horizontal_low,
        gaze_horizontal_high=profile.gaze_horizontal_high,
        gaze_vertical_low=profile.gaze_vertical_low,
        gaze_vertical_high=profile.gaze_vertical_high,
        pose_baseline_pitch_deg=profile.pose_baseline_pitch_deg,
        pose_baseline_yaw_deg=profile.pose_baseline_yaw_deg,
        pose_baseline_roll_deg=profile.pose_baseline_roll_deg,
        pose_pitch_threshold_deg=profile.pose_pitch_threshold_deg,
        pose_yaw_threshold_deg=profile.pose_yaw_threshold_deg,
        pose_roll_threshold_deg=profile.pose_roll_threshold_deg,
    )


def settings_for(
    config: AblationConfig,
    profile: CalibrationProfile | None,
    subject_id: str,
) -> ReplaySettings:
    if config.calibration == "global":
        return global_settings()

    if profile is None:
        raise ValueError(
            f"La configuración {config.config_id} requiere --profile"
        )

    if profile.subject_id != subject_id:
        raise ValueError(
            "El perfil no corresponde al sujeto del run: "
            f"{profile.subject_id} != {subject_id}"
        )

    return personal_settings(profile)


def new_experiment_id() -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{now}_{uuid4().hex[:8]}"


def replay_configuration(
    *,
    rows: list[dict[str, str]],
    config: AblationConfig,
    profile: CalibrationProfile | None,
    experiment_id: str,
) -> list[dict[str, object]]:
    if config.gaze_source == "mobileone":
        raise NotImplementedError(
            "La configuración G requiere MobileOne y es opcional"
        )

    subject_id = rows[0]["subject_id"]
    settings = settings_for(config, profile, subject_id)

    eye_analyzer = EyeTemporalAnalyzer(
        window_s=30.0,
        ear_threshold=settings.ear_threshold,
        min_blink_s=0.08,
        max_blink_s=0.80,
    )
    mouth_analyzer = MouthTemporalAnalyzer(
        mar_threshold=settings.mar_threshold,
        min_yawn_s=1.50,
    )
    gaze_analyzer = GazeTemporalAnalyzer(
        horizontal_low=settings.gaze_horizontal_low,
        horizontal_high=settings.gaze_horizontal_high,
        vertical_low=settings.gaze_vertical_low,
        vertical_high=settings.gaze_vertical_high,
        min_away_s=1.50,
        smoothing_tau_s=0.15,
    )
    pose_analyzer = HeadPoseTemporalAnalyzer(
        baseline_pitch_deg=settings.pose_baseline_pitch_deg,
        baseline_yaw_deg=settings.pose_baseline_yaw_deg,
        baseline_roll_deg=settings.pose_baseline_roll_deg,
        yaw_threshold_deg=settings.pose_yaw_threshold_deg,
        pitch_threshold_deg=settings.pose_pitch_threshold_deg,
        roll_threshold_deg=settings.pose_roll_threshold_deg,
        min_away_s=1.50,
        nod_down_threshold_deg=15.0,
        nod_return_threshold_deg=8.0,
        min_nod_s=0.30,
        max_nod_s=2.50,
        smoothing_tau_s=0.15,
    )
    engine = FatigueEngine(
        weights=config.weights,
        active_cues=config.active_cues,
    )

    predictions: list[dict[str, object]] = []

    for row in rows:
        timestamp_ms = int_value(row, "t_ms")
        timestamp_s = timestamp_ms / 1000.0
        face_detected = bool_value(row, "face_detected")
        ear = float_value(row, "ear")
        mar = float_value(row, "mar")
        gaze_h = float_value(row, "gaze_h_raw")
        gaze_v = float_value(row, "gaze_v_raw")
        gaze_h_asymmetry = float_value(row, "gaze_h_asymmetry")
        gaze_v_asymmetry = float_value(row, "gaze_v_asymmetry")
        pitch_deg = float_value(row, "pitch_deg")
        yaw_deg = float_value(row, "yaw_deg")
        roll_deg = float_value(row, "roll_deg")

        valid_eye = (
            face_detected
            and bool_value(row, "valid_eye")
            and math.isfinite(ear)
        )
        valid_mouth = (
            face_detected
            and bool_value(row, "valid_mouth")
            and math.isfinite(mar)
        )
        valid_pose = (
            face_detected
            and bool_value(row, "valid_pose")
            and all(
                math.isfinite(value)
                for value in (pitch_deg, yaw_deg, roll_deg)
            )
        )
        valid_gaze_values = all(
            math.isfinite(value)
            for value in (
                gaze_h,
                gaze_v,
                gaze_h_asymmetry,
                gaze_v_asymmetry,
            )
        )
        valid_gaze = (
            valid_eye
            and valid_gaze_values
            and ear >= settings.gaze_min_ear
            and gaze_h_asymmetry <= MAX_GAZE_ASYMMETRY
            and gaze_v_asymmetry <= MAX_GAZE_ASYMMETRY
        )

        eye_metrics = eye_analyzer.update(
            timestamp_s=timestamp_s,
            ear=ear,
            valid=valid_eye,
        )
        mouth_metrics = mouth_analyzer.update(
            timestamp_s=timestamp_s,
            mar=mar,
            valid=valid_mouth,
        )
        gaze_metrics = gaze_analyzer.update(
            timestamp_s=timestamp_s,
            horizontal=gaze_h,
            vertical=gaze_v,
            valid=valid_gaze,
        )
        pose_metrics = pose_analyzer.update(
            timestamp_s=timestamp_s,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=roll_deg,
            valid=valid_pose,
        )

        perclos = (
            eye_metrics.perclos
            if math.isfinite(eye_metrics.perclos)
            else 0.0
        )

        metrics = engine.update(
            timestamp_s=timestamp_s,
            closure_s=eye_metrics.closure_duration_s,
            perclos_ratio=perclos,
            gaze_away_s=gaze_metrics.away_duration_s,
            yawn_event=mouth_metrics.yawn_event,
            nod_event=pose_metrics.nod_event,
            face_valid=face_detected,
            eye_valid=valid_eye,
            mouth_valid=valid_mouth,
            gaze_valid=valid_gaze,
            pose_valid=valid_pose,
            # Política offline común: reconoce CRITICAL tan pronto como
            # desaparece la condición peligrosa y permite recuperación.
            acknowledged=True,
        )

        predictions.append(
            {
                "experiment_id": experiment_id,
                "feature_run_id": row["feature_run_id"],
                "subject_id": row["subject_id"],
                "clip_id": row["clip_id"],
                "frame_index": int_value(row, "frame_index"),
                "t_ms": timestamp_ms,
                "config_id": config.config_id,
                "config_sha256": config.config_sha256,
                "schema_version": config.schema_version,
                "active_cues": ";".join(config.active_cues),
                "calibration": config.calibration,
                "gaze_source": config.gaze_source,
                "acknowledgement_policy": ACKNOWLEDGEMENT_POLICY,
                "data_sufficient": int(metrics.data_sufficient),
                "available_weight_ratio": (
                    metrics.available_weight_ratio
                ),
                "score": metrics.score,
                "state": metrics.state.name,
                "state_label": metrics.state_label,
                "state_event": metrics.state_event or "",
                "evidence_closure": metrics.evidence.closure,
                "evidence_yawn": metrics.evidence.yawn,
                "evidence_perclos": metrics.evidence.perclos,
                "evidence_gaze": metrics.evidence.gaze,
                "evidence_nod": metrics.evidence.nod,
                "closure_s": eye_metrics.closure_duration_s,
                "perclos": perclos,
                "yawn_event": int(mouth_metrics.yawn_event),
                "yawns_last_5_min": metrics.yawns_last_5_min,
                "gaze_away_s": gaze_metrics.away_duration_s,
                "nod_event": int(pose_metrics.nod_event),
                "nods_last_2_min": metrics.nods_last_2_min,
            }
        )

    return predictions


def prediction_identities(
    rows: list[dict[str, object]],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            row["subject_id"],
            row["clip_id"],
            row["frame_index"],
            row["t_ms"],
        )
        for row in rows
    )


def verify_same_frames(
    predictions_by_config: dict[str, list[dict[str, object]]],
) -> None:
    reference = None

    for config_id, rows in predictions_by_config.items():
        identities = prediction_identities(rows)

        if len(identities) != len(set(identities)):
            raise RuntimeError(
                f"La configuración {config_id} contiene frames duplicados"
            )

        if reference is None:
            reference = identities
        elif identities != reference:
            raise RuntimeError(
                f"La configuración {config_id} no usa los mismos frames"
            )


def write_predictions(
    path: str | Path,
    predictions_by_config: dict[str, list[dict[str, object]]],
) -> None:
    verify_same_frames(predictions_by_config)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()

        for rows in predictions_by_config.values():
            writer.writerows(rows)


def print_summary(
    predictions_by_config: dict[str, list[dict[str, object]]],
) -> None:
    print()
    print(
        f"{'ID':<4} {'Frames':>8} {'Score medio':>13} "
        f"{'Score máx':>11} {'Estados'}"
    )

    for config_id, rows in predictions_by_config.items():
        scores = [float(row["score"]) for row in rows]
        states = Counter(str(row["state"]) for row in rows)
        state_text = ", ".join(
            f"{name}:{count}"
            for name, count in sorted(states.items())
        )
        print(
            f"{config_id:<4} {len(rows):>8} "
            f"{fmean(scores):>13.4f} "
            f"{max(scores):>11.4f} {state_text}"
        )


def main():
    args = parse_args()
    run_id, rows = load_feature_run(args.features, args.run_id)
    profile = (
        CalibrationProfile.load(args.profile)
        if args.profile is not None
        else None
    )
    experiment_id = new_experiment_id()

    configs = [
        load_ablation(args.configs, config_id)
        for config_id in args.ids
    ]

    predictions_by_config = {
        config.config_id: replay_configuration(
            rows=rows,
            config=config,
            profile=profile,
            experiment_id=experiment_id,
        )
        for config in configs
    }

    write_predictions(args.output, predictions_by_config)

    print("Experimento:", experiment_id)
    print("Run de características:", run_id)
    print("Frames por configuración:", len(rows))
    print("Configuraciones:", " ".join(args.ids))
    print("Política de reconocimiento:", ACKNOWLEDGEMENT_POLICY)
    print("Resultado:", args.output)
    print_summary(predictions_by_config)


if __name__ == "__main__":
    main()
