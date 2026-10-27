"""Compara configuración provisional y personal sobre los mismos frames."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path

from calibration import CalibrationProfile
from temporal import (
    EyeTemporalAnalyzer,
    GazeTemporalAnalyzer,
    HeadPoseTemporalAnalyzer,
    MouthTemporalAnalyzer,
)


MAX_GAZE_ASYMMETRY = 0.20


@dataclass(frozen=True)
class ReplayConfiguration:
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


@dataclass(frozen=True)
class ReplaySummary:
    feature_run_id: str
    configuration: str
    frames: int
    duration_s: float
    valid_eye_ratio: float
    valid_mouth_ratio: float
    valid_gaze_ratio: float
    valid_pose_ratio: float
    closed_frame_ratio: float
    mouth_open_frame_ratio: float
    gaze_off_center_ratio: float
    pose_off_center_ratio: float
    perclos_final: float
    perclos_max: float
    blink_count: int
    yawn_count: int
    gaze_away_count: int
    pose_away_count: int
    nod_count: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce una ejecución con umbrales H3 y H4 "
            "sin volver a capturar la cámara"
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("experiments/features/raw_features.csv"),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--run-id",
        help="Ejecución a comparar; por defecto utiliza la última",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Ruta del resumen CSV; se genera automáticamente si se omite",
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


def bool_value(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "si",
    }


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")

    return numerator / denominator


def load_run(
    path: Path,
    requested_run_id: str | None,
) -> tuple[str, list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el CSV: {path}")

    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError("El CSV no contiene frames")

    required = {
        "feature_run_id",
        "frame_index",
        "t_ms",
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
    missing = required - set(rows[0])

    if missing:
        raise RuntimeError(
            "Faltan columnas requeridas: "
            + ", ".join(sorted(missing))
        )

    run_id = requested_run_id or rows[-1]["feature_run_id"]
    selected = [
        row for row in rows
        if row["feature_run_id"] == run_id
    ]

    if not selected:
        raise RuntimeError(f"No existen filas para el run: {run_id}")

    selected.sort(key=lambda row: int(row["frame_index"]))
    return run_id, selected


def provisional_configuration() -> ReplayConfiguration:
    return ReplayConfiguration(
        name="h3_provisional",
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


def personal_configuration(
    profile: CalibrationProfile,
) -> ReplayConfiguration:
    condition = "glasses" if profile.glasses else "no_glasses"
    return ReplayConfiguration(
        name=f"h4_personal_{profile.subject_id}_{condition}",
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


def replay(
    run_id: str,
    rows: list[dict[str, str]],
    configuration: ReplayConfiguration,
) -> ReplaySummary:
    eye_analyzer = EyeTemporalAnalyzer(
        window_s=30.0,
        ear_threshold=configuration.ear_threshold,
        min_blink_s=0.08,
        max_blink_s=0.80,
    )
    mouth_analyzer = MouthTemporalAnalyzer(
        mar_threshold=configuration.mar_threshold,
        min_yawn_s=1.50,
    )
    gaze_analyzer = GazeTemporalAnalyzer(
        horizontal_low=configuration.gaze_horizontal_low,
        horizontal_high=configuration.gaze_horizontal_high,
        vertical_low=configuration.gaze_vertical_low,
        vertical_high=configuration.gaze_vertical_high,
        min_away_s=1.50,
        smoothing_tau_s=0.15,
    )
    pose_analyzer = HeadPoseTemporalAnalyzer(
        baseline_pitch_deg=configuration.pose_baseline_pitch_deg,
        baseline_yaw_deg=configuration.pose_baseline_yaw_deg,
        baseline_roll_deg=configuration.pose_baseline_roll_deg,
        yaw_threshold_deg=configuration.pose_yaw_threshold_deg,
        pitch_threshold_deg=configuration.pose_pitch_threshold_deg,
        roll_threshold_deg=configuration.pose_roll_threshold_deg,
        min_away_s=1.50,
        nod_down_threshold_deg=15.0,
        nod_return_threshold_deg=8.0,
        min_nod_s=0.30,
        max_nod_s=2.50,
        smoothing_tau_s=0.15,
    )

    valid_eye_frames = 0
    valid_mouth_frames = 0
    valid_gaze_frames = 0
    valid_pose_frames = 0
    closed_frames = 0
    mouth_open_frames = 0
    gaze_off_center_frames = 0
    pose_off_center_frames = 0
    finite_perclos_values: list[float] = []

    eye_metrics = None
    mouth_metrics = None
    gaze_metrics = None
    pose_metrics = None

    for row in rows:
        timestamp_s = float_value(row, "t_ms") / 1000.0
        ear = float_value(row, "ear")
        mar = float_value(row, "mar")
        gaze_h = float_value(row, "gaze_h_raw")
        gaze_v = float_value(row, "gaze_v_raw")
        gaze_h_asymmetry = float_value(
            row,
            "gaze_h_asymmetry",
        )
        gaze_v_asymmetry = float_value(
            row,
            "gaze_v_asymmetry",
        )
        pitch_deg = float_value(row, "pitch_deg")
        yaw_deg = float_value(row, "yaw_deg")
        roll_deg = float_value(row, "roll_deg")

        valid_eye = bool_value(row, "valid_eye") and math.isfinite(ear)
        valid_mouth = (
            bool_value(row, "valid_mouth")
            and math.isfinite(mar)
        )
        valid_pose = (
            bool_value(row, "valid_pose")
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
            and ear >= configuration.gaze_min_ear
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

        if valid_eye:
            valid_eye_frames += 1

            if ear < configuration.ear_threshold:
                closed_frames += 1

        if valid_mouth:
            valid_mouth_frames += 1

            if mouth_metrics.mouth_open:
                mouth_open_frames += 1

        if valid_gaze:
            valid_gaze_frames += 1

            if gaze_metrics.off_center:
                gaze_off_center_frames += 1

        if valid_pose:
            valid_pose_frames += 1

            if pose_metrics.off_center:
                pose_off_center_frames += 1

        if math.isfinite(eye_metrics.perclos):
            finite_perclos_values.append(eye_metrics.perclos)

    if not all(
        metric is not None
        for metric in (
            eye_metrics,
            mouth_metrics,
            gaze_metrics,
            pose_metrics,
        )
    ):
        raise RuntimeError("No fue posible reproducir la ejecución")

    duration_s = float_value(rows[-1], "t_ms") / 1000.0
    frame_count = len(rows)

    return ReplaySummary(
        feature_run_id=run_id,
        configuration=configuration.name,
        frames=frame_count,
        duration_s=duration_s,
        valid_eye_ratio=safe_ratio(valid_eye_frames, frame_count),
        valid_mouth_ratio=safe_ratio(
            valid_mouth_frames,
            frame_count,
        ),
        valid_gaze_ratio=safe_ratio(valid_gaze_frames, frame_count),
        valid_pose_ratio=safe_ratio(valid_pose_frames, frame_count),
        closed_frame_ratio=safe_ratio(
            closed_frames,
            valid_eye_frames,
        ),
        mouth_open_frame_ratio=safe_ratio(
            mouth_open_frames,
            valid_mouth_frames,
        ),
        gaze_off_center_ratio=safe_ratio(
            gaze_off_center_frames,
            valid_gaze_frames,
        ),
        pose_off_center_ratio=safe_ratio(
            pose_off_center_frames,
            valid_pose_frames,
        ),
        perclos_final=eye_metrics.perclos,
        perclos_max=(
            max(finite_perclos_values)
            if finite_perclos_values
            else float("nan")
        ),
        blink_count=eye_metrics.blink_count,
        yawn_count=mouth_metrics.yawn_count,
        gaze_away_count=gaze_metrics.gaze_away_count,
        pose_away_count=pose_metrics.pose_away_count,
        nod_count=pose_metrics.nod_count,
    )


def write_summary(path: Path, summaries: list[ReplaySummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(summary) for summary in summaries]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def display_value(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"

        return f"{value:.4f}"

    return str(value)


def main():
    args = parse_args()
    profile = CalibrationProfile.load(args.profile)
    run_id, rows = load_run(args.features, args.run_id)
    configurations = [
        provisional_configuration(),
        personal_configuration(profile),
    ]
    summaries = [
        replay(run_id, rows, configuration)
        for configuration in configurations
    ]

    output = args.output or (
        Path("experiments/results")
        / f"calibration_comparison_{run_id}.csv"
    )
    write_summary(output, summaries)

    print("Run comparado:", run_id)
    print("Frames:", len(rows))
    print("Resultado:", output)
    print()

    metric_names = (
        "valid_eye_ratio",
        "valid_mouth_ratio",
        "valid_gaze_ratio",
        "valid_pose_ratio",
        "closed_frame_ratio",
        "mouth_open_frame_ratio",
        "gaze_off_center_ratio",
        "pose_off_center_ratio",
        "perclos_final",
        "perclos_max",
        "blink_count",
        "yawn_count",
        "gaze_away_count",
        "pose_away_count",
        "nod_count",
    )
    print(
        f"{'Métrica':<28} "
        f"{'H3 provisional':>16} "
        f"{'H4 personal':>16}"
    )

    for metric_name in metric_names:
        provisional_value = getattr(summaries[0], metric_name)
        personal_value = getattr(summaries[1], metric_name)
        print(
            f"{metric_name:<28} "
            f"{display_value(provisional_value):>16} "
            f"{display_value(personal_value):>16}"
        )

    print()
    print(
        "Nota: esta tabla compara sensibilidad funcional. "
        "Sin etiquetas humanas no representa accuracy, precision o recall."
    )


if __name__ == "__main__":
    main()
