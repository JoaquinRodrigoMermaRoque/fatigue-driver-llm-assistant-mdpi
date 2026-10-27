"""Captura guiada de calibración personal con MediaPipe."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
from statistics import median
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from calibration import (
    CalibrationDataset,
    CalibrationError,
    build_calibration_profile,
)
from cues import mean_ear, mean_gaze_ratios, mouth_aspect_ratio
from pose import extract_head_pose
from quality import assess_frame_quality


MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "face_landmarker.task"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "calibration_profiles"
)

PHASE_DURATION_S = 4.0
PREPARE_DELAY_S = 2.0
MIN_PHASE_SAMPLES = 60
MIN_PHASE_VALID_RATIO = 0.70
MAX_HEAD_DELTA_FOR_GAZE_DEG = 10.0

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
BLACK = (0, 0, 0)

PHASE_INSTRUCTIONS = (
    (
        "neutral",
        "Mira al centro, ojos abiertos, boca cerrada y cabeza frontal",
    ),
    (
        "eyes_closed",
        "Cierra ambos ojos y manten la cabeza frontal",
    ),
    (
        "mouth_open",
        "Abre ampliamente la boca y manten los ojos abiertos",
    ),
    (
        "gaze_left",
        "Sin mover la cabeza, mira al extremo de tu izquierda",
    ),
    (
        "gaze_right",
        "Sin mover la cabeza, mira al extremo de tu derecha",
    ),
    (
        "gaze_up",
        "Sin mover la cabeza, mira hacia arriba",
    ),
    (
        "gaze_down",
        "Sin mover la cabeza, mira hacia abajo",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibración personal de indicadores faciales",
    )
    parser.add_argument(
        "--subject-id",
        required=True,
        help="Identificador anónimo, por ejemplo subject_001",
    )

    glasses_group = parser.add_mutually_exclusive_group(required=True)
    glasses_group.add_argument(
        "--glasses",
        dest="glasses",
        action="store_true",
        help="El usuario realiza la calibración con lentes",
    )
    glasses_group.add_argument(
        "--no-glasses",
        dest="glasses",
        action="store_false",
        help="El usuario realiza la calibración sin lentes",
    )

    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--phase-duration-s",
        type=float,
        default=PHASE_DURATION_S,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar un perfil existente",
    )
    return parser.parse_args()


def safe_identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    result = result.strip("_")

    if not result:
        raise ValueError("subject_id no contiene caracteres válidos")

    return result


def put_line(image, text, line, color=WHITE, scale=0.55):
    y = 28 + line * 30
    cv2.putText(
        image,
        text,
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
    )


def neutral_pose(dataset: CalibrationDataset):
    pitch = dataset.values("neutral", "pitch_deg")
    yaw = dataset.values("neutral", "yaw_deg")
    roll = dataset.values("neutral", "roll_deg")

    if not pitch or not yaw or not roll:
        return None

    return (
        float(median(pitch)),
        float(median(yaw)),
        float(median(roll)),
    )


def gaze_head_is_stable(
    pose_values,
    baseline,
) -> bool:
    if baseline is None:
        return False

    return all(
        abs(value - reference) <= MAX_HEAD_DELTA_FOR_GAZE_DEG
        for value, reference in zip(pose_values, baseline)
    )


def phase_is_valid(
    phase,
    quality,
    pose_values,
    pose_baseline,
) -> bool:
    if phase == "neutral":
        return quality.valid_all

    if phase == "eyes_closed":
        return quality.valid_eye

    if phase == "mouth_open":
        return quality.valid_mouth

    return (
        quality.valid_gaze
        and quality.valid_pose
        and gaze_head_is_stable(pose_values, pose_baseline)
    )


def main():
    args = parse_args()

    if args.phase_duration_s <= 0.0:
        raise ValueError("phase-duration-s debe ser mayor que cero")

    subject_id = safe_identifier(args.subject_id)
    condition = "glasses" if args.glasses else "no_glasses"
    profile_path = args.output_dir / f"{subject_id}_{condition}.json"

    if profile_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"El perfil ya existe: {profile_path}. "
            "Usa --overwrite solamente si deseas reemplazarlo."
        )

    if not MODEL.is_file() or MODEL.stat().st_size == 0:
        raise FileNotFoundError(f"Modelo no disponible: {MODEL}")

    print("Sujeto:", subject_id)
    print("Lentes:", args.glasses)
    print("Perfil de salida:", profile_path)
    print("Presiona ESPACIO para iniciar cada fase y Q para cancelar.")

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )

    cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara {args.camera_index}"
        )

    dataset = CalibrationDataset()
    phase_index = 0
    preparing = False
    prepare_started_s = 0.0
    recording = False
    phase_started_s = 0.0
    status = "Presiona ESPACIO cuando estes preparado"
    completed = False
    cancelled = False
    previous_ms = -1

    try:
        with vision.FaceLandmarker.create_from_options(
            options
        ) as landmarker:
            while cap.isOpened() and not completed:
                ok, bgr = cap.read()

                if not ok:
                    raise RuntimeError("No se pudo leer la cámara")

                frame_height, frame_width = bgr.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = time.monotonic_ns() // 1_000_000
                timestamp_ms = max(timestamp_ms, previous_ms + 1)
                previous_ms = timestamp_ms
                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                phase, instruction = PHASE_INSTRUCTIONS[phase_index]
                face_detected = bool(result.face_landmarks)
                valid_for_phase = False
                quality_reason = "no_face"
                ear = float("nan")
                mar = float("nan")
                gaze_h = float("nan")
                gaze_v = float("nan")
                pitch_deg = float("nan")
                yaw_deg = float("nan")
                roll_deg = float("nan")

                if face_detected:
                    landmarks = result.face_landmarks[0]
                    matrix_available = bool(
                        result.facial_transformation_matrixes
                    )
                    ear = mean_ear(landmarks)
                    mar = mouth_aspect_ratio(landmarks)
                    gaze = mean_gaze_ratios(landmarks)
                    gaze_h = gaze.horizontal
                    gaze_v = gaze.vertical

                    x_coordinates = [
                        float(point.x)
                        for point in landmarks
                        if math.isfinite(float(point.x))
                    ]
                    face_width_px = (
                        (max(x_coordinates) - min(x_coordinates))
                        * frame_width
                        if x_coordinates
                        else 0.0
                    )
                    minimum_face_width = max(
                        100.0,
                        0.15 * frame_width,
                    )

                    pose = None

                    if matrix_available:
                        try:
                            pose = extract_head_pose(
                                result.facial_transformation_matrixes[0]
                            )
                        except (
                            ValueError,
                            TypeError,
                            np.linalg.LinAlgError,
                        ):
                            pose = None

                    if pose is not None:
                        pitch_deg = pose.pitch_deg
                        yaw_deg = pose.yaw_deg
                        roll_deg = pose.roll_deg
                        pose_scale = pose.scale
                        pose_fit_error = pose.fit_error
                    else:
                        pose_scale = float("nan")
                        pose_fit_error = float("nan")

                    quality = assess_frame_quality(
                        face_detected=True,
                        landmark_count=len(landmarks),
                        face_width_px=face_width_px,
                        minimum_face_width_px=minimum_face_width,
                        ear=ear,
                        mar=mar,
                        gaze_horizontal=gaze.horizontal,
                        gaze_vertical=gaze.vertical,
                        gaze_horizontal_asymmetry=(
                            gaze.horizontal_asymmetry
                        ),
                        gaze_vertical_asymmetry=(
                            gaze.vertical_asymmetry
                        ),
                        matrix_available=matrix_available,
                        pose_pitch_deg=pitch_deg,
                        pose_yaw_deg=yaw_deg,
                        pose_roll_deg=roll_deg,
                        pose_scale=pose_scale,
                        pose_fit_error=pose_fit_error,
                    )
                    quality_reason = quality.reason_code
                    pose_baseline = neutral_pose(dataset)
                    valid_for_phase = phase_is_valid(
                        phase,
                        quality,
                        (pitch_deg, yaw_deg, roll_deg),
                        pose_baseline,
                    )

                if preparing:
                    prepare_elapsed_s = (
                        time.monotonic() - prepare_started_s
                    )

                    if prepare_elapsed_s >= PREPARE_DELAY_S:
                        preparing = False
                        recording = True
                        phase_started_s = time.monotonic()
                        status = "GRABANDO"

                if recording:
                    dataset.add(
                        phase,
                        valid=valid_for_phase,
                        ear=ear,
                        mar=mar,
                        gaze_h=gaze_h,
                        gaze_v=gaze_v,
                        pitch_deg=pitch_deg,
                        yaw_deg=yaw_deg,
                        roll_deg=roll_deg,
                    )

                    elapsed_s = time.monotonic() - phase_started_s

                    if elapsed_s >= args.phase_duration_s:
                        recording = False
                        samples = dataset.sample_count(phase)
                        ratio = dataset.valid_ratio(phase)

                        if (
                            samples >= MIN_PHASE_SAMPLES
                            and ratio >= MIN_PHASE_VALID_RATIO
                        ):
                            print(
                                f"{phase}: OK | muestras={samples} | "
                                f"valid_ratio={ratio:.3f}"
                            )
                            phase_index += 1

                            if phase_index == len(PHASE_INSTRUCTIONS):
                                completed = True
                            else:
                                status = (
                                    "Fase completada. Preparate y "
                                    "presiona ESPACIO"
                                )
                        else:
                            status = (
                                f"REPETIR: muestras={samples}, "
                                f"validos={100.0 * ratio:.0f}%"
                            )

                if completed:
                    break

                phase, instruction = PHASE_INSTRUCTIONS[phase_index]
                cv2.rectangle(
                    bgr,
                    (0, 0),
                    (frame_width, 225),
                    BLACK,
                    -1,
                )
                put_line(
                    bgr,
                    f"Calibracion {phase_index + 1}/{len(PHASE_INSTRUCTIONS)}",
                    0,
                    WHITE,
                )
                put_line(bgr, instruction, 1, YELLOW, 0.47)
                if preparing:
                    prepare_remaining = max(
                        0.0,
                        PREPARE_DELAY_S
                        - (time.monotonic() - prepare_started_s),
                    )
                    state_text = (
                        f"PREPARATE: inicia en {prepare_remaining:.1f}s"
                    )
                    state_color = YELLOW
                elif recording:
                    state_text = "GRABANDO"
                    state_color = GREEN
                else:
                    state_text = status
                    state_color = WHITE

                put_line(
                    bgr,
                    state_text,
                    2,
                    state_color,
                )

                if recording:
                    remaining = max(
                        0.0,
                        args.phase_duration_s
                        - (time.monotonic() - phase_started_s),
                    )
                    put_line(
                        bgr,
                        f"Tiempo restante: {remaining:.1f}s",
                        3,
                        GREEN,
                    )
                elif not preparing:
                    put_line(
                        bgr,
                        "ESPACIO: iniciar | Q: cancelar",
                        3,
                        WHITE,
                    )

                put_line(
                    bgr,
                    (
                        f"EAR={ear:.3f} | MAR={mar:.3f} | "
                        f"Gaze={gaze_h:.3f}/{gaze_v:.3f}"
                    ),
                    4,
                    GREEN if valid_for_phase else RED,
                    0.50,
                )
                put_line(
                    bgr,
                    (
                        f"Pose={pitch_deg:.1f}/{yaw_deg:.1f}/"
                        f"{roll_deg:.1f} | Calidad={quality_reason}"
                    ),
                    5,
                    GREEN if valid_for_phase else RED,
                    0.48,
                )
                put_line(
                    bgr,
                    (
                        "Muestras: "
                        f"{dataset.sample_count(phase)}/"
                        f"{dataset.total_frames[phase]}"
                    ),
                    6,
                    WHITE,
                )

                cv2.imshow("H4 - Calibracion personal", bgr)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    cancelled = True
                    break

                if (
                    key == ord(" ")
                    and not recording
                    and not preparing
                ):
                    dataset.reset_phase(phase)
                    preparing = True
                    prepare_started_s = time.monotonic()
                    status = "PREPARATE"

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if cancelled:
        print("Calibración cancelada. No se guardó ningún perfil.")
        return

    if not completed:
        raise RuntimeError("La calibración no se completó")

    try:
        profile = build_calibration_profile(
            subject_id=subject_id,
            glasses=args.glasses,
            dataset=dataset,
            minimum_samples=MIN_PHASE_SAMPLES,
        )
    except CalibrationError as error:
        raise RuntimeError(
            f"La calibración fue rechazada: {error}"
        ) from error

    profile.save(profile_path)
    print("Perfil guardado:", profile_path)
    print(f"EAR threshold: {profile.ear_threshold:.4f}")
    print(f"MAR threshold: {profile.mar_threshold:.4f}")
    print(
        "Mirada H:",
        f"{profile.gaze_horizontal_low:.4f}",
        "-",
        f"{profile.gaze_horizontal_high:.4f}",
    )
    print(
        "Mirada V:",
        f"{profile.gaze_vertical_low:.4f}",
        "-",
        f"{profile.gaze_vertical_high:.4f}",
    )
    print(
        "Pose base P/Y/R:",
        f"{profile.pose_baseline_pitch_deg:.2f}",
        f"{profile.pose_baseline_yaw_deg:.2f}",
        f"{profile.pose_baseline_roll_deg:.2f}",
    )


if __name__ == "__main__":
    main()
