from pathlib import Path
import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from cues import (
    LEFT_EYE,
    LEFT_IRIS,
    RIGHT_EYE,
    RIGHT_IRIS,
    mean_ear,
    mean_gaze_ratios,
    mouth_aspect_ratio,
)

from temporal import (
    EyeTemporalAnalyzer,
    GazeTemporalAnalyzer,
    HeadPoseTemporalAnalyzer,
    MouthTemporalAnalyzer,
)

from calibration import CalibrationProfile
from pose import extract_head_pose
from quality import assess_frame_quality
from feature_logger import RawFeatureLogger
from fatigue_engine import FatigueEngine
from state_machine import State
from speech import OfflineAlertSpeaker
from voice import OfflineVoiceRecognizer, VoiceCommand

# Ruta construida automáticamente desde este archivo.
MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "face_landmarker.task"
)

VOSK_MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "vosk-model-small-es-0.42"
)

PIPER_MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "optional"
    / "piper"
    / "es_MX-claude-high.onnx"
)

# Cambia este número solamente si /dev/video0 no funciona.
CAMERA_INDEX = 0

# Colores BGR utilizados por OpenCV.
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)

MAX_GAZE_ASYMMETRY = 0.20
MAX_POSE_FIT_ERROR = 0.05

# Identificadores editables mediante variables de entorno.
SUBJECT_ID_OVERRIDE = os.environ.get("FATIGUE_SUBJECT_ID")
CLIP_ID = os.environ.get("FATIGUE_CLIP_ID", "live_camera")
CONFIG_ID_OVERRIDE = os.environ.get("FATIGUE_CONFIG_ID")
PROFILE_PATH_VALUE = os.environ.get("FATIGUE_PROFILE")

VOICE_ENABLED = os.environ.get(
    "FATIGUE_VOICE_ENABLED",
    "1",
).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

PIPER_VOLUME = float(
    os.environ.get("FATIGUE_PIPER_VOLUME", "1.20")
)

FEATURES_CSV = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "features"
    / "raw_features.csv"
)


def load_runtime_settings():
    """Carga el perfil personal o conserva los valores de H3."""

    settings = {
        "profile": None,
        "profile_path": None,
        "subject_id": SUBJECT_ID_OVERRIDE or "pilot_001",
        "config_id": CONFIG_ID_OVERRIDE or "h5_fusion_provisional",
        "ear_threshold": 0.21,
        "mar_threshold": 0.75,
        "gaze_min_ear": 0.15,
        "gaze_horizontal_low": 0.36,
        "gaze_horizontal_high": 0.62,
        "gaze_vertical_low": 0.40,
        "gaze_vertical_high": 0.44,
        "pose_baseline_pitch_deg": 6.55,
        "pose_baseline_yaw_deg": 1.25,
        "pose_baseline_roll_deg": 0.75,
        "pose_pitch_threshold_deg": 15.0,
        "pose_yaw_threshold_deg": 20.0,
        "pose_roll_threshold_deg": 20.0,
    }

    if not PROFILE_PATH_VALUE:
        return settings

    profile_path = Path(PROFILE_PATH_VALUE).expanduser().resolve()
    profile = CalibrationProfile.load(profile_path)

    if (
        SUBJECT_ID_OVERRIDE
        and SUBJECT_ID_OVERRIDE != profile.subject_id
    ):
        raise ValueError(
            "FATIGUE_SUBJECT_ID no coincide con el sujeto "
            f"del perfil: {profile.subject_id}"
        )

    condition = "glasses" if profile.glasses else "no_glasses"
    settings.update(
        {
            "profile": profile,
            "profile_path": profile_path,
            "subject_id": profile.subject_id,
            "config_id": (
                CONFIG_ID_OVERRIDE
                or f"h5_fusion_personal_{profile.subject_id}_{condition}"
            ),
            "ear_threshold": profile.ear_threshold,
            "mar_threshold": profile.mar_threshold,
            # La mirada se invalida cuando el clasificador personal
            # considera que el ojo ya está cerrado.
            "gaze_min_ear": profile.ear_threshold,
            "gaze_horizontal_low": profile.gaze_horizontal_low,
            "gaze_horizontal_high": profile.gaze_horizontal_high,
            "gaze_vertical_low": profile.gaze_vertical_low,
            "gaze_vertical_high": profile.gaze_vertical_high,
            "pose_baseline_pitch_deg": (
                profile.pose_baseline_pitch_deg
            ),
            "pose_baseline_yaw_deg": profile.pose_baseline_yaw_deg,
            "pose_baseline_roll_deg": profile.pose_baseline_roll_deg,
            "pose_pitch_threshold_deg": (
                profile.pose_pitch_threshold_deg
            ),
            "pose_yaw_threshold_deg": (
                profile.pose_yaw_threshold_deg
            ),
            "pose_roll_threshold_deg": (
                profile.pose_roll_threshold_deg
            ),
        }
    )
    return settings


def draw_landmark_group(
    image,
    landmarks,
    indices,
    color,
):
    """
    Dibuja en la imagen los landmarks indicados.

    Los landmarks de MediaPipe utilizan coordenadas normalizadas
    entre 0 y 1. Por eso se multiplican por el ancho y alto
    de la imagen.
    """

    frame_height, frame_width = image.shape[:2]

    for index in indices:
        point = landmarks[index]

        x = int(point.x * frame_width)
        y = int(point.y * frame_height)

        if 0 <= x < frame_width and 0 <= y < frame_height:
            cv2.circle(
                image,
                (x, y),
                3,
                color,
                -1,
            )


def calculate_face_width_px(landmarks, frame_width):
    """
    Calcula aproximadamente el ancho del rostro en píxeles.
    """

    x_coordinates = [
        float(point.x)
        for point in landmarks
        if math.isfinite(float(point.x))
    ]

    if not x_coordinates:
        return 0.0

    normalized_width = max(x_coordinates) - min(x_coordinates)

    return normalized_width * frame_width


def put_line(
    image,
    text,
    line_number,
    color=WHITE,
):
    """
    Escribe una línea de información sobre la imagen.
    """

    # El espaciado compacto permite mostrar todos los indicadores
    # incluso en una captura de 640 x 480.
    y_position = 22 + (line_number * 23)

    cv2.putText(
        image,
        text,
        (15, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        color,
        1,
    )


def fatigue_state_color(state):
    """Selecciona un color BGR para el estado operativo de H5."""

    return {
        State.NORMAL: GREEN,
        State.SUSPECTED: YELLOW,
        State.FATIGUE: ORANGE,
        State.CRITICAL: RED,
        State.RECOVERY: CYAN,
    }[state]


def consume_voice_commands(
    recognizer,
    speaker,
):
    """Consume comandos disponibles sin bloquear la cámara."""

    acknowledge_requested = False
    last_command = None

    if recognizer is None:
        return acknowledge_requested, last_command

    for event in recognizer.drain():
        last_command = event.command.value

        print(
            "COMANDO H6 | "
            f"comando={event.command.value} | "
            f"texto={event.text!r}"
        )

        if event.command == VoiceCommand.ACK:
            acknowledge_requested = True

            if speaker is not None:
                speaker.cancel(clear_pending=True)

        elif event.command == VoiceCommand.REPEAT:
            repeated = (
                speaker.repeat_last()
                if speaker is not None
                else False
            )
            print(
                "REPETICIÓN H6 | "
                f"aceptada={repeated}"
            )

        elif event.command == VoiceCommand.CANCEL:
            if speaker is not None:
                speaker.cancel(clear_pending=True)

            print("CANCELACIÓN H6 solicitada.")

    return acknowledge_requested, last_command


def announce_fatigue_state(
    fatigue_metrics,
    speaker,
):
    """Programa la alerta aplicable y respeta su cooldown."""

    if speaker is None:
        return False

    if (
        fatigue_metrics.state == State.CRITICAL
        and fatigue_metrics.alert_acknowledged
    ):
        return False

    if (
        not fatigue_metrics.data_sufficient
        and fatigue_metrics.state != State.RECOVERY
    ):
        return False

    # State utiliza Enum.auto(), por lo que ``state.value`` es un
    # entero. speech.py trabaja con nombres como "CRITICAL"; enviamos
    # explícitamente ``name`` para que pueda localizar el mensaje.
    accepted = speaker.announce_state(
        fatigue_metrics.state.name
    )

    if accepted:
        print(
            "ALERTA H6 | "
            f"estado={fatigue_metrics.state.name} | "
            f"backend={speaker.backend}"
        )

    return accepted


def voice_status_text(
    recognizer,
    speaker,
    last_command,
):
    """Construye una línea compacta para la interfaz."""

    if recognizer is None:
        recognition_status = "SIN VOSK"
    elif recognizer.is_running:
        recognition_status = "VOSK"
    else:
        recognition_status = "VOSK OFF"

    if speaker is None:
        speech_status = "SIN TTS"
    else:
        speech_status = (
            speaker.backend.upper()
            if speaker.backend
            else "TTS OFF"
        )

    return (
        f"H6 {recognition_status}/{speech_status} | "
        f"Ultimo: {last_command} | Q/A"
    )


def main():
    settings = load_runtime_settings()
    profile = settings["profile"]
    subject_id = settings["subject_id"]
    config_id = settings["config_id"]

    print("Modelo:", MODEL)
    print("Cámara seleccionada:", CAMERA_INDEX)
    print(
        "Indicadores activos: EAR, MAR, PERCLOS, "
        "parpadeos, bostezos, mirada, pose, fusion y voz"
    )
    print("Sujeto:", subject_id)
    print("Clip:", CLIP_ID)
    print(
        "Calibración:",
        "PERSONAL" if profile is not None else "PROVISIONAL H3",
    )

    if profile is not None:
        print("Perfil:", settings["profile_path"])
        print("Lentes en calibración:", profile.glasses)

    print(
        "Umbrales EAR/MAR:",
        f"{settings['ear_threshold']:.4f} / "
        f"{settings['mar_threshold']:.4f}",
    )
    print("CSV de características:", FEATURES_CSV)
    print("Q: cerrar | A: reconocer alerta grave")

    if not MODEL.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo: {MODEL}"
        )

    if MODEL.stat().st_size == 0:
        raise RuntimeError(
            f"El modelo está vacío: {MODEL}"
        )

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

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara {CAMERA_INDEX}. "
            "Prueba cambiando CAMERA_INDEX a 1, 2 o 3."
        )

    previous_ms = -1
    first_detection_reported = False
    last_console_report_ms = 0

    eye_temporal = EyeTemporalAnalyzer(
        window_s=30.0,
        ear_threshold=settings["ear_threshold"],
        min_blink_s=0.08,
        max_blink_s=0.80,
    )

    mouth_temporal = MouthTemporalAnalyzer(
        mar_threshold=settings["mar_threshold"],
        min_yawn_s=1.50,
    )

    gaze_temporal = GazeTemporalAnalyzer(
        horizontal_low=settings["gaze_horizontal_low"],
        horizontal_high=settings["gaze_horizontal_high"],
        vertical_low=settings["gaze_vertical_low"],
        vertical_high=settings["gaze_vertical_high"],
        min_away_s=1.50,
        smoothing_tau_s=0.15,
    )

    head_pose_temporal = HeadPoseTemporalAnalyzer(
        baseline_pitch_deg=settings["pose_baseline_pitch_deg"],
        baseline_yaw_deg=settings["pose_baseline_yaw_deg"],
        baseline_roll_deg=settings["pose_baseline_roll_deg"],
        yaw_threshold_deg=settings["pose_yaw_threshold_deg"],
        pitch_threshold_deg=settings["pose_pitch_threshold_deg"],
        roll_threshold_deg=settings["pose_roll_threshold_deg"],
        min_away_s=1.50,
        nod_down_threshold_deg=15.0,
        nod_return_threshold_deg=8.0,
        min_nod_s=0.30,
        max_nod_s=2.50,
        smoothing_tau_s=0.15,
    )

    fatigue_engine = FatigueEngine()
    acknowledge_requested = False

    voice_recognizer = None
    alert_speaker = None
    last_voice_command = "NINGUNO"

    feature_logger = None

    try:
        if VOICE_ENABLED:
            try:
                alert_speaker = OfflineAlertSpeaker(
                    model_path=PIPER_MODEL,
                    volume=PIPER_VOLUME,
                )
                alert_speaker.start()
                print(
                    "Síntesis H6:",
                    alert_speaker.backend,
                )
                print("Modelo Piper:", PIPER_MODEL)
            except Exception as error:
                alert_speaker = None
                print(
                    "AVISO H6: síntesis no disponible:",
                    error,
                )

            try:
                voice_recognizer = OfflineVoiceRecognizer(
                    model_path=VOSK_MODEL,
                )
                voice_recognizer.start()
                print(
                    "Reconocimiento H6: Vosk | Micrófono:",
                    voice_recognizer.selected_device_index,
                    voice_recognizer.selected_device_name,
                )
            except Exception as error:
                voice_recognizer = None
                print(
                    "AVISO H6: reconocimiento no disponible:",
                    error,
                )
        else:
            print(
                "Voz H6 desactivada mediante "
                "FATIGUE_VOICE_ENABLED."
            )

        feature_logger = RawFeatureLogger(
            FEATURES_CSV,
            subject_id=subject_id,
            clip_id=CLIP_ID,
            source=f"camera_{CAMERA_INDEX}",
            config_id=config_id,
            flush_every=30,
        )

        print(
            "Ejecución de características:",
            feature_logger.feature_run_id,
        )

        with vision.FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            while cap.isOpened():
                ok, bgr = cap.read()

                if not ok:
                    print(
                        "No se pudo obtener una imagen "
                        "de la cámara."
                    )
                    break

                frame_height, frame_width = bgr.shape[:2]

                rgb = cv2.cvtColor(
                    bgr,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = (
                    time.monotonic_ns() // 1_000_000
                )

                timestamp_ms = max(
                    timestamp_ms,
                    previous_ms + 1,
                )

                previous_ms = timestamp_ms

                voice_acknowledged, voice_command = (
                    consume_voice_commands(
                        voice_recognizer,
                        alert_speaker,
                    )
                )

                if voice_acknowledged:
                    acknowledge_requested = True

                if voice_command is not None:
                    last_voice_command = voice_command

                detection_start_ns = time.perf_counter_ns()

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                face_latency_ms = (
                    time.perf_counter_ns() - detection_start_ns
                ) / 1_000_000.0

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]

                    matrix_available = bool(
                        result.facial_transformation_matrixes
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
                        pose_pitch_deg = pose.pitch_deg
                        pose_yaw_deg = pose.yaw_deg
                        pose_roll_deg = pose.roll_deg
                        pose_scale = pose.scale
                        pose_fit_error = pose.fit_error
                    else:
                        pose_pitch_deg = float("nan")
                        pose_yaw_deg = float("nan")
                        pose_roll_deg = float("nan")
                        pose_scale = float("nan")
                        pose_fit_error = float("nan")

                    # Calculamos el EAR promedio de ambos ojos.
                    ear = mean_ear(landmarks)

                    #Calculamos la apertura relativa de la boca
                    mar = mouth_aspect_ratio(landmarks)

                    # Calculamos la posición geométrica de los iris.
                    gaze = mean_gaze_ratios(landmarks)

                    # Calculamos el tamaño aproximado del rostro.
                    face_width_px = calculate_face_width_px(
                        landmarks,
                        frame_width,
                    )

                    minimum_face_width = max(
                        100.0,
                        0.15 * frame_width,
                    )

                    valid_ear = math.isfinite(ear)
                    valid_mar = math.isfinite(mar)

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
                        pose_pitch_deg=pose_pitch_deg,
                        pose_yaw_deg=pose_yaw_deg,
                        pose_roll_deg=pose_roll_deg,
                        pose_scale=pose_scale,
                        pose_fit_error=pose_fit_error,
                        gaze_min_ear=settings["gaze_min_ear"],
                        max_gaze_asymmetry=MAX_GAZE_ASYMMETRY,
                        max_pose_fit_error=MAX_POSE_FIT_ERROR,
                    )

                    valid_eye_measurement = quality.valid_eye
                    valid_mouth_measurement = quality.valid_mouth
                    valid_gaze = quality.valid_gaze
                    valid_pose_measurement = quality.valid_pose

                    temporal_metrics = eye_temporal.update(
                        timestamp_s=timestamp_ms / 1000.0,
                        ear=ear,
                        valid=valid_eye_measurement,
                    )

                    mouth_metrics = mouth_temporal.update(
                        timestamp_s=timestamp_ms / 1000.0,
                        mar=mar,
                        valid=valid_mouth_measurement,
                    )

                    gaze_metrics = gaze_temporal.update(
                        timestamp_s=timestamp_ms / 1000.0,
                        horizontal=gaze.horizontal,
                        vertical=gaze.vertical,
                        valid=valid_gaze,
                    )

                    pose_metrics = head_pose_temporal.update(
                        timestamp_s=timestamp_ms / 1000.0,
                        pitch_deg=pose_pitch_deg,
                        yaw_deg=pose_yaw_deg,
                        roll_deg=pose_roll_deg,
                        valid=valid_pose_measurement,
                    )

                    perclos_for_fusion = (
                        temporal_metrics.perclos
                        if math.isfinite(temporal_metrics.perclos)
                        else 0.0
                    )

                    fatigue_metrics = fatigue_engine.update(
                        timestamp_s=timestamp_ms / 1000.0,
                        closure_s=(
                            temporal_metrics.closure_duration_s
                        ),
                        perclos_ratio=perclos_for_fusion,
                        gaze_away_s=gaze_metrics.away_duration_s,
                        yawn_event=mouth_metrics.yawn_event,
                        nod_event=pose_metrics.nod_event,
                        face_valid=quality.valid_landmarks,
                        eye_valid=valid_eye_measurement,
                        mouth_valid=valid_mouth_measurement,
                        gaze_valid=valid_gaze,
                        pose_valid=valid_pose_measurement,
                        acknowledged=acknowledge_requested,
                    )
                    acknowledge_requested = False

                    announce_fatigue_state(
                        fatigue_metrics,
                        alert_speaker,
                    )

                    if fatigue_metrics.state_event:
                        print(
                            "EVENTO H5 | "
                            f"estado={fatigue_metrics.state_label} | "
                            f"evento={fatigue_metrics.state_event} | "
                            f"score={fatigue_metrics.score:.3f}"
                        )

                    if temporal_metrics.blink_event:
                        print(
                            "PARPADEO detectado | "
                            f"total={temporal_metrics.blink_count}"
                        )

                    if mouth_metrics.yawn_event:
                        print(
                            "BOSTEZO detectado | "
                            f"total={mouth_metrics.yawn_count}"
                        )

                    if gaze_metrics.gaze_away_event:
                        print(
                            "MIRADA FUERA DEL CENTRO | "
                            f"direccion={gaze_metrics.direction} | "
                            f"total={gaze_metrics.gaze_away_count}"
                        )

                    if pose_metrics.pose_away_event:
                        print(
                            "POSE FUERA DEL FRENTE | "
                            f"direccion={pose_metrics.direction} | "
                            f"total={pose_metrics.pose_away_count}"
                        )

                    if pose_metrics.nod_event:
                        print(
                            "CABECEO detectado | "
                            f"total={pose_metrics.nod_count}"
                        )

                    feature_logger.write(
                        timestamp_ms=timestamp_ms,
                        frame_width_px=frame_width,
                        frame_height_px=frame_height,
                        face_detected=True,
                        landmark_count=len(landmarks),
                        face_width_px=face_width_px,
                        valid_landmarks=quality.valid_landmarks,
                        valid_eye=quality.valid_eye,
                        valid_mouth=quality.valid_mouth,
                        valid_gaze=quality.valid_gaze,
                        valid_pose=quality.valid_pose,
                        valid_all=quality.valid_all,
                        invalid_reason=quality.reason_code,
                        ear=ear,
                        mar=mar,
                        perclos=temporal_metrics.perclos,
                        perclos_window_s=(
                            temporal_metrics.window_duration_s
                        ),
                        closure_s=(
                            temporal_metrics.closure_duration_s
                        ),
                        blink_event=temporal_metrics.blink_event,
                        blink_count=temporal_metrics.blink_count,
                        mouth_open=mouth_metrics.mouth_open,
                        opening_s=mouth_metrics.opening_duration_s,
                        yawn_event=mouth_metrics.yawn_event,
                        yawn_count=mouth_metrics.yawn_count,
                        gaze_h_raw=gaze.horizontal,
                        gaze_v_raw=gaze.vertical,
                        gaze_h_asymmetry=(
                            gaze.horizontal_asymmetry
                        ),
                        gaze_v_asymmetry=(
                            gaze.vertical_asymmetry
                        ),
                        gaze_h=gaze_metrics.horizontal,
                        gaze_v=gaze_metrics.vertical,
                        gaze_direction=gaze_metrics.direction,
                        gaze_away_s=gaze_metrics.away_duration_s,
                        gaze_away_event=(
                            gaze_metrics.gaze_away_event
                        ),
                        gaze_away_count=(
                            gaze_metrics.gaze_away_count
                        ),
                        pitch_deg=pose_pitch_deg,
                        yaw_deg=pose_yaw_deg,
                        roll_deg=pose_roll_deg,
                        pose_scale=pose_scale,
                        pose_fit_error=pose_fit_error,
                        pitch_relative_deg=(
                            pose_metrics.pitch_relative_deg
                        ),
                        yaw_relative_deg=(
                            pose_metrics.yaw_relative_deg
                        ),
                        roll_relative_deg=(
                            pose_metrics.roll_relative_deg
                        ),
                        pose_direction=pose_metrics.direction,
                        pose_away_s=pose_metrics.away_duration_s,
                        pose_away_event=(
                            pose_metrics.pose_away_event
                        ),
                        pose_away_count=(
                            pose_metrics.pose_away_count
                        ),
                        nod_active=pose_metrics.nod_active,
                        nod_s=pose_metrics.nod_duration_s,
                        nod_event=pose_metrics.nod_event,
                        nod_count=pose_metrics.nod_count,
                        valid_ratio=temporal_metrics.valid_ratio,
                        face_latency_ms=face_latency_ms,
                        fusion_score=fatigue_metrics.score,
                        evidence_closure=(
                            fatigue_metrics.evidence.closure
                        ),
                        evidence_perclos=(
                            fatigue_metrics.evidence.perclos
                        ),
                        evidence_yawn=fatigue_metrics.evidence.yawn,
                        evidence_gaze=fatigue_metrics.evidence.gaze,
                        evidence_nod=fatigue_metrics.evidence.nod,
                        fusion_reasons=";".join(
                            fatigue_metrics.reasons
                        ),
                        fatigue_state=fatigue_metrics.state.name,
                        fatigue_state_label=(
                            fatigue_metrics.state_label
                        ),
                        fatigue_state_event=(
                            fatigue_metrics.state_event
                        ),
                        yawns_last_5_min=(
                            fatigue_metrics.yawns_last_5_min
                        ),
                        nods_last_2_min=(
                            fatigue_metrics.nods_last_2_min
                        ),
                        fusion_data_sufficient=(
                            fatigue_metrics.data_sufficient
                        ),
                        fusion_available_weight_ratio=(
                            fatigue_metrics.available_weight_ratio
                        ),
                    )

                    # Ojo derecho del sujeto: verde.
                    draw_landmark_group(
                        bgr,
                        landmarks,
                        RIGHT_EYE,
                        GREEN,
                    )

                    # Ojo izquierdo del sujeto: azul.
                    draw_landmark_group(
                        bgr,
                        landmarks,
                        LEFT_EYE,
                        BLUE,
                    )

                    # Iris derecho: amarillo.
                    draw_landmark_group(
                        bgr,
                        landmarks,
                        RIGHT_IRIS,
                        YELLOW,
                    )

                    # Iris izquierdo: cian.
                    draw_landmark_group(
                        bgr,
                        landmarks,
                        LEFT_IRIS,
                        CYAN,
                    )

                    put_line(
                        bgr,
                        (
                            f"Landmarks: {len(landmarks)} | "
                            f"Matriz: "
                            f"{'SI' if matrix_available else 'NO'}"
                        ),
                        0,
                        GREEN,
                    )

                    if valid_ear:
                        put_line(
                            bgr,
                            f"EAR promedio: {ear:.4f}",
                            1,
                            GREEN
                            if valid_eye_measurement
                            else YELLOW,
                        )
                    else:
                        put_line(
                            bgr,
                            "EAR: NO VALIDO",
                            1,
                            RED,
                        )

                    if valid_mar:
                        put_line(
                            bgr,
                            (
                                f"MAR: {mar:.4f} | "
                                "Apertura: "
                                f"{mouth_metrics.opening_duration_s:.2f}s"
                            ),
                            2,
                            GREEN
                            if valid_mouth_measurement
                            else YELLOW,
                        )
                    else:
                        put_line(
                            bgr,
                            "MAR: NO VALIDO",
                            2,
                            RED,
                        )

                    if valid_gaze:
                        put_line(
                            bgr,
                            (
                                "Mirada H/V: "
                                f"{gaze_metrics.horizontal:.3f} / "
                                f"{gaze_metrics.vertical:.3f} | "
                                f"{gaze_metrics.direction}"
                            ),
                            3,
                            GREEN,
                        )

                        put_line(
                            bgr,
                            (
                                "Fuera: "
                                f"{gaze_metrics.away_duration_s:.2f}s | "
                                "Eventos: "
                                f"{gaze_metrics.gaze_away_count}"
                            ),
                            4,
                            RED
                            if gaze_metrics.distraction_active
                            else (
                                YELLOW
                                if gaze_metrics.off_center
                                else GREEN
                            ),
                        )

                    else:
                        put_line(
                            bgr,
                            "Mirada: NO VALIDA",
                            3,
                            RED,
                        )

                        put_line(
                            bgr,
                            (
                                "Eventos mirada: "
                                f"{gaze_metrics.gaze_away_count}"
                            ),
                            4,
                            YELLOW,
                        )

                    if valid_pose_measurement:
                        put_line(
                            bgr,
                            (
                                "Pose P/Y/R: "
                                f"{pose.pitch_deg:.1f} / "
                                f"{pose.yaw_deg:.1f} / "
                                f"{pose.roll_deg:.1f} deg"
                            ),
                            5,
                            GREEN,
                        )

                        put_line(
                            bgr,
                            (
                                "Pose relativa P/Y/R: "
                                f"{pose_metrics.pitch_relative_deg:.1f} / "
                                f"{pose_metrics.yaw_relative_deg:.1f} / "
                                f"{pose_metrics.roll_relative_deg:.1f} | "
                                f"{pose_metrics.direction}"
                            ),
                            6,
                            GREEN,
                        )
                    else:
                        put_line(
                            bgr,
                            "Pose: NO VALIDA",
                            5,
                            RED,
                        )

                        put_line(
                            bgr,
                            (
                                "Pose relativa: SIN MEDICION | "
                                f"{pose_metrics.direction}"
                            ),
                            6,
                            RED,
                        )

                    put_line(
                        bgr,
                        (
                            "Pose fuera: "
                            f"{pose_metrics.away_duration_s:.2f}s | "
                            "Eventos: "
                            f"{pose_metrics.pose_away_count}"
                        ),
                        7,
                        RED
                        if pose_metrics.away_duration_s >= 1.50
                        else (
                            YELLOW
                            if pose_metrics.off_center
                            else GREEN
                        ),
                    )

                    put_line(
                        bgr,
                        (
                            "Cabeceo: "
                            f"{pose_metrics.nod_duration_s:.2f}s | "
                            "Eventos: "
                            f"{pose_metrics.nod_count}"
                        ),
                        8,
                        YELLOW
                        if pose_metrics.nod_duration_s > 0.0
                        else GREEN,
                    )

                    if valid_pose_measurement:
                        put_line(
                            bgr,
                            (
                                "Pose escala/error: "
                                f"{pose.scale:.3f} / "
                                f"{pose.fit_error:.4f}"
                            ),
                            9,
                            GREEN,
                        )
                    else:
                        put_line(
                            bgr,
                            "Pose escala/error: SIN MEDICION",
                            9,
                            RED,
                        )

                    put_line(
                        bgr,
                        (
                            "Boca: "
                            f"{'ABIERTA' if mouth_metrics.mouth_open else 'CERRADA'}"
                            " | Bostezos: "
                            f"{mouth_metrics.yawn_count}"
                        ),
                        10,
                        YELLOW
                        if mouth_metrics.mouth_open
                        else GREEN,
                    )

                    if math.isfinite(temporal_metrics.perclos):
                        perclos_text = (
                            "PERCLOS 30 s: "
                            f"{100.0 * temporal_metrics.perclos:.1f}%"
                        )
                    else:
                        perclos_text = "PERCLOS 30 s: CALCULANDO"

                    put_line(
                        bgr,
                        perclos_text,
                        11,
                        GREEN
                        if valid_eye_measurement
                        else YELLOW,
                    )

                    put_line(
                        bgr,
                        (
                            "Cierre: "
                            f"{temporal_metrics.closure_duration_s:.2f}s | "
                            "Parpadeos: "
                            f"{temporal_metrics.blink_count}"
                        ),
                        12,
                        YELLOW
                        if temporal_metrics.closure_duration_s > 0
                        else GREEN,
                    )

                    put_line(
                        bgr,
                        (
                            "Datos oculares validos: "
                            f"{100.0 * temporal_metrics.valid_ratio:.1f}%"
                        ),
                        13,
                        GREEN
                        if temporal_metrics.valid_ratio >= 0.80
                        else YELLOW,
                    )

                    put_line(
                        bgr,
                        (
                            "Rostro: "
                            f"{face_width_px:.1f}px | Calidad: "
                            f"{'COMPLETA' if quality.valid_all else 'PARCIAL'}"
                        ),
                        14,
                        GREEN if quality.valid_all else YELLOW,
                    )

                    put_line(
                        bgr,
                        (
                            "H5 Score: "
                            f"{fatigue_metrics.score:.3f} | "
                            f"{fatigue_metrics.state_label}"
                        ),
                        15,
                        fatigue_state_color(fatigue_metrics.state),
                    )

                    evidence_text = (
                        ",".join(fatigue_metrics.reasons)
                        if fatigue_metrics.reasons
                        else "ninguna"
                    )

                    put_line(
                        bgr,
                        f"H5 Evidencias: {evidence_text}",
                        16,
                        fatigue_state_color(fatigue_metrics.state),
                    )

                    put_line(
                        bgr,
                        (
                            "Recientes: Bostezos "
                            f"{fatigue_metrics.yawns_last_5_min} (5m) | "
                            "Cabeceos "
                            f"{fatigue_metrics.nods_last_2_min} (2m)"
                        ),
                        17,
                        GREEN,
                    )

                    if fatigue_metrics.state == State.CRITICAL:
                        if fatigue_metrics.alert_acknowledged:
                            critical_status_text = (
                                "Reconocida: SI | Estable: "
                                f"{fatigue_metrics.critical_recovery_elapsed_s:.1f}"
                                " / "
                                f"{fatigue_metrics.critical_recovery_required_s:.0f}s"
                            )
                            critical_status_color = YELLOW
                        else:
                            critical_status_text = (
                                "Reconocida: NO | Di entendido o pulsa A"
                            )
                            critical_status_color = RED

                        put_line(
                            bgr,
                            critical_status_text,
                            18,
                            critical_status_color,
                        )
                    else:
                        put_line(
                            bgr,
                            (
                                "Fusion: "
                                f"{'DATOS OK' if fatigue_metrics.data_sufficient else 'INSUFICIENTE'}"
                                " | Disponible: "
                                f"{100.0 * fatigue_metrics.available_weight_ratio:.0f}%"
                            ),
                            18,
                            GREEN
                            if fatigue_metrics.data_sufficient
                            else RED,
                        )

                    put_line(
                        bgr,
                        voice_status_text(
                            voice_recognizer,
                            alert_speaker,
                            last_voice_command,
                        ),
                        19,
                        WHITE,
                    )

                    if not first_detection_reported:
                        print("Primer rostro detectado")
                        print(
                            "Cantidad de landmarks:",
                            len(landmarks),
                        )
                        print(
                            "Matriz facial disponible:",
                            matrix_available,
                        )
                        print(
                            "Ancho mínimo requerido:",
                            f"{minimum_face_width:.1f} px",
                        )

                        first_detection_reported = True

                    # Escribe un valor aproximadamente cada segundo.
                    if (
                        timestamp_ms - last_console_report_ms
                        >= 1000
                    ):
                        print(
                            f"EAR={ear:.4f} | "
                            f"MAR={mar:.4f} | "
                            f"gaze_h={gaze.horizontal:.3f} | "
                            f"gaze_v={gaze.vertical:.3f} | "
                            f"gaze_valid={valid_gaze} | "
                            f"gaze_dir={gaze_metrics.direction} | "
                            f"gaze_away={gaze_metrics.away_duration_s:.2f}s | "
                            "gaze_events="
                            f"{gaze_metrics.gaze_away_count} | "
                            f"pitch={pose_pitch_deg:.2f} | "
                            f"yaw={pose_yaw_deg:.2f} | "
                            f"roll={pose_roll_deg:.2f} | "
                            f"pose_error={pose_fit_error:.4f} | "
                            f"pose_valid={valid_pose_measurement} | "
                            "pose_rel="
                            f"{pose_metrics.pitch_relative_deg:.2f}/"
                            f"{pose_metrics.yaw_relative_deg:.2f}/"
                            f"{pose_metrics.roll_relative_deg:.2f} | "
                            f"pose_dir={pose_metrics.direction} | "
                            f"pose_away={pose_metrics.away_duration_s:.2f}s | "
                            "pose_events="
                            f"{pose_metrics.pose_away_count} | "
                            f"nod={pose_metrics.nod_duration_s:.2f}s | "
                            f"nods={pose_metrics.nod_count} | "
                            f"PERCLOS={temporal_metrics.perclos:.3f} | "
                            f"cierre={temporal_metrics.closure_duration_s:.2f}s | "
                            f"parpadeos={temporal_metrics.blink_count} | "
                            f"apertura={mouth_metrics.opening_duration_s:.2f}s | "
                            f"bostezos={mouth_metrics.yawn_count} | "
                            f"valid_ratio={temporal_metrics.valid_ratio:.3f} | "
                            f"ancho={face_width_px:.1f}px | "
                            f"quality={quality.reason_code} | "
                            f"fusion_score={fatigue_metrics.score:.3f} | "
                            f"fatigue_state={fatigue_metrics.state.name} | "
                            "fusion_valid="
                            f"{fatigue_metrics.data_sufficient} | "
                            "yawns_5m="
                            f"{fatigue_metrics.yawns_last_5_min} | "
                            "nods_2m="
                            f"{fatigue_metrics.nods_last_2_min} | "
                            "ack="
                            f"{fatigue_metrics.alert_acknowledged} | "
                            "critical_recovery="
                            f"{fatigue_metrics.critical_recovery_elapsed_s:.1f}s"
                        )

                        last_console_report_ms = timestamp_ms

                else:
                    quality = assess_frame_quality(
                        face_detected=False,
                    )

                    temporal_metrics = eye_temporal.update(
                        timestamp_s=timestamp_ms / 1000.0,
                        ear=float("nan"),
                        valid=False,
                    )

                    mouth_metrics = mouth_temporal.update(
                        timestamp_s=timestamp_ms / 1000.0,
                        mar=float("nan"),
                        valid=False,
                    )

                    gaze_metrics = gaze_temporal.update(
                        timestamp_s=timestamp_ms / 1000.0,
                        horizontal=float("nan"),
                        vertical=float("nan"),
                        valid=False,
                    )

                    pose_metrics = head_pose_temporal.update(
                        timestamp_s=timestamp_ms / 1000.0,
                        pitch_deg=float("nan"),
                        yaw_deg=float("nan"),
                        roll_deg=float("nan"),
                        valid=False,
                    )

                    fatigue_metrics = fatigue_engine.update(
                        timestamp_s=timestamp_ms / 1000.0,
                        closure_s=float("nan"),
                        perclos_ratio=float("nan"),
                        gaze_away_s=float("nan"),
                        yawn_event=mouth_metrics.yawn_event,
                        nod_event=pose_metrics.nod_event,
                        face_valid=False,
                        eye_valid=False,
                        mouth_valid=False,
                        gaze_valid=False,
                        pose_valid=False,
                        acknowledged=acknowledge_requested,
                    )
                    acknowledge_requested = False

                    announce_fatigue_state(
                        fatigue_metrics,
                        alert_speaker,
                    )

                    feature_logger.write(
                        timestamp_ms=timestamp_ms,
                        frame_width_px=frame_width,
                        frame_height_px=frame_height,
                        face_detected=False,
                        landmark_count=0,
                        valid_landmarks=quality.valid_landmarks,
                        valid_eye=quality.valid_eye,
                        valid_mouth=quality.valid_mouth,
                        valid_gaze=quality.valid_gaze,
                        valid_pose=quality.valid_pose,
                        valid_all=quality.valid_all,
                        invalid_reason=quality.reason_code,
                        perclos=temporal_metrics.perclos,
                        perclos_window_s=(
                            temporal_metrics.window_duration_s
                        ),
                        closure_s=(
                            temporal_metrics.closure_duration_s
                        ),
                        blink_event=temporal_metrics.blink_event,
                        blink_count=temporal_metrics.blink_count,
                        mouth_open=mouth_metrics.mouth_open,
                        opening_s=mouth_metrics.opening_duration_s,
                        yawn_event=mouth_metrics.yawn_event,
                        yawn_count=mouth_metrics.yawn_count,
                        gaze_h=gaze_metrics.horizontal,
                        gaze_v=gaze_metrics.vertical,
                        gaze_direction=gaze_metrics.direction,
                        gaze_away_s=gaze_metrics.away_duration_s,
                        gaze_away_event=(
                            gaze_metrics.gaze_away_event
                        ),
                        gaze_away_count=(
                            gaze_metrics.gaze_away_count
                        ),
                        pitch_relative_deg=(
                            pose_metrics.pitch_relative_deg
                        ),
                        yaw_relative_deg=(
                            pose_metrics.yaw_relative_deg
                        ),
                        roll_relative_deg=(
                            pose_metrics.roll_relative_deg
                        ),
                        pose_direction=pose_metrics.direction,
                        pose_away_s=pose_metrics.away_duration_s,
                        pose_away_event=(
                            pose_metrics.pose_away_event
                        ),
                        pose_away_count=(
                            pose_metrics.pose_away_count
                        ),
                        nod_active=pose_metrics.nod_active,
                        nod_s=pose_metrics.nod_duration_s,
                        nod_event=pose_metrics.nod_event,
                        nod_count=pose_metrics.nod_count,
                        valid_ratio=temporal_metrics.valid_ratio,
                        face_latency_ms=face_latency_ms,
                        fusion_score=fatigue_metrics.score,
                        evidence_closure=(
                            fatigue_metrics.evidence.closure
                        ),
                        evidence_perclos=(
                            fatigue_metrics.evidence.perclos
                        ),
                        evidence_yawn=fatigue_metrics.evidence.yawn,
                        evidence_gaze=fatigue_metrics.evidence.gaze,
                        evidence_nod=fatigue_metrics.evidence.nod,
                        fusion_reasons=";".join(
                            fatigue_metrics.reasons
                        ),
                        fatigue_state=fatigue_metrics.state.name,
                        fatigue_state_label=(
                            fatigue_metrics.state_label
                        ),
                        fatigue_state_event=(
                            fatigue_metrics.state_event
                        ),
                        yawns_last_5_min=(
                            fatigue_metrics.yawns_last_5_min
                        ),
                        nods_last_2_min=(
                            fatigue_metrics.nods_last_2_min
                        ),
                        fusion_data_sufficient=(
                            fatigue_metrics.data_sufficient
                        ),
                        fusion_available_weight_ratio=(
                            fatigue_metrics.available_weight_ratio
                        ),
                    )

                    put_line(
                        bgr,
                        "SIN ROSTRO",
                        0,
                        RED,
                    )

                    put_line(
                        bgr,
                        "EAR: SIN MEDICION",
                        1,
                        RED,
                    )

                    put_line(
                        bgr,
                        "MAR: SIN MEDICION",
                        2,
                        RED,
                    )

                    put_line(
                        bgr,
                        (
                            "Mirada: SIN MEDICION | Eventos: "
                            f"{gaze_metrics.gaze_away_count}"
                        ),
                        3,
                        RED,
                    )

                    put_line(
                        bgr,
                        (
                            "Pose: SIN MEDICION | Eventos: "
                            f"{pose_metrics.pose_away_count}"
                        ),
                        4,
                        RED,
                    )

                    put_line(
                        bgr,
                        (
                            "Cabeceos: "
                            f"{pose_metrics.nod_count}"
                        ),
                        5,
                        RED,
                    )

                    if math.isfinite(temporal_metrics.perclos):
                        invalid_perclos_text = (
                            "PERCLOS previo: "
                            f"{100.0 * temporal_metrics.perclos:.1f}%"
                        )
                    else:
                        invalid_perclos_text = "PERCLOS: SIN DATOS"

                    put_line(
                        bgr,
                        invalid_perclos_text,
                        6,
                        YELLOW,
                    )

                    put_line(
                        bgr,
                        (
                            "Parpadeos: "
                            f"{temporal_metrics.blink_count} | "
                            "Bostezos: "
                            f"{mouth_metrics.yawn_count}"
                        ),
                        7,
                        YELLOW,
                    )

                    put_line(
                        bgr,
                        (
                            "Datos oculares validos: "
                            f"{100.0 * temporal_metrics.valid_ratio:.1f}%"
                        ),
                        8,
                        YELLOW,
                    )

                    put_line(
                        bgr,
                        "El frame no se usa como cierre ni bostezo",
                        9,
                        YELLOW,
                    )

                    put_line(
                        bgr,
                        (
                            "H5 Estado conservado: "
                            f"{fatigue_metrics.state_label} | "
                            f"Score: {fatigue_metrics.score:.3f}"
                        ),
                        10,
                        fatigue_state_color(fatigue_metrics.state),
                    )

                    put_line(
                        bgr,
                        "Fusion: DATOS INSUFICIENTES",
                        11,
                        RED,
                    )

                    put_line(
                        bgr,
                        (
                            "Recientes: Bostezos "
                            f"{fatigue_metrics.yawns_last_5_min} (5m) | "
                            "Cabeceos "
                            f"{fatigue_metrics.nods_last_2_min} (2m)"
                        ),
                        12,
                        YELLOW,
                    )

                    if fatigue_metrics.state == State.CRITICAL:
                        put_line(
                            bgr,
                            (
                                "Reconocida: "
                                f"{'SI' if fatigue_metrics.alert_acknowledged else 'NO'}"
                                " | Progreso pausado sin rostro"
                            ),
                            13,
                            YELLOW,
                        )
                    else:
                        put_line(
                            bgr,
                            "Transicion pendiente cancelada",
                            13,
                            YELLOW,
                        )

                    put_line(
                        bgr,
                        voice_status_text(
                            voice_recognizer,
                            alert_speaker,
                            last_voice_command,
                        ),
                        14,
                        WHITE,
                    )

                    if (
                        timestamp_ms - last_console_report_ms
                        >= 1000
                    ):
                        print(
                            "SIN ROSTRO | "
                            f"PERCLOS={temporal_metrics.perclos:.3f} | "
                            "parpadeos="
                            f"{temporal_metrics.blink_count} | "
                            "bostezos="
                            f"{mouth_metrics.yawn_count} | "
                            "mirada="
                            f"{gaze_metrics.gaze_away_count} | "
                            "pose_events="
                            f"{pose_metrics.pose_away_count} | "
                            "cabeceos="
                            f"{pose_metrics.nod_count} | "
                            f"quality={quality.reason_code} | "
                            "valid_ratio="
                            f"{temporal_metrics.valid_ratio:.3f} | "
                            f"fusion_score={fatigue_metrics.score:.3f} | "
                            f"fatigue_state={fatigue_metrics.state.name} | "
                            "fusion_valid=False | "
                            "yawns_5m="
                            f"{fatigue_metrics.yawns_last_5_min} | "
                            "nods_2m="
                            f"{fatigue_metrics.nods_last_2_min} | "
                            "ack="
                            f"{fatigue_metrics.alert_acknowledged} | "
                            "critical_recovery="
                            f"{fatigue_metrics.critical_recovery_elapsed_s:.1f}s"
                        )

                        last_console_report_ms = timestamp_ms

                cv2.imshow(
                    "Hito H6 - Fatiga multimodal y voz offline",
                    bgr,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("a"):
                    acknowledge_requested = True
                    last_voice_command = "A_MANUAL"

                    if alert_speaker is not None:
                        alert_speaker.cancel(
                            clear_pending=True
                        )

                    print(
                        "Reconocimiento solicitado para alerta grave."
                    )

    finally:
        if voice_recognizer is not None:
            voice_recognizer.stop()
            print(
                "Reconocedor Vosk liberado correctamente."
            )

        if alert_speaker is not None:
            alert_speaker.stop()
            print(
                "Emisor Piper liberado correctamente."
            )

        if feature_logger is not None:
            feature_logger.close()
            print(
                "Frames guardados:",
                feature_logger.frame_index,
            )
            print("CSV actualizado:", FEATURES_CSV)

        cap.release()
        cv2.destroyAllWindows()
        print("Cámara liberada correctamente.")


if __name__ == "__main__":
    main()
