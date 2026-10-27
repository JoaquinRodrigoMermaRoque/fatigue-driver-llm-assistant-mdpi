"""Control de calidad independiente para los indicadores faciales."""

from dataclasses import dataclass
import math


EXPECTED_LANDMARKS = 478
GAZE_MIN_EAR = 0.15
MAX_GAZE_ASYMMETRY = 0.20
MAX_POSE_FIT_ERROR = 0.05


def _finite(*values: float) -> bool:
    """Indica si todos los valores recibidos son números finitos."""

    return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class FrameQuality:
    """Resultado de calidad de un frame para cada grupo de variables."""

    face_detected: bool
    valid_landmarks: bool
    valid_eye: bool
    valid_mouth: bool
    valid_gaze: bool
    valid_pose: bool
    valid_all: bool
    invalid_reasons: tuple[str, ...]

    @property
    def reason_code(self) -> str:
        """Devuelve razones compactas para almacenarlas en un CSV."""

        if not self.invalid_reasons:
            return "ok"

        return "|".join(self.invalid_reasons)


def assess_frame_quality(
    *,
    face_detected: bool,
    landmark_count: int = 0,
    face_width_px: float = float("nan"),
    minimum_face_width_px: float = 100.0,
    ear: float = float("nan"),
    mar: float = float("nan"),
    gaze_horizontal: float = float("nan"),
    gaze_vertical: float = float("nan"),
    gaze_horizontal_asymmetry: float = float("nan"),
    gaze_vertical_asymmetry: float = float("nan"),
    matrix_available: bool = False,
    pose_pitch_deg: float = float("nan"),
    pose_yaw_deg: float = float("nan"),
    pose_roll_deg: float = float("nan"),
    pose_scale: float = float("nan"),
    pose_fit_error: float = float("nan"),
    expected_landmarks: int = EXPECTED_LANDMARKS,
    gaze_min_ear: float = GAZE_MIN_EAR,
    max_gaze_asymmetry: float = MAX_GAZE_ASYMMETRY,
    max_pose_fit_error: float = MAX_POSE_FIT_ERROR,
) -> FrameQuality:
    """
    Evalúa la calidad sin mezclar la validez de indicadores distintos.

    Un frame puede ser válido para EAR y PERCLOS, pero inválido para
    mirada, boca o pose. Esto evita eliminar información útil durante
    la ablación.
    """

    if not face_detected:
        return FrameQuality(
            face_detected=False,
            valid_landmarks=False,
            valid_eye=False,
            valid_mouth=False,
            valid_gaze=False,
            valid_pose=False,
            valid_all=False,
            invalid_reasons=("no_face",),
        )

    reasons: list[str] = []

    valid_landmark_count = landmark_count == expected_landmarks
    valid_face_width = (
        _finite(face_width_px, minimum_face_width_px)
        and face_width_px >= minimum_face_width_px
    )
    valid_landmarks = valid_landmark_count and valid_face_width

    if not valid_landmark_count:
        reasons.append("landmark_count")

    if not _finite(face_width_px, minimum_face_width_px):
        reasons.append("face_width_nonfinite")
    elif face_width_px < minimum_face_width_px:
        reasons.append("face_too_small")

    valid_ear_value = _finite(ear)
    valid_mar_value = _finite(mar)
    valid_eye = valid_landmarks and valid_ear_value
    valid_mouth = valid_landmarks and valid_mar_value

    if valid_landmarks and not valid_ear_value:
        reasons.append("ear_nonfinite")

    if valid_landmarks and not valid_mar_value:
        reasons.append("mar_nonfinite")

    valid_gaze_values = _finite(
        gaze_horizontal,
        gaze_vertical,
        gaze_horizontal_asymmetry,
        gaze_vertical_asymmetry,
    )
    eyes_open_for_gaze = valid_eye and ear >= gaze_min_ear
    valid_gaze_asymmetry = (
        valid_gaze_values
        and gaze_horizontal_asymmetry <= max_gaze_asymmetry
        and gaze_vertical_asymmetry <= max_gaze_asymmetry
    )
    valid_gaze = (
        eyes_open_for_gaze
        and valid_gaze_values
        and valid_gaze_asymmetry
    )

    if valid_eye:
        if not eyes_open_for_gaze:
            reasons.append("eyes_closed_for_gaze")
        elif not valid_gaze_values:
            reasons.append("gaze_nonfinite")
        elif not valid_gaze_asymmetry:
            reasons.append("gaze_asymmetry")

    valid_pose_values = _finite(
        pose_pitch_deg,
        pose_yaw_deg,
        pose_roll_deg,
        pose_scale,
        pose_fit_error,
    )
    valid_pose_error = (
        valid_pose_values
        and pose_fit_error <= max_pose_fit_error
    )
    valid_pose = (
        valid_landmarks
        and matrix_available
        and valid_pose_values
        and valid_pose_error
    )

    if valid_landmarks:
        if not matrix_available:
            reasons.append("pose_matrix_missing")
        elif not valid_pose_values:
            reasons.append("pose_nonfinite")
        elif not valid_pose_error:
            reasons.append("pose_fit_error")

    valid_all = (
        valid_eye
        and valid_mouth
        and valid_gaze
        and valid_pose
    )

    return FrameQuality(
        face_detected=True,
        valid_landmarks=valid_landmarks,
        valid_eye=valid_eye,
        valid_mouth=valid_mouth,
        valid_gaze=valid_gaze,
        valid_pose=valid_pose,
        valid_all=valid_all,
        invalid_reasons=tuple(reasons),
    )
