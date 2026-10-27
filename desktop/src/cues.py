from dataclasses import dataclass
from math import hypot

import numpy as np


# Índices iniciales para la topología de 478 landmarks.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)

RIGHT_IRIS = (469, 470, 471, 472)
LEFT_IRIS = (474, 475, 476, 477)

MOUTH_CORNERS = (61, 291)
MOUTH_VERTICAL = ((13, 14), (82, 87), (312, 317))


@dataclass(frozen=True)
class P2:
    """Punto bidimensional normalizado."""

    x: float
    y: float


def p2(landmarks, index: int) -> P2:
    """Obtiene un landmark y lo convierte a un punto 2D."""

    point = landmarks[index]
    return P2(float(point.x), float(point.y))


def dist(a: P2, b: P2) -> float:
    """Calcula la distancia euclidiana entre dos puntos."""

    return hypot(a.x - b.x, a.y - b.y)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Limita un número al intervalo indicado."""

    return max(minimum, min(maximum, value))


def eye_aspect_ratio(landmarks, indices) -> float:
    """Calcula el EAR de un ojo."""

    p1, p2_vertical, p3, p4, p5, p6 = [
        p2(landmarks, index) for index in indices
    ]

    denominator = 2.0 * dist(p1, p4)

    if denominator < 1e-8:
        return float("nan")

    numerator = dist(p2_vertical, p6) + dist(p3, p5)

    return numerator / denominator


def mean_ear(landmarks) -> float:
    """Calcula el EAR promedio de ambos ojos."""

    values = [
        eye_aspect_ratio(landmarks, RIGHT_EYE),
        eye_aspect_ratio(landmarks, LEFT_EYE),
    ]

    valid_values = [
        value for value in values
        if np.isfinite(value)
    ]

    if not valid_values:
        return float("nan")

    return float(np.mean(valid_values))

def mouth_aspect_ratio(landmarks) -> float:
    """
    Calcula el Mouth Aspect Ratio o MAR.

    Un valor mayor representa una apertura vertical
    más grande de la boca.
    """

    left_corner = p2(
        landmarks,
        MOUTH_CORNERS[0],
    )

    right_corner = p2(
        landmarks,
        MOUTH_CORNERS[1],
    )

    horizontal_distance = dist(
        left_corner,
        right_corner,
    )

    if horizontal_distance < 1e-8:
        return float("nan")

    vertical_distances = [
        dist(
            p2(landmarks, upper_index),
            p2(landmarks, lower_index),
        )
        for upper_index, lower_index in MOUTH_VERTICAL
    ]

    vertical_mean = sum(vertical_distances) / 3.0

    return vertical_mean / horizontal_distance

@dataclass(frozen=True)
class GazeRatios:
    """Posición normalizada del iris dentro de ambos ojos."""

    horizontal: float
    vertical: float
    horizontal_asymmetry: float
    vertical_asymmetry: float


def iris_center(landmarks, indices) -> P2:
    """Calcula el centro promedio de los landmarks del iris."""

    points = [
        p2(landmarks, index)
        for index in indices
    ]

    if not points:
        return P2(
            float("nan"),
            float("nan"),
        )

    center_x = sum(
        point.x for point in points
    ) / len(points)

    center_y = sum(
        point.y for point in points
    ) / len(points)

    return P2(center_x, center_y)


def project_on_axis(
    point: P2,
    origin: P2,
    axis_x: float,
    axis_y: float,
) -> float:
    """Proyecta un punto sobre un eje bidimensional."""

    return (
        (point.x - origin.x) * axis_x
        + (point.y - origin.y) * axis_y
    )


def eye_gaze_ratios(
    landmarks,
    eye_indices,
    iris_indices,
) -> P2:
    """
    Calcula las razones horizontal y vertical de un ojo.

    El cálculo utiliza un sistema local definido por los
    extremos del ojo, por lo que tolera cierta rotación facial.
    """

    if len(eye_indices) != 6:
        raise ValueError(
            "eye_indices debe contener seis landmarks."
        )

    corner_a = p2(
        landmarks,
        eye_indices[0],
    )

    corner_b = p2(
        landmarks,
        eye_indices[3],
    )

    # Ordenamos los extremos según la imagen.
    if corner_a.x <= corner_b.x:
        image_left = corner_a
        image_right = corner_b
    else:
        image_left = corner_b
        image_right = corner_a

    eye_axis_x = image_right.x - image_left.x
    eye_axis_y = image_right.y - image_left.y

    eye_width = hypot(
        eye_axis_x,
        eye_axis_y,
    )

    if eye_width < 1e-8:
        return P2(
            float("nan"),
            float("nan"),
        )

    # Eje horizontal unitario.
    horizontal_axis_x = eye_axis_x / eye_width
    horizontal_axis_y = eye_axis_y / eye_width

    # Eje perpendicular orientado hacia abajo en la imagen.
    vertical_axis_x = -horizontal_axis_y
    vertical_axis_y = horizontal_axis_x

    if vertical_axis_y < 0:
        vertical_axis_x *= -1.0
        vertical_axis_y *= -1.0

    iris = iris_center(
        landmarks,
        iris_indices,
    )

    iris_horizontal = project_on_axis(
        iris,
        image_left,
        horizontal_axis_x,
        horizontal_axis_y,
    )

    horizontal_ratio = (
        iris_horizontal / eye_width
    )

    upper_points = [
        p2(landmarks, eye_indices[1]),
        p2(landmarks, eye_indices[2]),
    ]

    lower_points = [
        p2(landmarks, eye_indices[4]),
        p2(landmarks, eye_indices[5]),
    ]

    upper_position = sum(
        project_on_axis(
            point,
            image_left,
            vertical_axis_x,
            vertical_axis_y,
        )
        for point in upper_points
    ) / len(upper_points)

    lower_position = sum(
        project_on_axis(
            point,
            image_left,
            vertical_axis_x,
            vertical_axis_y,
        )
        for point in lower_points
    ) / len(lower_points)

    top_position = min(
        upper_position,
        lower_position,
    )

    bottom_position = max(
        upper_position,
        lower_position,
    )

    eye_height = (
        bottom_position - top_position
    )

    if eye_height < 1e-8:
        return P2(
            float("nan"),
            float("nan"),
        )

    iris_vertical = project_on_axis(
        iris,
        image_left,
        vertical_axis_x,
        vertical_axis_y,
    )

    vertical_ratio = (
        iris_vertical - top_position
    ) / eye_height

    return P2(
        clamp(horizontal_ratio),
        clamp(vertical_ratio),
    )


def mean_gaze_ratios(landmarks) -> GazeRatios:
    """Combina las razones geométricas de ambos ojos."""

    right_ratios = eye_gaze_ratios(
        landmarks,
        RIGHT_EYE,
        RIGHT_IRIS,
    )

    left_ratios = eye_gaze_ratios(
        landmarks,
        LEFT_EYE,
        LEFT_IRIS,
    )

    values = (
        right_ratios.x,
        right_ratios.y,
        left_ratios.x,
        left_ratios.y,
    )

    if not all(np.isfinite(value) for value in values):
        return GazeRatios(
            horizontal=float("nan"),
            vertical=float("nan"),
            horizontal_asymmetry=float("nan"),
            vertical_asymmetry=float("nan"),
        )

    return GazeRatios(
        horizontal=float(
            np.mean(
                [
                    right_ratios.x,
                    left_ratios.x,
                ]
            )
        ),
        vertical=float(
            np.mean(
                [
                    right_ratios.y,
                    left_ratios.y,
                ]
            )
        ),
        horizontal_asymmetry=abs(
            right_ratios.x - left_ratios.x
        ),
        vertical_asymmetry=abs(
            right_ratios.y - left_ratios.y
        ),
    )