from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class HeadPose:
    """Ángulos de orientación extraídos de la matriz facial."""

    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    scale: float
    fit_error: float


def normalize_transformation_matrix(matrix) -> np.ndarray:
    """
    Convierte la matriz de MediaPipe a un arreglo 4x4.

    Acepta una matriz 4x4 o una secuencia plana de 16 valores.
    """

    matrix_array = np.asarray(
        matrix,
        dtype=np.float64,
    )

    matrix_array = np.squeeze(matrix_array)

    if matrix_array.size != 16:
        raise ValueError(
            "La matriz facial debe contener 16 valores."
        )

    matrix_array = matrix_array.reshape(4, 4)

    if not np.all(np.isfinite(matrix_array)):
        raise ValueError(
            "La matriz facial contiene valores no finitos."
        )

    return matrix_array


def nearest_rotation_matrix(linear_matrix):
    """
    Separa la rotación del escalamiento mediante SVD.

    La matriz facial puede contener escala además de rotación.
    La descomposición obtiene la rotación ortonormal más cercana.
    """

    linear_matrix = np.asarray(
        linear_matrix,
        dtype=np.float64,
    )

    if linear_matrix.shape != (3, 3):
        raise ValueError(
            "El bloque lineal debe tener forma 3x3."
        )

    u_matrix, singular_values, vh_matrix = (
        np.linalg.svd(linear_matrix)
    )

    rotation = u_matrix @ vh_matrix

    if np.linalg.det(rotation) < 0:
        u_matrix[:, -1] *= -1.0
        rotation = u_matrix @ vh_matrix

    scale = float(
        np.mean(singular_values)
    )

    denominator = max(
        float(np.linalg.norm(linear_matrix)),
        1e-12,
    )

    fit_error = float(
        np.linalg.norm(
            linear_matrix - scale * rotation
        )
        / denominator
    )

    return rotation, scale, fit_error


def rotation_to_euler_degrees(rotation) -> tuple:
    """
    Convierte una rotación 3x3 a pitch, yaw y roll.

    Convención utilizada:

    - pitch: rotación alrededor del eje X.
    - yaw: rotación alrededor del eje Y.
    - roll: rotación alrededor del eje Z.
    - composición: Rz(roll) @ Ry(yaw) @ Rx(pitch).
    """

    rotation = np.asarray(
        rotation,
        dtype=np.float64,
    )

    if rotation.shape != (3, 3):
        raise ValueError(
            "La matriz de rotación debe tener forma 3x3."
        )

    sin_yaw = float(
        np.clip(
            -rotation[2, 0],
            -1.0,
            1.0,
        )
    )

    yaw = math.asin(sin_yaw)
    cos_yaw = math.cos(yaw)

    if abs(cos_yaw) > 1e-6:
        pitch = math.atan2(
            rotation[2, 1],
            rotation[2, 2],
        )

        roll = math.atan2(
            rotation[1, 0],
            rotation[0, 0],
        )
    else:
        # Caso próximo al bloqueo de cardán.
        pitch = math.atan2(
            -rotation[1, 2],
            rotation[1, 1],
        )

        roll = 0.0

    return (
        math.degrees(pitch),
        math.degrees(yaw),
        math.degrees(roll),
    )


def extract_head_pose(matrix) -> HeadPose:
    """Extrae pitch, yaw y roll de una matriz facial 4x4."""

    transformation = normalize_transformation_matrix(
        matrix
    )

    linear_matrix = transformation[:3, :3]

    rotation, scale, fit_error = (
        nearest_rotation_matrix(linear_matrix)
    )

    pitch_deg, yaw_deg, roll_deg = (
        rotation_to_euler_degrees(rotation)
    )

    return HeadPose(
        pitch_deg=float(pitch_deg),
        yaw_deg=float(yaw_deg),
        roll_deg=float(roll_deg),
        scale=scale,
        fit_error=fit_error,
    )