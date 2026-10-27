from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Ruta absoluta construida automáticamente desde este archivo.
MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "face_landmarker.task"
)

# Cambia este número solamente si /dev/video0 no funciona.
CAMERA_INDEX = 0


def main():
    print("Modelo:", MODEL)
    print("Cámara seleccionada:", CAMERA_INDEX)

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

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ok, bgr = cap.read()

                if not ok:
                    print("No se pudo obtener una imagen de la cámara.")
                    break

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

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    point_33 = landmarks[33]

                    matrix_available = bool(
                        result.facial_transformation_matrixes
                    )

                    text = (
                        f"landmarks={len(landmarks)} "
                        f"p33=({point_33.x:.3f},{point_33.y:.3f}) "
                        f"matrix={'SI' if matrix_available else 'NO'}"
                    )

                    color = (0, 255, 0)

                    if not first_detection_reported:
                        print("Primer rostro detectado")
                        print("Cantidad de landmarks:", len(landmarks))
                        print(
                            "Matriz facial disponible:",
                            matrix_available,
                        )
                        first_detection_reported = True
                else:
                    text = "SIN ROSTRO"
                    color = (0, 0, 255)

                cv2.putText(
                    bgr,
                    text,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    color,
                    2,
                )

                cv2.putText(
                    bgr,
                    "Presiona Q para cerrar",
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Face Landmarker - Hito H2", bgr)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Cámara liberada correctamente.")


if __name__ == "__main__":
    main()
