from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EyeSample:
    """Muestra ocular asociada a un instante."""

    timestamp_s: float
    valid: bool
    closed: bool


@dataclass(frozen=True)
class EyeTemporalMetrics:
    """Resultados temporales calculados para el instante actual."""

    perclos: float
    valid_ratio: float
    closure_duration_s: float
    blink_count: int
    blink_event: bool
    window_duration_s: float
    valid_duration_s: float


class EyeTemporalAnalyzer:
    """
    Analiza temporalmente el estado de los ojos.

    Calcula:

    - Duración continua del cierre.
    - Eventos de parpadeo.
    - Cantidad acumulada de parpadeos.
    - PERCLOS dentro de una ventana móvil.
    - Proporción de tiempo con mediciones válidas.

    Los periodos sin rostro o de baja calidad no se consideran
    ojos cerrados.
    """

    def __init__(
        self,
        window_s: float = 30.0,
        ear_threshold: float = 0.21,
        min_blink_s: float = 0.08,
        max_blink_s: float = 0.80,
    ):
        if window_s <= 0:
            raise ValueError("window_s debe ser mayor que cero.")

        if ear_threshold <= 0:
            raise ValueError(
                "ear_threshold debe ser mayor que cero."
            )

        if min_blink_s < 0:
            raise ValueError(
                "min_blink_s no puede ser negativo."
            )

        if max_blink_s <= min_blink_s:
            raise ValueError(
                "max_blink_s debe ser mayor que min_blink_s."
            )

        self.window_s = float(window_s)
        self.ear_threshold = float(ear_threshold)
        self.min_blink_s = float(min_blink_s)
        self.max_blink_s = float(max_blink_s)

        self.samples = deque()

        self.last_timestamp_s = None
        self.closure_start_s = None
        self.previous_closed = False
        self.blink_count = 0

    def reset(self):
        """Reinicia por completo el analizador temporal."""

        self.samples.clear()

        self.last_timestamp_s = None
        self.closure_start_s = None
        self.previous_closed = False
        self.blink_count = 0

    def update(
        self,
        timestamp_s: float,
        ear: float,
        valid: bool,
    ) -> EyeTemporalMetrics:
        """
        Añade una medición y devuelve las métricas actualizadas.

        Un frame inválido se registra para calcular valid_ratio,
        pero nunca se interpreta como ojo cerrado.
        """

        timestamp_s = float(timestamp_s)

        if not math.isfinite(timestamp_s):
            raise ValueError(
                "timestamp_s debe ser un número finito."
            )

        if (
            self.last_timestamp_s is not None
            and timestamp_s <= self.last_timestamp_s
        ):
            raise ValueError(
                "Los timestamps deben ser estrictamente crecientes."
            )

        valid_ear = (
            bool(valid)
            and math.isfinite(float(ear))
        )

        closed = (
            valid_ear
            and float(ear) < self.ear_threshold
        )

        blink_event = False

        if valid_ear:
            if closed and not self.previous_closed:
                self.closure_start_s = timestamp_s

            elif not closed and self.previous_closed:
                if self.closure_start_s is not None:
                    closure_duration = (
                        timestamp_s - self.closure_start_s
                    )

                    if (
                        self.min_blink_s
                        <= closure_duration
                        <= self.max_blink_s
                    ):
                        self.blink_count += 1
                        blink_event = True

                self.closure_start_s = None

            self.previous_closed = closed

        else:
            # Una pérdida de rostro interrumpe el evento.
            # No debe generar un parpadeo falso.
            self.closure_start_s = None
            self.previous_closed = False
            closed = False

        self.samples.append(
            EyeSample(
                timestamp_s=timestamp_s,
                valid=valid_ear,
                closed=closed,
            )
        )

        self.last_timestamp_s = timestamp_s

        self._discard_old_samples(timestamp_s)

        (
            perclos,
            valid_ratio,
            window_duration_s,
            valid_duration_s,
        ) = self._calculate_window(timestamp_s)

        if (
            valid_ear
            and closed
            and self.closure_start_s is not None
        ):
            closure_duration_s = (
                timestamp_s - self.closure_start_s
            )
        else:
            closure_duration_s = 0.0

        return EyeTemporalMetrics(
            perclos=perclos,
            valid_ratio=valid_ratio,
            closure_duration_s=closure_duration_s,
            blink_count=self.blink_count,
            blink_event=blink_event,
            window_duration_s=window_duration_s,
            valid_duration_s=valid_duration_s,
        )

    def _discard_old_samples(
        self,
        current_timestamp_s: float,
    ):
        """
        Descarta muestras antiguas.

        Se conserva una muestra anterior al inicio de la ventana
        porque su estado puede extenderse hasta el límite.
        """

        cutoff_s = current_timestamp_s - self.window_s

        while (
            len(self.samples) >= 2
            and self.samples[1].timestamp_s <= cutoff_s
        ):
            self.samples.popleft()

    def _calculate_window(
        self,
        current_timestamp_s: float,
    ):
        """Calcula PERCLOS y valid_ratio ponderados por tiempo."""

        if len(self.samples) < 2:
            sample = self.samples[-1]

            valid_ratio = 1.0 if sample.valid else 0.0

            return (
                float("nan"),
                valid_ratio,
                0.0,
                0.0,
            )

        cutoff_s = current_timestamp_s - self.window_s

        window_start_s = max(
            cutoff_s,
            self.samples[0].timestamp_s,
        )

        window_duration_s = (
            current_timestamp_s - window_start_s
        )

        valid_duration_s = 0.0
        closed_duration_s = 0.0

        samples = list(self.samples)

        for index, sample in enumerate(samples):
            segment_start_s = max(
                sample.timestamp_s,
                window_start_s,
            )

            if index + 1 < len(samples):
                segment_end_s = min(
                    samples[index + 1].timestamp_s,
                    current_timestamp_s,
                )
            else:
                segment_end_s = current_timestamp_s

            segment_duration_s = (
                segment_end_s - segment_start_s
            )

            if segment_duration_s <= 0:
                continue

            if sample.valid:
                valid_duration_s += segment_duration_s

                if sample.closed:
                    closed_duration_s += segment_duration_s

        if valid_duration_s > 0:
            perclos = (
                closed_duration_s / valid_duration_s
            )
        else:
            perclos = float("nan")

        if window_duration_s > 0:
            valid_ratio = (
                valid_duration_s / window_duration_s
            )
        else:
            valid_ratio = 0.0

        return (
            perclos,
            valid_ratio,
            window_duration_s,
            valid_duration_s,
        )
@dataclass(frozen=True)
class MouthTemporalMetrics:
    """Resultados temporales asociados con la apertura bucal."""

    mouth_open: bool
    opening_duration_s: float
    yawn_count: int
    yawn_event: bool


class MouthTemporalAnalyzer:
    """
    Detecta bostezos a partir de MAR y su duración.

    Una apertura solo se registra como bostezo cuando MAR
    permanece por encima del umbral durante el tiempo mínimo.
    """

    def __init__(
        self,
        mar_threshold: float = 0.75,
        min_yawn_s: float = 1.50,
    ):
        if mar_threshold <= 0:
            raise ValueError(
                "mar_threshold debe ser mayor que cero."
            )

        if min_yawn_s <= 0:
            raise ValueError(
                "min_yawn_s debe ser mayor que cero."
            )

        self.mar_threshold = float(mar_threshold)
        self.min_yawn_s = float(min_yawn_s)

        self.last_timestamp_s = None
        self.opening_start_s = None
        self.yawn_reported = False
        self.yawn_count = 0

    def reset(self):
        """Reinicia el estado temporal de la boca."""

        self.last_timestamp_s = None
        self.opening_start_s = None
        self.yawn_reported = False
        self.yawn_count = 0

    def update(
        self,
        timestamp_s: float,
        mar: float,
        valid: bool,
    ) -> MouthTemporalMetrics:
        """
        Añade una medición MAR y devuelve el estado temporal.

        Una medición inválida interrumpe la apertura actual y
        nunca se interpreta como bostezo.
        """

        timestamp_s = float(timestamp_s)

        if not math.isfinite(timestamp_s):
            raise ValueError(
                "timestamp_s debe ser un número finito."
            )

        if (
            self.last_timestamp_s is not None
            and timestamp_s <= self.last_timestamp_s
        ):
            raise ValueError(
                "Los timestamps deben ser estrictamente crecientes."
            )

        valid_mar = (
            bool(valid)
            and math.isfinite(float(mar))
        )

        mouth_open = (
            valid_mar
            and float(mar) >= self.mar_threshold
        )

        yawn_event = False
        opening_duration_s = 0.0

        if mouth_open:
            if self.opening_start_s is None:
                self.opening_start_s = timestamp_s
                self.yawn_reported = False

            opening_duration_s = (
                timestamp_s - self.opening_start_s
            )

            if (
                opening_duration_s >= self.min_yawn_s
                and not self.yawn_reported
            ):
                self.yawn_count += 1
                self.yawn_reported = True
                yawn_event = True

        else:
            self.opening_start_s = None
            self.yawn_reported = False

        if not valid_mar:
            mouth_open = False
            opening_duration_s = 0.0

        self.last_timestamp_s = timestamp_s

        return MouthTemporalMetrics(
            mouth_open=mouth_open,
            opening_duration_s=opening_duration_s,
            yawn_count=self.yawn_count,
            yawn_event=yawn_event,
        )

@dataclass(frozen=True)
class GazeTemporalMetrics:
    """Resultados temporales de la dirección de mirada."""

    horizontal: float
    vertical: float
    direction: str
    off_center: bool
    away_duration_s: float
    distraction_active: bool
    gaze_away_count: int
    gaze_away_event: bool


class GazeTemporalAnalyzer:
    """
    Suaviza la mirada y detecta desviaciones sostenidas.

    Los umbrales son provisionales y posteriormente serán
    reemplazados por la calibración personal.
    """

    def __init__(
        self,
        horizontal_low: float = 0.36,
        horizontal_high: float = 0.62,
        vertical_low: float = 0.40,
        vertical_high: float = 0.44,
        min_away_s: float = 1.50,
        smoothing_tau_s: float = 0.15,
    ):
        if horizontal_low >= horizontal_high:
            raise ValueError(
                "Los umbrales horizontales no son válidos."
            )

        if vertical_low >= vertical_high:
            raise ValueError(
                "Los umbrales verticales no son válidos."
            )

        if min_away_s <= 0:
            raise ValueError(
                "min_away_s debe ser mayor que cero."
            )

        if smoothing_tau_s <= 0:
            raise ValueError(
                "smoothing_tau_s debe ser mayor que cero."
            )

        self.horizontal_low = float(horizontal_low)
        self.horizontal_high = float(horizontal_high)
        self.vertical_low = float(vertical_low)
        self.vertical_high = float(vertical_high)
        self.min_away_s = float(min_away_s)
        self.smoothing_tau_s = float(smoothing_tau_s)

        self.last_timestamp_s = None
        self.filtered_horizontal = None
        self.filtered_vertical = None
        self.away_start_s = None
        self.away_reported = False
        self.gaze_away_count = 0

    def reset(self):
        """Reinicia el estado temporal de mirada."""

        self.last_timestamp_s = None
        self.filtered_horizontal = None
        self.filtered_vertical = None
        self.away_start_s = None
        self.away_reported = False
        self.gaze_away_count = 0

    def _classify_direction(
        self,
        horizontal: float,
        vertical: float,
    ) -> str:
        """Clasifica la dirección usando los umbrales actuales."""

        horizontal_direction = ""
        vertical_direction = ""

        if horizontal < self.horizontal_low:
            horizontal_direction = "DERECHA"
        elif horizontal > self.horizontal_high:
            horizontal_direction = "IZQUIERDA"

        if vertical < self.vertical_low:
            vertical_direction = "ARRIBA"
        elif vertical > self.vertical_high:
            vertical_direction = "ABAJO"

        if horizontal_direction and vertical_direction:
            return (
                f"{horizontal_direction}_"
                f"{vertical_direction}"
            )

        if horizontal_direction:
            return horizontal_direction

        if vertical_direction:
            return vertical_direction

        return "CENTRO"

    def update(
        self,
        timestamp_s: float,
        horizontal: float,
        vertical: float,
        valid: bool,
    ) -> GazeTemporalMetrics:
        """Añade una medición geométrica de mirada."""

        timestamp_s = float(timestamp_s)

        if not math.isfinite(timestamp_s):
            raise ValueError(
                "timestamp_s debe ser un número finito."
            )

        if (
            self.last_timestamp_s is not None
            and timestamp_s <= self.last_timestamp_s
        ):
            raise ValueError(
                "Los timestamps deben ser estrictamente crecientes."
            )

        valid_gaze = (
            bool(valid)
            and math.isfinite(float(horizontal))
            and math.isfinite(float(vertical))
        )

        gaze_away_event = False

        if not valid_gaze:
            self.filtered_horizontal = None
            self.filtered_vertical = None
            self.away_start_s = None
            self.away_reported = False
            self.last_timestamp_s = timestamp_s

            return GazeTemporalMetrics(
                horizontal=float("nan"),
                vertical=float("nan"),
                direction="NO_VALIDA",
                off_center=False,
                away_duration_s=0.0,
                distraction_active=False,
                gaze_away_count=self.gaze_away_count,
                gaze_away_event=False,
            )

        horizontal = float(horizontal)
        vertical = float(vertical)

        if (
            self.filtered_horizontal is None
            or self.filtered_vertical is None
            or self.last_timestamp_s is None
        ):
            self.filtered_horizontal = horizontal
            self.filtered_vertical = vertical
        else:
            delta_s = (
                timestamp_s - self.last_timestamp_s
            )

            alpha = 1.0 - math.exp(
                -delta_s / self.smoothing_tau_s
            )

            self.filtered_horizontal += alpha * (
                horizontal - self.filtered_horizontal
            )

            self.filtered_vertical += alpha * (
                vertical - self.filtered_vertical
            )

        direction = self._classify_direction(
            self.filtered_horizontal,
            self.filtered_vertical,
        )

        off_center = direction != "CENTRO"

        if off_center:
            if self.away_start_s is None:
                self.away_start_s = timestamp_s
                self.away_reported = False

            away_duration_s = (
                timestamp_s - self.away_start_s
            )

            distraction_active = (
                away_duration_s >= self.min_away_s
            )

            if (
                distraction_active
                and not self.away_reported
            ):
                self.gaze_away_count += 1
                self.away_reported = True
                gaze_away_event = True

        else:
            self.away_start_s = None
            self.away_reported = False
            away_duration_s = 0.0
            distraction_active = False

        self.last_timestamp_s = timestamp_s

        return GazeTemporalMetrics(
            horizontal=self.filtered_horizontal,
            vertical=self.filtered_vertical,
            direction=direction,
            off_center=off_center,
            away_duration_s=away_duration_s,
            distraction_active=distraction_active,
            gaze_away_count=self.gaze_away_count,
            gaze_away_event=gaze_away_event,
        )

@dataclass(frozen=True)
class HeadPoseTemporalMetrics:
    """Resultados temporales de la orientación de cabeza."""

    pitch_relative_deg: float
    yaw_relative_deg: float
    roll_relative_deg: float
    direction: str
    off_center: bool
    away_duration_s: float
    pose_away_count: int
    pose_away_event: bool
    nod_active: bool
    nod_duration_s: float
    nod_count: int
    nod_event: bool


class HeadPoseTemporalAnalyzer:
    """
    Filtra la pose, detecta desviaciones sostenidas y cabeceos.

    Los ángulos se expresan respecto a una pose frontal base.
    """

    def __init__(
        self,
        baseline_pitch_deg: float = 6.55,
        baseline_yaw_deg: float = 1.25,
        baseline_roll_deg: float = 0.75,
        yaw_threshold_deg: float = 20.0,
        pitch_threshold_deg: float = 15.0,
        roll_threshold_deg: float = 20.0,
        min_away_s: float = 1.50,
        nod_down_threshold_deg: float = 15.0,
        nod_return_threshold_deg: float = 8.0,
        min_nod_s: float = 0.30,
        max_nod_s: float = 2.50,
        smoothing_tau_s: float = 0.15,
    ):
        if yaw_threshold_deg <= 0:
            raise ValueError(
                "yaw_threshold_deg debe ser positivo."
            )

        if pitch_threshold_deg <= 0:
            raise ValueError(
                "pitch_threshold_deg debe ser positivo."
            )

        if roll_threshold_deg <= 0:
            raise ValueError(
                "roll_threshold_deg debe ser positivo."
            )

        if min_away_s <= 0:
            raise ValueError(
                "min_away_s debe ser positivo."
            )

        if not (
            0 <= nod_return_threshold_deg
            < nod_down_threshold_deg
        ):
            raise ValueError(
                "Los umbrales de cabeceo no son válidos."
            )

        if not (0 < min_nod_s < max_nod_s):
            raise ValueError(
                "Las duraciones de cabeceo no son válidas."
            )

        if smoothing_tau_s <= 0:
            raise ValueError(
                "smoothing_tau_s debe ser positivo."
            )

        self.baseline_pitch_deg = float(
            baseline_pitch_deg
        )

        self.baseline_yaw_deg = float(
            baseline_yaw_deg
        )

        self.baseline_roll_deg = float(
            baseline_roll_deg
        )

        self.yaw_threshold_deg = float(
            yaw_threshold_deg
        )

        self.pitch_threshold_deg = float(
            pitch_threshold_deg
        )

        self.roll_threshold_deg = float(
            roll_threshold_deg
        )

        self.min_away_s = float(min_away_s)

        self.nod_down_threshold_deg = float(
            nod_down_threshold_deg
        )

        self.nod_return_threshold_deg = float(
            nod_return_threshold_deg
        )

        self.min_nod_s = float(min_nod_s)
        self.max_nod_s = float(max_nod_s)
        self.smoothing_tau_s = float(smoothing_tau_s)

        self.last_timestamp_s = None

        self.filtered_pitch = None
        self.filtered_yaw = None
        self.filtered_roll = None

        self.away_start_s = None
        self.away_reported = False
        self.pose_away_count = 0

        self.nod_start_s = None
        self.nod_count = 0

    def reset(self):
        """Reinicia el estado temporal de la pose."""

        self.last_timestamp_s = None

        self.filtered_pitch = None
        self.filtered_yaw = None
        self.filtered_roll = None

        self.away_start_s = None
        self.away_reported = False
        self.pose_away_count = 0

        self.nod_start_s = None
        self.nod_count = 0

    def _classify_direction(
        self,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> str:
        """Clasifica la pose relativa de la cabeza."""

        directions = []

        if yaw > self.yaw_threshold_deg:
            directions.append("IZQUIERDA")
        elif yaw < -self.yaw_threshold_deg:
            directions.append("DERECHA")

        if pitch > self.pitch_threshold_deg:
            directions.append("ABAJO")
        elif pitch < -self.pitch_threshold_deg:
            directions.append("ARRIBA")

        if roll > self.roll_threshold_deg:
            directions.append("INCLINADA_DERECHA")
        elif roll < -self.roll_threshold_deg:
            directions.append("INCLINADA_IZQUIERDA")

        if not directions:
            return "FRONTAL"

        return "_".join(directions)

    def update(
        self,
        timestamp_s: float,
        pitch_deg: float,
        yaw_deg: float,
        roll_deg: float,
        valid: bool,
    ) -> HeadPoseTemporalMetrics:
        """Añade una medición de pose facial."""

        timestamp_s = float(timestamp_s)

        if not math.isfinite(timestamp_s):
            raise ValueError(
                "timestamp_s debe ser finito."
            )

        if (
            self.last_timestamp_s is not None
            and timestamp_s <= self.last_timestamp_s
        ):
            raise ValueError(
                "Los timestamps deben ser crecientes."
            )

        valid_pose = (
            bool(valid)
            and math.isfinite(float(pitch_deg))
            and math.isfinite(float(yaw_deg))
            and math.isfinite(float(roll_deg))
        )

        if not valid_pose:
            self.filtered_pitch = None
            self.filtered_yaw = None
            self.filtered_roll = None

            self.away_start_s = None
            self.away_reported = False
            self.nod_start_s = None

            self.last_timestamp_s = timestamp_s

            return HeadPoseTemporalMetrics(
                pitch_relative_deg=float("nan"),
                yaw_relative_deg=float("nan"),
                roll_relative_deg=float("nan"),
                direction="NO_VALIDA",
                off_center=False,
                away_duration_s=0.0,
                pose_away_count=self.pose_away_count,
                pose_away_event=False,
                nod_active=False,
                nod_duration_s=0.0,
                nod_count=self.nod_count,
                nod_event=False,
            )

        pitch_relative = (
            float(pitch_deg)
            - self.baseline_pitch_deg
        )

        yaw_relative = (
            float(yaw_deg)
            - self.baseline_yaw_deg
        )

        roll_relative = (
            float(roll_deg)
            - self.baseline_roll_deg
        )

        if (
            self.filtered_pitch is None
            or self.filtered_yaw is None
            or self.filtered_roll is None
            or self.last_timestamp_s is None
        ):
            self.filtered_pitch = pitch_relative
            self.filtered_yaw = yaw_relative
            self.filtered_roll = roll_relative
        else:
            delta_s = (
                timestamp_s - self.last_timestamp_s
            )

            alpha = 1.0 - math.exp(
                -delta_s / self.smoothing_tau_s
            )

            self.filtered_pitch += alpha * (
                pitch_relative - self.filtered_pitch
            )

            self.filtered_yaw += alpha * (
                yaw_relative - self.filtered_yaw
            )

            self.filtered_roll += alpha * (
                roll_relative - self.filtered_roll
            )

        direction = self._classify_direction(
            self.filtered_pitch,
            self.filtered_yaw,
            self.filtered_roll,
        )

        off_center = direction != "FRONTAL"
        pose_away_event = False

        if off_center:
            if self.away_start_s is None:
                self.away_start_s = timestamp_s
                self.away_reported = False

            away_duration_s = (
                timestamp_s - self.away_start_s
            )

            if (
                away_duration_s >= self.min_away_s
                and not self.away_reported
            ):
                self.pose_away_count += 1
                self.away_reported = True
                pose_away_event = True
        else:
            self.away_start_s = None
            self.away_reported = False
            away_duration_s = 0.0

        nod_event = False
        nod_active = False
        nod_duration_s = 0.0

        if self.nod_start_s is None:
            if (
                self.filtered_pitch
                >= self.nod_down_threshold_deg
            ):
                self.nod_start_s = timestamp_s
                nod_active = True
        else:
            nod_duration_s = (
                timestamp_s - self.nod_start_s
            )

            nod_active = True

            if (
                self.filtered_pitch
                <= self.nod_return_threshold_deg
            ):
                if (
                    self.min_nod_s
                    <= nod_duration_s
                    <= self.max_nod_s
                ):
                    self.nod_count += 1
                    nod_event = True

                self.nod_start_s = None
                nod_active = False
                nod_duration_s = 0.0

        self.last_timestamp_s = timestamp_s

        return HeadPoseTemporalMetrics(
            pitch_relative_deg=self.filtered_pitch,
            yaw_relative_deg=self.filtered_yaw,
            roll_relative_deg=self.filtered_roll,
            direction=direction,
            off_center=off_center,
            away_duration_s=away_duration_s,
            pose_away_count=self.pose_away_count,
            pose_away_event=pose_away_event,
            nod_active=nod_active,
            nod_duration_s=nod_duration_s,
            nod_count=self.nod_count,
            nod_event=nod_event,
        )