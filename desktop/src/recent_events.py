"""Contadores reproducibles de eventos dentro de ventanas temporales."""

from collections import deque
from dataclasses import dataclass
import math


class RollingEventCounter:
    """Cuenta flancos de subida ocurridos dentro de una ventana móvil."""

    def __init__(self, window_s: float):
        window = float(window_s)

        if not math.isfinite(window) or window <= 0.0:
            raise ValueError("window_s debe ser finito y positivo")

        self.window_s = window
        self.event_timestamps_s: deque[float] = deque()
        self.last_timestamp_s: float | None = None
        self.previous_event = False

    def reset(self) -> None:
        """Elimina todos los eventos y reinicia el detector de flancos."""

        self.event_timestamps_s.clear()
        self.last_timestamp_s = None
        self.previous_event = False

    def _validate_timestamp(self, timestamp_s: float) -> float:
        timestamp = float(timestamp_s)

        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(
                "timestamp_s debe ser finito y no negativo"
            )

        if (
            self.last_timestamp_s is not None
            and timestamp < self.last_timestamp_s
        ):
            raise ValueError("timestamp_s debe ser monotónico")

        self.last_timestamp_s = timestamp
        return timestamp

    def _expire_old_events(self, timestamp_s: float) -> None:
        cutoff_s = timestamp_s - self.window_s

        while (
            self.event_timestamps_s
            and self.event_timestamps_s[0] < cutoff_s
        ):
            self.event_timestamps_s.popleft()

    def update(
        self,
        *,
        timestamp_s: float,
        event: bool,
        valid: bool = True,
    ) -> int:
        """Actualiza la ventana y devuelve la cantidad de eventos recientes.

        Un evento se registra solamente en su flanco de subida. Por ello una
        señal ``True`` sostenida durante varios frames cuenta una sola vez.
        Los frames inválidos nunca agregan eventos.
        """

        timestamp = self._validate_timestamp(timestamp_s)
        self._expire_old_events(timestamp)

        if not valid:
            self.previous_event = False
            return len(self.event_timestamps_s)

        current_event = bool(event)

        if current_event and not self.previous_event:
            self.event_timestamps_s.append(timestamp)

        self.previous_event = current_event
        return len(self.event_timestamps_s)


@dataclass(frozen=True)
class RecentEventMetrics:
    """Cantidades utilizadas por la fusión multimodal."""

    yawns_last_5_min: int
    nods_last_2_min: int


class RecentEventWindows:
    """Mantiene las ventanas de bostezos y cabeceos."""

    def __init__(
        self,
        *,
        yawn_window_s: float = 300.0,
        nod_window_s: float = 120.0,
    ):
        self.yawns = RollingEventCounter(yawn_window_s)
        self.nods = RollingEventCounter(nod_window_s)

    def reset(self) -> None:
        self.yawns.reset()
        self.nods.reset()

    def update(
        self,
        *,
        timestamp_s: float,
        yawn_event: bool,
        nod_event: bool,
        yawn_valid: bool = True,
        nod_valid: bool = True,
    ) -> RecentEventMetrics:
        yawn_count = self.yawns.update(
            timestamp_s=timestamp_s,
            event=yawn_event,
            valid=yawn_valid,
        )

        nod_count = self.nods.update(
            timestamp_s=timestamp_s,
            event=nod_event,
            valid=nod_valid,
        )

        return RecentEventMetrics(
            yawns_last_5_min=yawn_count,
            nods_last_2_min=nod_count,
        )
