"""Máquina temporal de estados para el nivel operativo de alerta.

Los estados de este módulo no representan un diagnóstico médico. Describen
únicamente el nivel operativo de alerta producido por el prototipo.
"""

from enum import Enum, auto
import math


class State(Enum):
    NORMAL = auto()
    SUSPECTED = auto()
    FATIGUE = auto()
    CRITICAL = auto()
    RECOVERY = auto()


VISIBLE_LABELS = {
    State.NORMAL: "NORMAL",
    State.SUSPECTED: "FATIGA LEVE",
    State.FATIGUE: "FATIGA MODERADA",
    State.CRITICAL: "FATIGA GRAVE",
    State.RECOVERY: "RECUPERACION",
}


class FatigueStateMachine:
    """Aplica persistencia, histéresis y enclavamiento de CRITICAL."""

    def __init__(self):
        self.state = State.NORMAL
        self.candidate_key: str | None = None
        self.candidate_since_ms: int | None = None
        self.last_timestamp_ms: int | None = None
        self.critical_acknowledged = False

    def reset(self) -> None:
        """Restablece completamente la máquina al estado NORMAL."""

        self.state = State.NORMAL
        self.candidate_key = None
        self.candidate_since_ms = None
        self.last_timestamp_ms = None
        self.critical_acknowledged = False

    def _validate(
        self,
        t_ms: int,
        score: float,
        closure_s: float,
    ) -> tuple[int, float, float]:
        timestamp = int(t_ms)
        score = float(score)
        closure = float(closure_s)

        if timestamp < 0:
            raise ValueError("t_ms no puede ser negativo")

        if (
            self.last_timestamp_ms is not None
            and timestamp < self.last_timestamp_ms
        ):
            raise ValueError("t_ms debe ser monotónico")

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score debe ser finito y estar entre 0 y 1")

        if not math.isfinite(closure) or closure < 0.0:
            raise ValueError("closure_s debe ser finito y no negativo")

        self.last_timestamp_ms = timestamp
        return timestamp, score, closure

    def _clear_candidate(self) -> None:
        self.candidate_key = None
        self.candidate_since_ms = None

    def _candidate_elapsed(
        self,
        key: str,
        dwell_ms: int,
        t_ms: int,
    ) -> bool:
        """Indica si una condición se mantuvo durante todo ``dwell_ms``."""

        if self.candidate_key != key:
            self.candidate_key = key
            self.candidate_since_ms = t_ms
            return dwell_ms == 0

        assert self.candidate_since_ms is not None
        return t_ms - self.candidate_since_ms >= dwell_ms

    def _enter(
        self,
        state: State,
        event: str,
    ) -> tuple[State, str | None]:
        """Cambia de estado y emite el evento una sola vez."""

        if self.state == state:
            self._clear_candidate()
            return self.state, None

        self.state = state
        self._clear_candidate()

        if state == State.CRITICAL:
            self.critical_acknowledged = False
        elif state == State.NORMAL:
            self.critical_acknowledged = False

        return self.state, event

    def _persist_transition(
        self,
        *,
        key: str,
        target: State,
        dwell_ms: int,
        t_ms: int,
        event: str,
    ) -> tuple[State, str | None]:
        if self._candidate_elapsed(key, dwell_ms, t_ms):
            return self._enter(target, event)

        return self.state, None

    @property
    def alert_acknowledged(self) -> bool:
        """Indica si la alerta CRITICAL actual fue reconocida."""

        return (
            self.state == State.CRITICAL
            and self.critical_acknowledged
        )

    @property
    def critical_recovery_elapsed_s(self) -> float:
        """Tiempo estable acumulado para abandonar CRITICAL."""

        if (
            not self.alert_acknowledged
            or self.candidate_key != "critical_to_recovery"
            or self.candidate_since_ms is None
            or self.last_timestamp_ms is None
        ):
            return 0.0

        elapsed_ms = (
            self.last_timestamp_ms - self.candidate_since_ms
        )
        return max(0.0, elapsed_ms / 1000.0)

    def invalidate(self, t_ms: int) -> tuple[State, None]:
        """Avanza el tiempo sin interpretar un frame inválido como normal.

        El estado confirmado se conserva, pero cualquier transición todavía
        candidata se cancela. Así un intervalo sin rostro no puede completar
        una entrada ni una salida de estado.
        """

        timestamp = int(t_ms)

        if timestamp < 0:
            raise ValueError("t_ms no puede ser negativo")

        if (
            self.last_timestamp_ms is not None
            and timestamp < self.last_timestamp_ms
        ):
            raise ValueError("t_ms debe ser monotónico")

        self.last_timestamp_ms = timestamp
        self._clear_candidate()
        return self.state, None

    def update(
        self,
        t_ms: int,
        score: float,
        closure_s: float,
        acknowledged: bool = False,
    ) -> tuple[State, str | None]:
        """Actualiza la máquina y devuelve ``(estado, evento_nuevo)``.

        ``acknowledged`` representa la confirmación explícita del usuario
        después de entrar en CRITICAL. No elimina el estado inmediatamente:
        además se requieren 30 segundos con score menor que 0.40.
        """

        t_ms, score, closure_s = self._validate(
            t_ms,
            score,
            closure_s,
        )

        # Una oclusión continua peligrosa tiene prioridad absoluta.
        if closure_s >= 2.0:
            if self.state != State.CRITICAL:
                return self._enter(
                    State.CRITICAL,
                    "prolonged_eye_closure",
                )

            # Un nuevo cierre peligroso requiere un reconocimiento nuevo.
            self.critical_acknowledged = False
            self._clear_candidate()
            return self.state, None

        if self.state == State.CRITICAL:
            acknowledgement_event = None

            if acknowledged and not self.critical_acknowledged:
                acknowledgement_event = "critical_acknowledged"

            if acknowledged:
                self.critical_acknowledged = True

            if not self.critical_acknowledged:
                self._clear_candidate()
                return self.state, None

            if score < 0.40:
                transition = self._persist_transition(
                    key="critical_to_recovery",
                    target=State.RECOVERY,
                    dwell_ms=30_000,
                    t_ms=t_ms,
                    event="state_recovery",
                )

                if transition[1] is not None:
                    return transition

                return self.state, acknowledgement_event

            self._clear_candidate()
            return self.state, acknowledgement_event

        # Escalada. Un score muy alto puede alcanzar directamente un estado
        # superior, pero debe persistir el tiempo requerido.
        if score >= 0.75:
            return self._persist_transition(
                key="to_critical",
                target=State.CRITICAL,
                dwell_ms=2_000,
                t_ms=t_ms,
                event="state_critical",
            )

        if score >= 0.50 and self.state in {
            State.NORMAL,
            State.SUSPECTED,
            State.RECOVERY,
        }:
            return self._persist_transition(
                key="to_fatigue",
                target=State.FATIGUE,
                dwell_ms=5_000,
                t_ms=t_ms,
                event="state_fatigue",
            )

        if score >= 0.30 and self.state in {
            State.NORMAL,
            State.RECOVERY,
        }:
            return self._persist_transition(
                key="to_suspected",
                target=State.SUSPECTED,
                dwell_ms=5_000,
                t_ms=t_ms,
                event="state_suspected",
            )

        # Descenso con umbrales menores y tiempos más largos que la entrada.
        if self.state == State.SUSPECTED and score < 0.20:
            return self._persist_transition(
                key="suspected_to_normal",
                target=State.NORMAL,
                dwell_ms=10_000,
                t_ms=t_ms,
                event="state_normal",
            )

        if self.state == State.FATIGUE and score < 0.35:
            return self._persist_transition(
                key="fatigue_to_recovery",
                target=State.RECOVERY,
                dwell_ms=15_000,
                t_ms=t_ms,
                event="state_recovery",
            )

        if self.state == State.RECOVERY and score < 0.20:
            return self._persist_transition(
                key="recovery_to_normal",
                target=State.NORMAL,
                dwell_ms=10_000,
                t_ms=t_ms,
                event="state_normal",
            )

        self._clear_candidate()
        return self.state, None
