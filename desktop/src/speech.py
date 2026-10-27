"""Síntesis offline y no bloqueante para alertas de fatiga."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from queue import Empty, Full, PriorityQueue
from threading import Event, Lock, Thread
import hashlib
import itertools
import shutil
import subprocess
import tempfile
import time
import wave


class SpeechPriority(IntEnum):
    """Prioridad de reproducción de menor a mayor."""

    INFORMATION = 10
    SUSPECTED = 20
    FATIGUE = 40
    REPEAT = 80
    CRITICAL = 100


@dataclass(frozen=True)
class AlertDefinition:
    """Mensaje asociado con un estado operacional."""

    text: str
    priority: SpeechPriority
    cooldown_s: float


ALERTS_BY_STATE = {
    "SUSPECTED": AlertDefinition(
        text=(
            "Atención. Se detectaron signos leves de "
            "fatiga. Mantenga la atención."
        ),
        priority=SpeechPriority.SUSPECTED,
        cooldown_s=60.0,
    ),
    "FATIGUE": AlertDefinition(
        text=(
            "Advertencia. Se detectaron signos de "
            "fatiga. Considere detenerse y descansar "
            "en un lugar seguro."
        ),
        priority=SpeechPriority.FATIGUE,
        cooldown_s=45.0,
    ),
    "CRITICAL": AlertDefinition(
        text=(
            "Alerta crítica. Deténgase en un lugar "
            "seguro lo antes posible."
        ),
        priority=SpeechPriority.CRITICAL,
        cooldown_s=20.0,
    ),
    "RECOVERY": AlertDefinition(
        text=(
            "Alerta reconocida. Mantenga la atención "
            "y descanse si es necesario."
        ),
        priority=SpeechPriority.INFORMATION,
        cooldown_s=60.0,
    ),
}


def normalize_state_name(state) -> str:
    """Convierte enum o texto de estado a una clave uniforme."""

    value = getattr(state, "value", state)
    return str(value).strip().upper()


def alert_for_state(state) -> AlertDefinition | None:
    """Obtiene el mensaje de un estado; NORMAL no habla."""

    return ALERTS_BY_STATE.get(
        normalize_state_name(state)
    )


@dataclass(order=True)
class _SpeechRequest:
    sort_priority: int
    sequence: int
    text: str = field(compare=False)
    priority: SpeechPriority = field(compare=False)
    cache_key: str = field(compare=False)


class OfflineAlertSpeaker:
    """
    Emisor asíncrono con Piper principal y eSpeak como respaldo.

    Piper se carga una sola vez dentro de un hilo. ``speak`` y
    ``announce_state`` únicamente insertan solicitudes en una cola y
    regresan inmediatamente.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        volume: float = 1.20,
        length_scale: float = 1.0,
        queue_size: int = 24,
        fallback_espeak: bool = True,
        espeak_voice: str = "es",
        espeak_speed: int = 145,
        espeak_pitch: int = 48,
        espeak_amplitude: int = 170,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(
            f"{self.model_path}.json"
        )
        self.volume = float(volume)
        self.length_scale = float(length_scale)
        self.fallback_espeak = bool(fallback_espeak)
        self.espeak_voice = str(espeak_voice)
        self.espeak_speed = int(espeak_speed)
        self.espeak_pitch = int(espeak_pitch)
        self.espeak_amplitude = int(espeak_amplitude)

        if self.volume <= 0.0:
            raise ValueError("volume debe ser positivo")

        if self.length_scale <= 0.0:
            raise ValueError(
                "length_scale debe ser positivo"
            )

        self._queue: PriorityQueue[_SpeechRequest] = (
            PriorityQueue(maxsize=int(queue_size))
        )
        self._sequence = itertools.count()
        self._stop_event = Event()
        self._ready_event = Event()
        self._worker: Thread | None = None
        self._voice = None
        self._syn_config = None
        self._current_process: subprocess.Popen | None = None
        self._process_lock = Lock()
        self._state_lock = Lock()
        self._cooldown_lock = Lock()
        self._last_by_key: dict[str, float] = {}
        self._last_request: tuple[str, SpeechPriority] | None = None
        self._temporary_directory = None

        self.backend: str | None = None
        self.last_error: str | None = None
        self.completed_count = 0
        self.failed_count = 0

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return (
                self._worker is not None
                and self._worker.is_alive()
            )

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    def start(self, timeout_s: float = 20.0) -> None:
        """Carga el backend y comienza el hilo de voz."""

        with self._state_lock:
            if (
                self._worker is not None
                and self._worker.is_alive()
            ):
                return

            self._stop_event.clear()
            self._ready_event.clear()
            self.last_error = None
            self.backend = None
            self._temporary_directory = (
                tempfile.TemporaryDirectory(
                    prefix="fatigue_speech_"
                )
            )
            self._worker = Thread(
                target=self._worker_loop,
                name="offline-alert-speaker",
                daemon=True,
            )
            self._worker.start()

        if not self._ready_event.wait(timeout_s):
            self.stop()
            raise TimeoutError(
                "Piper no terminó de cargar a tiempo."
            )

        if self.backend is None:
            error = self.last_error or (
                "No se pudo iniciar ningún sintetizador."
            )
            self.stop()
            raise RuntimeError(error)

    def stop(self) -> None:
        """Cancela la reproducción y libera todos los recursos."""

        with self._state_lock:
            worker = self._worker
            self._worker = None
            self._stop_event.set()

        self.cancel(clear_pending=True)

        if worker is not None:
            worker.join(timeout=5.0)

        self._voice = None
        self._syn_config = None
        self.backend = None

        temporary_directory = self._temporary_directory
        self._temporary_directory = None

        if temporary_directory is not None:
            temporary_directory.cleanup()

    def speak(
        self,
        text: str,
        *,
        priority: SpeechPriority = (
            SpeechPriority.INFORMATION
        ),
        cooldown_s: float = 0.0,
        key: str | None = None,
        force: bool = False,
    ) -> bool:
        """Programa un mensaje sin bloquear al llamador."""

        clean_text = " ".join(str(text).split())

        if not clean_text:
            return False

        priority = SpeechPriority(priority)
        cooldown_s = max(0.0, float(cooldown_s))
        request_key = key or clean_text.lower()
        now = time.monotonic()

        if not force and cooldown_s > 0.0:
            with self._cooldown_lock:
                previous = self._last_by_key.get(
                    request_key
                )

                if (
                    previous is not None
                    and now - previous < cooldown_s
                ):
                    return False

                self._last_by_key[request_key] = now

        cache_key = hashlib.sha256(
            clean_text.encode("utf-8")
        ).hexdigest()
        request = _SpeechRequest(
            sort_priority=-int(priority),
            sequence=next(self._sequence),
            text=clean_text,
            priority=priority,
            cache_key=cache_key,
        )

        if priority == SpeechPriority.CRITICAL:
            self.cancel(clear_pending=False)
            self._discard_lower_priority(priority)

        try:
            self._queue.put_nowait(request)
        except Full:
            self.last_error = (
                "La cola de voz está llena."
            )
            return False

        self._last_request = (clean_text, priority)
        return True

    def announce_state(
        self,
        state,
        *,
        force: bool = False,
    ) -> bool:
        """Programa la alerta definida para un estado."""

        state_name = normalize_state_name(state)
        definition = alert_for_state(state_name)

        if definition is None:
            return False

        return self.speak(
            definition.text,
            priority=definition.priority,
            cooldown_s=definition.cooldown_s,
            key=f"state:{state_name}",
            force=force,
        )

    def repeat_last(self) -> bool:
        """Repite el último mensaje solicitado."""

        if self._last_request is None:
            return False

        text, previous_priority = self._last_request
        priority = max(
            previous_priority,
            SpeechPriority.REPEAT,
        )

        return self.speak(
            text,
            priority=SpeechPriority(priority),
            force=True,
        )

    def cancel(self, *, clear_pending: bool = True) -> None:
        """Interrumpe el audio actual y, opcionalmente, la cola."""

        with self._process_lock:
            process = self._current_process

            if (
                process is not None
                and process.poll() is None
            ):
                process.terminate()

        if clear_pending:
            self._clear_queue()

    def _load_backend(self) -> None:
        piper_error = None

        try:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Modelo Piper no encontrado: "
                    f"{self.model_path}"
                )

            if not self.config_path.is_file():
                raise FileNotFoundError(
                    f"Configuración Piper no encontrada: "
                    f"{self.config_path}"
                )

            from piper import PiperVoice, SynthesisConfig

            self._voice = PiperVoice.load(
                str(self.model_path)
            )
            self._syn_config = SynthesisConfig(
                volume=self.volume,
                length_scale=self.length_scale,
            )
            self.backend = "piper"
            return
        except Exception as error:
            piper_error = error
            self._voice = None
            self._syn_config = None

        if (
            self.fallback_espeak
            and shutil.which("espeak-ng")
        ):
            self.backend = "espeak-ng"
            self.last_error = (
                "Piper no disponible; usando eSpeak NG: "
                f"{piper_error}"
            )
            return

        self.backend = None
        self.last_error = str(piper_error)

    def _worker_loop(self) -> None:
        try:
            self._load_backend()
        finally:
            self._ready_event.set()

        if self.backend is None:
            return

        while not self._stop_event.is_set():
            try:
                request = self._queue.get(timeout=0.20)
            except Empty:
                continue

            try:
                self._play_request(request)
                self.completed_count += 1
            except Exception as error:
                self.failed_count += 1
                self.last_error = str(error)
            finally:
                self._queue.task_done()

    def _play_request(
        self,
        request: _SpeechRequest,
    ) -> None:
        if self.backend == "piper":
            try:
                self._play_with_piper(request)
                return
            except Exception as error:
                self.last_error = (
                    "Fallo Piper; intentando eSpeak NG: "
                    f"{error}"
                )

                if not (
                    self.fallback_espeak
                    and shutil.which("espeak-ng")
                ):
                    raise

        self._play_with_espeak(request.text)

    def _play_with_piper(
        self,
        request: _SpeechRequest,
    ) -> None:
        if self._voice is None:
            raise RuntimeError("Piper no está cargado.")

        temporary_directory = self._temporary_directory

        if temporary_directory is None:
            raise RuntimeError(
                "No existe directorio temporal de audio."
            )

        wav_path = Path(temporary_directory.name) / (
            f"{request.cache_key}.wav"
        )

        if not wav_path.is_file():
            with wave.open(str(wav_path), "wb") as wav_file:
                self._voice.synthesize_wav(
                    request.text,
                    wav_file,
                    syn_config=self._syn_config,
                )

        self._run_player(
            ["aplay", "-q", str(wav_path)]
        )

    def _play_with_espeak(self, text: str) -> None:
        self._run_player(
            [
                "espeak-ng",
                "-v",
                self.espeak_voice,
                "-s",
                str(self.espeak_speed),
                "-p",
                str(self.espeak_pitch),
                "-a",
                str(self.espeak_amplitude),
                text,
            ]
        )

    def _run_player(self, command: list[str]) -> None:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        with self._process_lock:
            self._current_process = process

        try:
            return_code = process.wait()
        finally:
            with self._process_lock:
                if self._current_process is process:
                    self._current_process = None

        if return_code not in (0, -15):
            raise RuntimeError(
                "El reproductor terminó con código "
                f"{return_code}."
            )

    def _discard_lower_priority(
        self,
        minimum_priority: SpeechPriority,
    ) -> None:
        retained = []

        while True:
            try:
                request = self._queue.get_nowait()
            except Empty:
                break

            self._queue.task_done()

            if request.priority >= minimum_priority:
                retained.append(request)

        for request in retained:
            try:
                self._queue.put_nowait(request)
            except Full:
                break

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return
            else:
                self._queue.task_done()

    def __enter__(self) -> "OfflineAlertSpeaker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.stop()
