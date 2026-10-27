"""Reconocimiento offline y no bloqueante de comandos de voz."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import json
import re
import time
import unicodedata


class VoiceCommand(str, Enum):
    """Comandos aceptados por el prototipo."""

    ACK = "ACK"
    REPEAT = "REPEAT"
    CANCEL = "CANCEL"


@dataclass(frozen=True)
class VoiceCommandEvent:
    """Comando reconocido y listo para ser consumido por la aplicación."""

    command: VoiceCommand
    text: str
    timestamp_s: float


ACK_PHRASES = frozenset(
    {
        "entendido",
        "estoy bien",
        "de acuerdo",
        "confirmo",
    }
)

REPEAT_PHRASES = frozenset(
    {
        "repetir",
        "repite",
        "repite la alerta",
    }
)

CANCEL_PHRASES = frozenset(
    {
        "cancelar",
    }
)

GRAMMAR_PHRASES = tuple(
    sorted(
        ACK_PHRASES
        | REPEAT_PHRASES
        | CANCEL_PHRASES
    )
)


def normalize_voice_text(text: str) -> str:
    """Normaliza texto sin aceptar coincidencias parciales."""

    normalized = unicodedata.normalize(
        "NFKD",
        str(text).lower(),
    )
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    normalized = re.sub(
        r"[^a-z0-9ñ]+",
        " ",
        normalized,
    )
    return " ".join(normalized.split())


def classify_voice_command(
    text: str,
) -> VoiceCommand | None:
    """Convierte una frase completa en un comando seguro."""

    normalized = normalize_voice_text(text)

    if normalized in ACK_PHRASES:
        return VoiceCommand.ACK

    if normalized in REPEAT_PHRASES:
        return VoiceCommand.REPEAT

    if normalized in CANCEL_PHRASES:
        return VoiceCommand.CANCEL

    return None


def choose_input_device(
    devices,
    default_input: int | None = None,
) -> tuple[int, str]:
    """Elige una entrada real priorizando la fuente analógica."""

    priorities = (
        "alsa_input",
        "default source",
        "hda intel pch",
    )

    for priority in priorities:
        for index, device in enumerate(devices):
            name = str(device.get("name", ""))
            channels = int(
                device.get("max_input_channels", 0)
            )

            if (
                channels > 0
                and priority in name.lower()
            ):
                return index, name

    if (
        default_input is not None
        and 0 <= int(default_input) < len(devices)
    ):
        device = devices[int(default_input)]

        if int(device.get("max_input_channels", 0)) > 0:
            return int(default_input), str(device["name"])

    for index, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) > 0:
            return index, str(device["name"])

    raise RuntimeError(
        "No se encontró un dispositivo de entrada de audio."
    )


class OfflineVoiceRecognizer:
    """
    Reconocedor Vosk ejecutado en segundo plano.

    El callback de audio solo copia bloques a una cola. El análisis de
    Vosk se realiza en un hilo independiente, por lo que el bucle de
    cámara puede consultar ``poll()`` sin bloquearse.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: int | None = None,
        sample_rate: int = 16_000,
        blocksize: int = 8_000,
        audio_queue_size: int = 16,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.sample_rate = int(sample_rate)
        self.blocksize = int(blocksize)

        if self.sample_rate <= 0:
            raise ValueError(
                "sample_rate debe ser positivo"
            )

        if self.blocksize <= 0:
            raise ValueError(
                "blocksize debe ser positivo"
            )

        self._audio_queue: Queue[bytes] = Queue(
            maxsize=int(audio_queue_size)
        )
        self._event_queue: Queue[VoiceCommandEvent] = (
            Queue()
        )
        self._stop_event = Event()
        self._worker: Thread | None = None
        self._stream = None
        self._model = None
        self._recognizer = None
        self._state_lock = Lock()

        self.selected_device_index: int | None = None
        self.selected_device_name: str | None = None
        self.last_error: str | None = None

    @property
    def is_running(self) -> bool:
        """Indica si la captura se encuentra activa."""

        with self._state_lock:
            return self._stream is not None

    def start(self) -> None:
        """Carga el modelo y comienza la captura no bloqueante."""

        with self._state_lock:
            if self._stream is not None:
                return

            if not self.model_path.is_dir():
                raise FileNotFoundError(
                    "No se encontró el modelo Vosk: "
                    f"{self.model_path}"
                )

            import sounddevice as sd
            from vosk import (
                KaldiRecognizer,
                Model,
                SetLogLevel,
            )

            SetLogLevel(-1)

            devices = sd.query_devices()
            default_input = sd.default.device[0]

            if self.device is None:
                device_index, device_name = (
                    choose_input_device(
                        devices,
                        default_input=default_input,
                    )
                )
            else:
                device_index = int(self.device)
                device_info = devices[device_index]

                if (
                    int(
                        device_info[
                            "max_input_channels"
                        ]
                    )
                    <= 0
                ):
                    raise ValueError(
                        f"El dispositivo {device_index} "
                        "no tiene canales de entrada."
                    )

                device_name = str(device_info["name"])

            model = Model(str(self.model_path))
            self._model = model
            grammar = json.dumps(
                [*GRAMMAR_PHRASES, "[unk]"],
                ensure_ascii=False,
            )
            self._recognizer = KaldiRecognizer(
                model,
                self.sample_rate,
                grammar,
            )

            self._stop_event.clear()
            self.last_error = None
            self.selected_device_index = device_index
            self.selected_device_name = device_name

            self._worker = Thread(
                target=self._worker_loop,
                name="vosk-voice-worker",
                daemon=True,
            )
            self._worker.start()

            try:
                stream = sd.RawInputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.blocksize,
                    device=device_index,
                    dtype="int16",
                    channels=1,
                    callback=self._audio_callback,
                )
                stream.start()
                self._stream = stream
            except Exception:
                self._stop_event.set()
                self._worker.join(timeout=2.0)
                self._worker = None
                self._recognizer = None
                self._model = None
                raise

    def stop(self) -> None:
        """Detiene audio e hilo y libera los recursos."""

        with self._state_lock:
            stream = self._stream
            worker = self._worker
            self._stream = None
            self._worker = None
            self._stop_event.set()

        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

        if worker is not None:
            worker.join(timeout=2.0)

        self._recognizer = None
        self._model = None
        self._clear_queue(self._audio_queue)

    def poll(self) -> VoiceCommandEvent | None:
        """Devuelve un comando disponible sin esperar."""

        try:
            return self._event_queue.get_nowait()
        except Empty:
            return None

    def drain(self) -> list[VoiceCommandEvent]:
        """Extrae todos los comandos pendientes."""

        events = []

        while True:
            event = self.poll()

            if event is None:
                return events

            events.append(event)

    def _audio_callback(
        self,
        indata,
        frames,
        callback_time,
        status,
    ) -> None:
        del frames, callback_time

        if status:
            self.last_error = str(status)

        block = bytes(indata)

        try:
            self._audio_queue.put_nowait(block)
        except Full:
            try:
                self._audio_queue.get_nowait()
            except Empty:
                pass

            try:
                self._audio_queue.put_nowait(block)
            except Full:
                pass

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio = self._audio_queue.get(
                    timeout=0.20
                )
            except Empty:
                continue

            recognizer = self._recognizer

            if recognizer is None:
                continue

            try:
                complete = recognizer.AcceptWaveform(
                    audio
                )
            except Exception as error:
                self.last_error = str(error)
                continue

            if not complete:
                continue

            try:
                result = json.loads(
                    recognizer.Result()
                )
            except (TypeError, json.JSONDecodeError) as error:
                self.last_error = str(error)
                continue

            text = str(
                result.get("text", "")
            ).strip()
            command = classify_voice_command(text)

            if command is None:
                continue

            self._event_queue.put(
                VoiceCommandEvent(
                    command=command,
                    text=normalize_voice_text(text),
                    timestamp_s=time.monotonic(),
                )
            )

    @staticmethod
    def _clear_queue(target: Queue) -> None:
        while True:
            try:
                target.get_nowait()
            except Empty:
                return

    def __enter__(self) -> "OfflineVoiceRecognizer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.stop()
