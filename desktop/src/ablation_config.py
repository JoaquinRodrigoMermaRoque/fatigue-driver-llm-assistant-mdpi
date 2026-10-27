"""Carga y validación reproducible de configuraciones de ablación."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

try:
    from .fusion import normalized_score as _normalized_score
except ImportError:
    from fusion import normalized_score as _normalized_score


ALLOWED_CUES = {"closure", "yawn", "perclos", "gaze", "nod"}
ALLOWED_CALIBRATIONS = {"global", "personal"}
ALLOWED_GAZE_SOURCES = {"none", "geometric", "mobileone"}
RESERVED_KEYS = {"schema_version", "weights"}


@dataclass(frozen=True)
class AblationConfig:
    config_id: str
    active_cues: tuple[str, ...]
    calibration: str
    gaze_source: str
    optional: bool
    weights: dict[str, float]
    schema_version: int
    config_sha256: str


def _canonical_hash(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_document(path: str | Path) -> tuple[Path, dict]:
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"No se encontró la configuración: {config_path}"
        )

    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON de ablación inválido: {error}"
        ) from error

    if not isinstance(document, dict):
        raise ValueError("La raíz de ablations.json debe ser un objeto")

    return config_path, document


def available_config_ids(path: str | Path) -> tuple[str, ...]:
    """Devuelve los identificadores definidos, excluyendo metadatos."""

    _, document = _load_document(path)
    return tuple(
        key
        for key in document
        if key not in RESERVED_KEYS
    )


def load_ablation(
    path: str | Path,
    config_id: str,
) -> AblationConfig:
    """Carga una configuración y valida su estructura completa."""

    _, document = _load_document(path)
    config_id = str(config_id)

    schema_version = document.get("schema_version")

    if schema_version != 1:
        raise ValueError(
            f"schema_version no compatible: {schema_version!r}"
        )

    if config_id not in document or config_id in RESERVED_KEYS:
        raise ValueError(f"Configuración desconocida: {config_id}")

    raw_weights = document.get("weights")

    if not isinstance(raw_weights, dict):
        raise ValueError("weights debe ser un objeto")

    if set(raw_weights) != ALLOWED_CUES:
        raise ValueError(
            "weights debe contener exactamente: "
            + ", ".join(sorted(ALLOWED_CUES))
        )

    weights: dict[str, float] = {}

    for name, value in raw_weights.items():
        try:
            weight = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Peso inválido para {name}") from error

        if weight < 0.0:
            raise ValueError(f"El peso {name} no puede ser negativo")

        weights[name] = weight

    if sum(weights.values()) <= 0.0:
        raise ValueError("La suma de pesos debe ser positiva")

    raw_config = document[config_id]

    if not isinstance(raw_config, dict):
        raise ValueError(f"La configuración {config_id} debe ser un objeto")

    raw_active = raw_config.get("active_cues")

    if not isinstance(raw_active, list) or not raw_active:
        raise ValueError("active_cues debe ser una lista no vacía")

    active_cues = tuple(str(name) for name in raw_active)

    if len(active_cues) != len(set(active_cues)):
        raise ValueError("active_cues no puede contener duplicados")

    unknown = sorted(set(active_cues) - ALLOWED_CUES)

    if unknown:
        raise ValueError(f"Señales desconocidas: {unknown}")

    calibration = str(raw_config.get("calibration", ""))
    gaze_source = str(raw_config.get("gaze_source", ""))

    if calibration not in ALLOWED_CALIBRATIONS:
        raise ValueError(f"Calibración inválida: {calibration}")

    if gaze_source not in ALLOWED_GAZE_SOURCES:
        raise ValueError(f"Fuente de mirada inválida: {gaze_source}")

    if "gaze" in active_cues and gaze_source == "none":
        raise ValueError(
            "Una configuración con gaze activo requiere una fuente"
        )

    if "gaze" not in active_cues and gaze_source != "none":
        raise ValueError(
            "gaze_source debe ser none cuando gaze no está activo"
        )

    payload = {
        "schema_version": schema_version,
        "weights": weights,
        "configuration": raw_config,
        "config_id": config_id,
    }

    return AblationConfig(
        config_id=config_id,
        active_cues=active_cues,
        calibration=calibration,
        gaze_source=gaze_source,
        optional=bool(raw_config.get("optional", False)),
        weights=weights,
        schema_version=schema_version,
        config_sha256=_canonical_hash(payload),
    )


def normalized_score(
    evidence: Mapping[str, float],
    weights: Mapping[str, float],
    active_cues: Sequence[str],
) -> float:
    """Delega el cálculo al mismo código utilizado por H5 en vivo."""

    return _normalized_score(evidence, weights, active_cues)
