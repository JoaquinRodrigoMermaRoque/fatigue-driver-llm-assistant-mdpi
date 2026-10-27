import json
import math

import pytest

from src.ablation_config import (
    available_config_ids,
    load_ablation,
    normalized_score,
)


def write_config(tmp_path):
    source = tmp_path / "ablations.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "weights": {
                    "closure": 0.30,
                    "perclos": 0.25,
                    "yawn": 0.15,
                    "gaze": 0.10,
                    "nod": 0.20,
                },
                "A": {
                    "active_cues": ["closure"],
                    "calibration": "global",
                    "gaze_source": "none",
                },
                "B": {
                    "active_cues": ["closure", "yawn"],
                    "calibration": "global",
                    "gaze_source": "none",
                },
                "C": {
                    "active_cues": ["closure", "yawn", "perclos"],
                    "calibration": "global",
                    "gaze_source": "none",
                },
                "D": {
                    "active_cues": [
                        "closure", "yawn", "perclos", "gaze"
                    ],
                    "calibration": "global",
                    "gaze_source": "geometric",
                },
                "E": {
                    "active_cues": [
                        "closure", "yawn", "perclos", "gaze", "nod"
                    ],
                    "calibration": "global",
                    "gaze_source": "geometric",
                },
                "F": {
                    "active_cues": [
                        "closure", "yawn", "perclos", "gaze", "nod"
                    ],
                    "calibration": "personal",
                    "gaze_source": "geometric",
                },
            }
        ),
        encoding="utf-8",
    )
    return source


def score(config, **changes):
    evidence = {
        "closure": 0.0,
        "yawn": 0.0,
        "perclos": 0.0,
        "gaze": 0.0,
        "nod": 0.0,
    }
    evidence.update(changes)
    return normalized_score(
        evidence,
        config.weights,
        config.active_cues,
    )


def test_configurations_a_to_f_are_available(tmp_path):
    path = write_config(tmp_path)

    assert available_config_ids(path) == ("A", "B", "C", "D", "E", "F")


def test_a_uses_only_closure(tmp_path):
    config = load_ablation(write_config(tmp_path), "A")

    assert score(config, closure=1.0) == 1.0
    assert score(config, yawn=1.0, perclos=1.0, gaze=1.0, nod=1.0) == 0.0


def test_b_changes_with_yawn_but_not_perclos(tmp_path):
    config = load_ablation(write_config(tmp_path), "B")

    assert score(config, yawn=1.0) > 0.0
    assert score(config, perclos=1.0) == 0.0


def test_c_changes_with_perclos_but_not_gaze(tmp_path):
    config = load_ablation(write_config(tmp_path), "C")

    assert score(config, perclos=1.0) > 0.0
    assert score(config, gaze=1.0) == 0.0


def test_d_changes_with_gaze_but_not_nod(tmp_path):
    config = load_ablation(write_config(tmp_path), "D")

    assert score(config, gaze=1.0) > 0.0
    assert score(config, nod=1.0) == 0.0


def test_e_changes_with_nod(tmp_path):
    config = load_ablation(write_config(tmp_path), "E")

    assert score(config, nod=1.0) > 0.0


def test_f_requires_personal_calibration(tmp_path):
    config = load_ablation(write_config(tmp_path), "F")

    assert config.calibration == "personal"
    assert config.gaze_source == "geometric"
    assert set(config.active_cues) == {
        "closure", "yawn", "perclos", "gaze", "nod"
    }


def test_scores_always_remain_between_zero_and_one(tmp_path):
    path = write_config(tmp_path)

    for config_id in available_config_ids(path):
        config = load_ablation(path, config_id)

        for value in (0.0, 0.25, 0.50, 0.75, 1.0):
            result = score(
                config,
                closure=value,
                yawn=value,
                perclos=value,
                gaze=value,
                nod=value,
            )
            assert 0.0 <= result <= 1.0
            assert math.isclose(result, value)


def test_hash_is_stable_and_configuration_specific(tmp_path):
    path = write_config(tmp_path)
    first = load_ablation(path, "A")
    second = load_ablation(path, "A")
    different = load_ablation(path, "B")

    assert first.config_sha256 == second.config_sha256
    assert len(first.config_sha256) == 64
    assert first.config_sha256 != different.config_sha256


def test_unknown_configuration_is_rejected(tmp_path):
    path = write_config(tmp_path)

    with pytest.raises(ValueError):
        load_ablation(path, "Z")
