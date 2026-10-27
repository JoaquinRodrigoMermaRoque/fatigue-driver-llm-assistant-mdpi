import csv
from pathlib import Path

import pytest

from src.ablation_config import load_ablation
from src.calibration import CalibrationProfile
from src.run_experiment import (
    PREDICTION_FIELDS,
    load_feature_run,
    prediction_identities,
    replay_configuration,
    verify_same_frames,
    write_predictions,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "ablations.json"


def synthetic_profile():
    return CalibrationProfile(
        schema_version=1,
        subject_id="pilot_test",
        created_at_utc="2026-01-01T00:00:00Z",
        glasses=True,
        sample_counts={},
        valid_ratios={},
        ear_open_median=0.35,
        ear_closed_median=0.08,
        ear_threshold=0.20,
        mar_neutral_median=0.20,
        mar_open_median=1.00,
        mar_threshold=0.50,
        gaze_center_h=0.50,
        gaze_center_v=0.42,
        gaze_horizontal_low=0.35,
        gaze_horizontal_high=0.65,
        gaze_vertical_low=0.38,
        gaze_vertical_high=0.46,
        pose_baseline_pitch_deg=6.55,
        pose_baseline_yaw_deg=1.25,
        pose_baseline_roll_deg=0.75,
        pose_pitch_threshold_deg=15.0,
        pose_yaw_threshold_deg=20.0,
        pose_roll_threshold_deg=20.0,
    )


def synthetic_rows(frame_count=20):
    rows = []

    for index in range(frame_count):
        rows.append(
            {
                "feature_run_id": "run_test",
                "subject_id": "pilot_test",
                "clip_id": "clip_test",
                "frame_index": str(index),
                "t_ms": str(index * 100),
                "face_detected": "1",
                "valid_eye": "1",
                "valid_mouth": "1",
                "valid_pose": "1",
                "ear": "0.35",
                "mar": "0.20",
                "gaze_h_raw": "0.50",
                "gaze_v_raw": "0.42",
                "gaze_h_asymmetry": "0.01",
                "gaze_v_asymmetry": "0.01",
                "pitch_deg": "6.55",
                "yaw_deg": "1.25",
                "roll_deg": "0.75",
            }
        )

    return rows


def write_features(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_load_feature_run_preserves_order(tmp_path):
    path = tmp_path / "features.csv"
    rows = synthetic_rows(5)
    write_features(path, list(reversed(rows)))

    run_id, loaded = load_feature_run(path, "run_test")

    assert run_id == "run_test"
    assert [int(row["frame_index"]) for row in loaded] == list(range(5))


def test_missing_raw_column_is_rejected(tmp_path):
    path = tmp_path / "features.csv"
    rows = synthetic_rows(2)

    for row in rows:
        row.pop("ear")

    write_features(path, rows)

    with pytest.raises(RuntimeError):
        load_feature_run(path)


def test_a_to_f_use_exactly_the_same_frames():
    rows = synthetic_rows()
    profile = synthetic_profile()
    predictions = {}

    for config_id in "ABCDEF":
        config = load_ablation(CONFIG_PATH, config_id)
        predictions[config_id] = replay_configuration(
            rows=rows,
            config=config,
            profile=profile,
            experiment_id="experiment_test",
        )

    verify_same_frames(predictions)
    reference = prediction_identities(predictions["A"])

    assert len(reference) == len(rows)
    assert all(
        prediction_identities(predictions[config_id]) == reference
        for config_id in "ABCDEF"
    )
    assert all(
        row["calibration"] == "personal"
        for row in predictions["F"]
    )


def test_changed_identity_is_detected():
    base = {
        "subject_id": "pilot_test",
        "clip_id": "clip_test",
        "frame_index": 0,
        "t_ms": 0,
    }
    altered = dict(base)
    altered["t_ms"] = 1

    with pytest.raises(RuntimeError):
        verify_same_frames({"A": [base], "B": [altered]})


def test_predictions_include_hash_and_are_written(tmp_path):
    rows = synthetic_rows(3)
    config = load_ablation(CONFIG_PATH, "A")
    predictions = replay_configuration(
        rows=rows,
        config=config,
        profile=None,
        experiment_id="experiment_test",
    )
    output = tmp_path / "predictions.csv"

    write_predictions(output, {"A": predictions})

    with output.open(encoding="utf-8", newline="") as file:
        written = list(csv.DictReader(file))

    assert len(written) == 3
    assert tuple(written[0]) == PREDICTION_FIELDS
    assert written[0]["config_sha256"] == config.config_sha256
    assert written[0]["acknowledgement_policy"] == (
        "automatic_after_critical"
    )
