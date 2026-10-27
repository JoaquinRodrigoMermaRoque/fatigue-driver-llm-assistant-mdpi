import csv

import pytest

from src.feature_logger import RAW_FEATURE_FIELDS, RawFeatureLogger


H5_FIELDS = {
    "fusion_score",
    "evidence_closure",
    "evidence_perclos",
    "evidence_yawn",
    "evidence_gaze",
    "evidence_nod",
    "fusion_reasons",
    "fatigue_state",
    "fatigue_state_label",
    "fatigue_state_event",
    "yawns_last_5_min",
    "nods_last_2_min",
    "fusion_data_sufficient",
    "fusion_available_weight_ratio",
}


def test_h5_fields_are_present_and_unique():
    assert H5_FIELDS <= set(RAW_FEATURE_FIELDS)
    assert len(RAW_FEATURE_FIELDS) == len(set(RAW_FEATURE_FIELDS))
    assert len(RAW_FEATURE_FIELDS) == 72


def test_h5_values_are_written(tmp_path):
    path = tmp_path / "raw_features.csv"
    logger = RawFeatureLogger(
        path,
        subject_id="pilot_test",
        clip_id="synthetic",
        source="pytest",
        config_id="h5_test",
        flush_every=1,
    )

    logger.write(
        timestamp_ms=1_000,
        fusion_score=0.42,
        evidence_closure=0.50,
        evidence_perclos=0.40,
        evidence_yawn=0.30,
        evidence_gaze=0.20,
        evidence_nod=0.10,
        fusion_reasons="closure;perclos",
        fatigue_state="SUSPECTED",
        fatigue_state_label="FATIGA LEVE",
        fatigue_state_event="state_suspected",
        yawns_last_5_min=1,
        nods_last_2_min=2,
        fusion_data_sufficient=True,
        fusion_available_weight_ratio=0.90,
    )
    logger.close()

    with path.open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))

    assert row["fusion_score"] == "0.42"
    assert row["fatigue_state"] == "SUSPECTED"
    assert row["fusion_data_sufficient"] == "1"
    assert row["yawns_last_5_min"] == "1"


def test_old_csv_header_is_rejected(tmp_path):
    path = tmp_path / "raw_features.csv"
    path.write_text("old,header\n1,2\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        RawFeatureLogger(
            path,
            subject_id="pilot_test",
            clip_id="synthetic",
            source="pytest",
        )
