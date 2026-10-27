import math

import pytest

from src.fusion import (
    CUE_NAMES,
    DEFAULT_WEIGHTS,
    FatigueFusion,
    compute_evidence,
    normalized_score,
    ramp,
)


def test_ramp_limits_and_midpoint():
    assert ramp(0.0, 1.0, 3.0) == 0.0
    assert ramp(4.0, 1.0, 3.0) == 1.0
    assert ramp(2.0, 1.0, 3.0) == 0.5


def test_normal_input_has_zero_evidence():
    result = FatigueFusion().evaluate(
        max_closure_s=0.20,
        perclos_ratio=0.05,
        yawns_last_5_min=0,
        gaze_away_s=0.50,
        nods_last_2_min=0,
    )

    assert result.score == 0.0
    assert result.reasons == ()


def test_maximum_input_has_score_one():
    result = FatigueFusion().evaluate(
        max_closure_s=2.10,
        perclos_ratio=0.50,
        yawns_last_5_min=3,
        gaze_away_s=5.50,
        nods_last_2_min=3,
    )

    assert result.score == 1.0
    assert set(result.reasons) == set(CUE_NAMES)


def test_intermediate_values_are_reproducible():
    evidence = compute_evidence(
        max_closure_s=1.20,
        perclos_ratio=0.275,
        yawns_last_5_min=1,
        gaze_away_s=3.25,
        nods_last_2_min=1,
    )

    assert math.isclose(evidence.closure, 0.5)
    assert math.isclose(evidence.perclos, 0.5)
    assert math.isclose(evidence.yawn, 1.0 / 3.0)
    assert math.isclose(evidence.gaze, 0.5)
    assert math.isclose(evidence.nod, 1.0 / 3.0)


def test_ablation_uses_only_active_weights():
    evidence = {
        "closure": 1.0,
        "perclos": 0.0,
        "yawn": 0.0,
        "gaze": 0.0,
        "nod": 0.0,
    }

    score_a = normalized_score(
        evidence,
        DEFAULT_WEIGHTS,
        ["closure"],
    )

    score_all = normalized_score(
        evidence,
        DEFAULT_WEIGHTS,
        CUE_NAMES,
    )

    assert score_a == 1.0
    assert math.isclose(score_all, 0.30)


def test_nonfinite_input_is_rejected():
    with pytest.raises(ValueError):
        compute_evidence(
            max_closure_s=float("nan"),
            perclos_ratio=0.0,
            yawns_last_5_min=0,
            gaze_away_s=0.0,
            nods_last_2_min=0,
        )
