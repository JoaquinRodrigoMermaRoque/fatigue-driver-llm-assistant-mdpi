import math

from src.fatigue_engine import FatigueEngine
from src.state_machine import State


def valid_update(engine, timestamp_s, **changes):
    values = {
        "timestamp_s": timestamp_s,
        "closure_s": 0.0,
        "perclos_ratio": 0.0,
        "gaze_away_s": 0.0,
        "yawn_event": False,
        "nod_event": False,
        "face_valid": True,
        "eye_valid": True,
        "mouth_valid": True,
        "gaze_valid": True,
        "pose_valid": True,
    }
    values.update(changes)
    return engine.update(**values)


def test_normal_valid_frame_has_zero_score():
    metrics = valid_update(FatigueEngine(), 0.0)

    assert metrics.score == 0.0
    assert metrics.state == State.NORMAL
    assert metrics.data_sufficient
    assert metrics.available_weight_ratio == 1.0


def test_event_windows_feed_fusion():
    engine = FatigueEngine()

    metrics = valid_update(engine, 1.0, yawn_event=True)

    assert metrics.yawns_last_5_min == 1
    assert metrics.evidence.yawn == 1.0 / 3.0
    assert metrics.score > 0.0


def test_sustained_event_flag_is_not_duplicated():
    engine = FatigueEngine()

    valid_update(engine, 1.0, nod_event=True)
    metrics = valid_update(engine, 2.0, nod_event=True)

    assert metrics.nods_last_2_min == 1


def test_no_face_does_not_add_events_or_change_state():
    engine = FatigueEngine()

    metrics = engine.update(
        timestamp_s=1.0,
        closure_s=float("nan"),
        perclos_ratio=float("nan"),
        gaze_away_s=float("nan"),
        yawn_event=True,
        nod_event=True,
        face_valid=False,
        eye_valid=False,
        mouth_valid=False,
        gaze_valid=False,
        pose_valid=False,
    )

    assert metrics.state == State.NORMAL
    assert not metrics.data_sufficient
    assert metrics.available_weight_ratio == 0.0
    assert metrics.yawns_last_5_min == 0
    assert metrics.nods_last_2_min == 0


def test_no_face_cancels_pending_transition():
    engine = FatigueEngine()

    valid_update(engine, 0.0, perclos_ratio=0.40)
    valid_update(engine, 4.0, perclos_ratio=0.40)

    engine.update(
        timestamp_s=4.1,
        closure_s=0.0,
        perclos_ratio=0.0,
        gaze_away_s=0.0,
        yawn_event=False,
        nod_event=False,
        face_valid=False,
        eye_valid=False,
        mouth_valid=False,
        gaze_valid=False,
        pose_valid=False,
    )

    assert valid_update(
        engine,
        5.0,
        perclos_ratio=0.40,
    ).state == State.NORMAL


def test_closed_eyes_can_trigger_critical_with_invalid_gaze():
    engine = FatigueEngine()

    metrics = valid_update(
        engine,
        2.0,
        closure_s=2.10,
        gaze_valid=False,
    )

    assert metrics.data_sufficient
    assert metrics.state == State.CRITICAL
    assert metrics.state_event == "prolonged_eye_closure"


def test_gaze_invalid_does_not_mean_all_data_invalid():
    engine = FatigueEngine()

    metrics = valid_update(
        engine,
        1.0,
        gaze_away_s=float("nan"),
        gaze_valid=False,
    )

    assert metrics.data_sufficient
    assert math.isclose(
        metrics.available_weight_ratio,
        0.90,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


def test_critical_acknowledgement_and_progress_are_exposed():
    engine = FatigueEngine()

    valid_update(engine, 0.0, closure_s=2.10)
    metrics = valid_update(engine, 1.0, acknowledged=True)

    assert metrics.state == State.CRITICAL
    assert metrics.state_event == "critical_acknowledged"
    assert metrics.alert_acknowledged
    assert metrics.critical_recovery_elapsed_s == 0.0
    assert metrics.critical_recovery_required_s == 30.0

    metrics = valid_update(engine, 11.0)

    assert metrics.alert_acknowledged
    assert metrics.critical_recovery_elapsed_s == 10.0
