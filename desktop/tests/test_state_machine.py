import pytest

from src.state_machine import FatigueStateMachine, State


def test_one_bad_frame_does_not_trigger():
    machine = FatigueStateMachine()

    machine.update(0, 0.0, 0.0)
    state, event = machine.update(100, 0.90, 0.10)
    state, event = machine.update(200, 0.0, 0.0)

    assert state == State.NORMAL
    assert event is None


def test_suspected_requires_five_seconds():
    machine = FatigueStateMachine()

    assert machine.update(0, 0.35, 0.0)[0] == State.NORMAL
    assert machine.update(4_999, 0.35, 0.0)[0] == State.NORMAL

    state, event = machine.update(5_000, 0.35, 0.0)

    assert state == State.SUSPECTED
    assert event == "state_suspected"


def test_candidate_is_cancelled_when_score_recovers():
    machine = FatigueStateMachine()

    machine.update(0, 0.35, 0.0)
    machine.update(4_000, 0.35, 0.0)
    machine.update(4_100, 0.10, 0.0)

    assert machine.update(5_000, 0.35, 0.0)[0] == State.NORMAL
    assert machine.update(9_999, 0.35, 0.0)[0] == State.NORMAL
    assert machine.update(10_000, 0.35, 0.0)[0] == State.SUSPECTED


def test_moderate_fatigue_requires_five_seconds():
    machine = FatigueStateMachine()

    machine.update(0, 0.60, 0.0)
    assert machine.update(4_999, 0.60, 0.0)[0] == State.NORMAL

    state, event = machine.update(5_000, 0.60, 0.0)

    assert state == State.FATIGUE
    assert event == "state_fatigue"


def test_critical_score_requires_two_seconds():
    machine = FatigueStateMachine()

    machine.update(0, 0.80, 0.0)
    assert machine.update(1_999, 0.80, 0.0)[0] == State.NORMAL

    state, event = machine.update(2_000, 0.80, 0.0)

    assert state == State.CRITICAL
    assert event == "state_critical"


def test_two_second_closure_is_immediately_critical():
    machine = FatigueStateMachine()

    state, event = machine.update(5_000, 0.20, 2.10)

    assert state == State.CRITICAL
    assert event == "prolonged_eye_closure"


def test_critical_requires_acknowledgement_and_low_score():
    machine = FatigueStateMachine()

    machine.update(0, 0.20, 2.10)

    # Aunque pasen 40 s, sin reconocimiento permanece enclavado.
    assert machine.update(40_000, 0.0, 0.0)[0] == State.CRITICAL

    # El reconocimiento inicia, pero no completa, la recuperación.
    assert machine.update(41_000, 0.0, 0.0, True)[0] == State.CRITICAL
    assert machine.update(70_999, 0.0, 0.0)[0] == State.CRITICAL

    state, event = machine.update(71_000, 0.0, 0.0)

    assert state == State.RECOVERY
    assert event == "state_recovery"


def test_recovery_requires_ten_stable_seconds_to_normal():
    machine = FatigueStateMachine()

    machine.update(0, 0.20, 2.10)
    machine.update(1_000, 0.0, 0.0, True)
    machine.update(31_000, 0.0, 0.0)

    assert machine.state == State.RECOVERY
    machine.update(31_001, 0.0, 0.0)
    assert machine.update(41_000, 0.0, 0.0)[0] == State.RECOVERY

    state, event = machine.update(41_001, 0.0, 0.0)

    assert state == State.NORMAL
    assert event == "state_normal"


def test_timestamps_must_be_monotonic():
    machine = FatigueStateMachine()

    machine.update(100, 0.0, 0.0)

    with pytest.raises(ValueError):
        machine.update(99, 0.0, 0.0)


def test_invalid_score_is_rejected():
    machine = FatigueStateMachine()

    with pytest.raises(ValueError):
        machine.update(0, float("nan"), 0.0)


def test_invalid_data_cancels_candidate_without_changing_state():
    machine = FatigueStateMachine()

    machine.update(0, 0.35, 0.0)
    machine.update(4_000, 0.35, 0.0)

    state, event = machine.invalidate(4_100)

    assert state == State.NORMAL
    assert event is None

    # La persistencia debe comenzar otra vez después del dato inválido.
    assert machine.update(5_000, 0.35, 0.0)[0] == State.NORMAL
    assert machine.update(9_999, 0.35, 0.0)[0] == State.NORMAL
    assert machine.update(10_000, 0.35, 0.0)[0] == State.SUSPECTED


def test_acknowledgement_is_visible_and_reports_progress():
    machine = FatigueStateMachine()

    machine.update(0, 0.20, 2.10)
    state, event = machine.update(1_000, 0.0, 0.0, True)

    assert state == State.CRITICAL
    assert event == "critical_acknowledged"
    assert machine.alert_acknowledged
    assert machine.critical_recovery_elapsed_s == 0.0

    machine.update(11_000, 0.0, 0.0)

    assert machine.critical_recovery_elapsed_s == 10.0


def test_new_prolonged_closure_cancels_acknowledgement():
    machine = FatigueStateMachine()

    machine.update(0, 0.20, 2.10)
    machine.update(1_000, 0.0, 0.0, True)

    assert machine.alert_acknowledged

    machine.update(2_000, 0.80, 2.10)

    assert not machine.alert_acknowledged
    assert machine.critical_recovery_elapsed_s == 0.0
