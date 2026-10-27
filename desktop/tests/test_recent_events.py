import pytest

from src.recent_events import RecentEventWindows, RollingEventCounter


def test_single_event_is_counted():
    counter = RollingEventCounter(window_s=10.0)

    assert counter.update(timestamp_s=0.0, event=False) == 0
    assert counter.update(timestamp_s=1.0, event=True) == 1


def test_sustained_true_counts_only_once():
    counter = RollingEventCounter(window_s=10.0)

    assert counter.update(timestamp_s=1.0, event=True) == 1
    assert counter.update(timestamp_s=2.0, event=True) == 1
    assert counter.update(timestamp_s=3.0, event=True) == 1


def test_two_separate_pulses_count_twice():
    counter = RollingEventCounter(window_s=10.0)

    counter.update(timestamp_s=1.0, event=True)
    counter.update(timestamp_s=2.0, event=False)

    assert counter.update(timestamp_s=3.0, event=True) == 2


def test_old_events_expire_from_window():
    counter = RollingEventCounter(window_s=10.0)

    counter.update(timestamp_s=1.0, event=True)
    counter.update(timestamp_s=2.0, event=False)

    # El evento de t=1 todavía está incluido exactamente en el límite.
    assert counter.update(timestamp_s=11.0, event=False) == 1
    assert counter.update(timestamp_s=11.001, event=False) == 0


def test_yawn_and_nod_windows_are_independent():
    windows = RecentEventWindows()

    metrics = windows.update(
        timestamp_s=1.0,
        yawn_event=True,
        nod_event=False,
    )

    assert metrics.yawns_last_5_min == 1
    assert metrics.nods_last_2_min == 0

    metrics = windows.update(
        timestamp_s=2.0,
        yawn_event=False,
        nod_event=True,
    )

    assert metrics.yawns_last_5_min == 1
    assert metrics.nods_last_2_min == 1


def test_invalid_frame_never_adds_event():
    counter = RollingEventCounter(window_s=10.0)

    assert counter.update(
        timestamp_s=1.0,
        event=True,
        valid=False,
    ) == 0

    # Al volver la señal válida puede registrarse un evento nuevo.
    assert counter.update(
        timestamp_s=2.0,
        event=True,
        valid=True,
    ) == 1


def test_timestamps_must_be_monotonic():
    counter = RollingEventCounter(window_s=10.0)

    counter.update(timestamp_s=2.0, event=False)

    with pytest.raises(ValueError):
        counter.update(timestamp_s=1.0, event=False)
