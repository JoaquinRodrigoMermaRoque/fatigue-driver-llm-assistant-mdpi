from enum import Enum

from src.speech import (
    SpeechPriority,
    alert_for_state,
    normalize_state_name,
)


class ExampleState(Enum):
    CRITICAL = "CRITICAL"


def test_normal_state_has_no_spoken_alert():
    assert alert_for_state("NORMAL") is None


def test_suspected_message_is_low_priority():
    alert = alert_for_state("suspected")

    assert alert is not None
    assert alert.priority == SpeechPriority.SUSPECTED
    assert "signos leves" in alert.text


def test_fatigue_message_recommends_safe_rest():
    alert = alert_for_state("FATIGUE")

    assert alert is not None
    assert alert.priority == SpeechPriority.FATIGUE
    assert "lugar seguro" in alert.text


def test_critical_has_maximum_priority():
    alert = alert_for_state(ExampleState.CRITICAL)

    assert alert is not None
    assert alert.priority == SpeechPriority.CRITICAL
    assert "Alerta crítica" in alert.text


def test_recovery_confirms_acknowledgement():
    alert = alert_for_state("RECOVERY")

    assert alert is not None
    assert "reconocida" in alert.text


def test_unknown_state_is_silent():
    assert alert_for_state("UNKNOWN") is None


def test_state_name_accepts_enums_and_whitespace():
    assert normalize_state_name(ExampleState.CRITICAL) == (
        "CRITICAL"
    )
    assert normalize_state_name(" fatigue ") == "FATIGUE"
