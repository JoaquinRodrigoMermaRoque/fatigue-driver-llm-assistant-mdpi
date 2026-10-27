from src.voice import (
    VoiceCommand,
    choose_input_device,
    classify_voice_command,
    normalize_voice_text,
)


def test_normalize_voice_text():
    assert (
        normalize_voice_text("  ¡DE ACUERDO!  ")
        == "de acuerdo"
    )


def test_ack_phrases_are_recognized():
    for phrase in (
        "entendido",
        "estoy bien",
        "de acuerdo",
        "confirmo",
    ):
        assert (
            classify_voice_command(phrase)
            == VoiceCommand.ACK
        )


def test_repeat_and_cancel_are_recognized():
    assert (
        classify_voice_command("repetir")
        == VoiceCommand.REPEAT
    )
    assert (
        classify_voice_command("cancelar")
        == VoiceCommand.CANCEL
    )


def test_partial_or_unknown_phrases_are_ignored():
    for phrase in (
        "bien [unk]",
        "bien",
        "entendido ahora",
        "[unk]",
        "",
    ):
        assert classify_voice_command(phrase) is None


def test_choose_input_device_prioritizes_pulse_source():
    devices = [
        {
            "name": "HDA Intel PCH: ALC269VB Analog",
            "max_input_channels": 2,
        },
        {
            "name": "Default Source",
            "max_input_channels": 32,
        },
        {
            "name": (
                "alsa_input.pci-0000_00_1f.3."
                "analog-stereo"
            ),
            "max_input_channels": 2,
        },
    ]

    index, name = choose_input_device(
        devices,
        default_input=1,
    )

    assert index == 2
    assert name.startswith("alsa_input")


def test_choose_input_device_uses_first_available_fallback():
    devices = [
        {
            "name": "HDMI output",
            "max_input_channels": 0,
        },
        {
            "name": "USB microphone",
            "max_input_channels": 1,
        },
    ]

    assert choose_input_device(devices) == (
        1,
        "USB microphone",
    )
