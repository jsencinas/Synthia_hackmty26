from src.escalation import detect_call
from src.types import StageResult


def test_voice_skipped_when_json_certainty_80(monkeypatch):
    called = {"voice": False}

    monkeypatch.setattr(
        "src.escalation.run_json",
        lambda path: StageResult("json", True, 0.80, 80.0),
    )

    def fake_voice(*args, **kwargs):
        called["voice"] = True
        return StageResult("voice", True, 0.60, 60.0)

    monkeypatch.setattr("src.escalation.run_voice", fake_voice)
    monkeypatch.setattr(
        "src.escalation.run_stt",
        lambda *args, **kwargs: StageResult("stt", True, 0.60, 60.0),
    )

    payload = detect_call("audio.wav", turns_path="turns.json")
    assert payload["stopped_at"] == "json"
    assert called["voice"] is False


def test_voice_runs_when_json_certainty_60(monkeypatch):
    called = {"voice": False}

    monkeypatch.setattr(
        "src.escalation.run_json",
        lambda path: StageResult("json", True, 0.60, 60.0),
    )

    def fake_voice(*args, **kwargs):
        called["voice"] = True
        return StageResult("voice", True, 0.90, 90.0)

    monkeypatch.setattr("src.escalation.run_voice", fake_voice)
    monkeypatch.setattr(
        "src.escalation.run_stt",
        lambda *args, **kwargs: StageResult("stt", True, 0.60, 60.0),
    )

    payload = detect_call("audio.wav", turns_path="turns.json")
    assert called["voice"] is True
    assert payload["stopped_at"] == "voice"
    assert payload["confidence"] == 0.90
