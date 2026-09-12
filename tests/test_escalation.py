from src.escalation import detect_call, finalize, is_certain
from src.types import StageResult


def _result(certainty_pct: float, p_synthetic: float = 0.8, error: str | None = None) -> StageResult:
    return StageResult(
        stage="json",
        is_synthetic=p_synthetic >= 0.5,
        p_synthetic=p_synthetic,
        certainty_pct=certainty_pct,
        error=error,
    )


def test_certainty_74_9_escalates():
    assert is_certain(_result(74.9)) is False


def test_certainty_75_stops():
    assert is_certain(_result(75.0)) is True


def test_human_90_percent_stops():
    result = _result(certainty_pct=90.0, p_synthetic=0.10)
    assert is_certain(result) is True
    payload = finalize(result)
    assert payload["is_synthetic"] is False
    assert payload["confidence"] == 0.10


def test_error_always_escalates():
    assert is_certain(_result(99.0, error="json_failed")) is False


def test_detect_call_stops_after_certain_json(monkeypatch):
    called = {"voice": False, "stt": False}

    monkeypatch.setattr(
        "src.escalation.run_json",
        lambda path: _result(80.0, p_synthetic=0.82),
    )

    def fake_voice(*args, **kwargs):
        called["voice"] = True
        return _result(60.0)

    def fake_stt(*args, **kwargs):
        called["stt"] = True
        return _result(60.0)

    monkeypatch.setattr("src.escalation.run_voice", fake_voice)
    monkeypatch.setattr("src.escalation.run_stt", fake_stt)

    payload = detect_call("audio.wav", turns_path="turns.json")
    assert payload["stopped_at"] == "json"
    assert called["voice"] is False
    assert called["stt"] is False
