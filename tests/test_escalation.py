from src.escalation import detect_call, finalize
from src.types import StageResult


def _result(p: float, stage: str, error: str | None = None) -> StageResult:
    return StageResult(stage, p >= 0.5, p, 0.0, error=error)


def test_finalize_reports_decision_confidence():
    human = finalize(_result(0.1, "fusion"))
    synthetic = finalize(_result(0.8, "fusion"))
    assert human["is_synthetic"] is False
    assert human["confidence"] == 0.9
    assert synthetic["confidence"] == 0.8


def test_agreeing_heads_return_fusion(monkeypatch):
    monkeypatch.setattr(
        "src.escalation.extract_turns",
        lambda path: {"turns": [{"channel": 0, "start": 0.0, "end": 1.0}]},
    )
    monkeypatch.setattr("src.escalation.run_timing", lambda payload: _result(0.8, "timing"))
    monkeypatch.setattr(
        "src.escalation.run_voice", lambda path, payload: _result(0.7, "voice")
    )
    monkeypatch.setattr(
        "src.escalation.fuse",
        lambda timing, voice: _result(0.76, "fusion"),
    )
    monkeypatch.setattr(
        "src.escalation.run_stt",
        lambda *args: (_ for _ in ()).throw(AssertionError("STT should not run")),
    )
    payload = detect_call("audio.wav")
    assert payload["stopped_at"] == "fusion"
    assert payload["confidence"] == 0.76


def test_disagreement_uses_stt(monkeypatch):
    monkeypatch.setattr(
        "src.escalation.extract_turns",
        lambda path: {"turns": [{"channel": 0, "start": 0.0, "end": 1.0}]},
    )
    monkeypatch.setattr("src.escalation.run_timing", lambda payload: _result(0.8, "timing"))
    monkeypatch.setattr(
        "src.escalation.run_voice", lambda path, payload: _result(0.2, "voice")
    )
    monkeypatch.setattr(
        "src.escalation.fuse", lambda timing, voice: _result(0.6, "fusion")
    )
    monkeypatch.setattr(
        "src.escalation.run_stt", lambda path, payload: _result(0.15, "stt")
    )
    payload = detect_call("audio.wav")
    assert payload["stopped_at"] == "stt"
    assert payload["is_synthetic"] is False
    assert payload["confidence"] == 0.85


def test_failed_stt_keeps_fusion(monkeypatch):
    monkeypatch.setattr(
        "src.escalation.extract_turns",
        lambda path: {"turns": [{"channel": 0, "start": 0.0, "end": 1.0}]},
    )
    monkeypatch.setattr("src.escalation.run_timing", lambda payload: _result(0.8, "timing"))
    monkeypatch.setattr(
        "src.escalation.run_voice", lambda path, payload: _result(0.2, "voice")
    )
    monkeypatch.setattr(
        "src.escalation.fuse", lambda timing, voice: _result(0.6, "fusion")
    )
    monkeypatch.setattr(
        "src.escalation.run_stt",
        lambda path, payload: _result(0.5, "stt", "stt_failed"),
    )
    assert detect_call("audio.wav")["stopped_at"] == "fusion"
