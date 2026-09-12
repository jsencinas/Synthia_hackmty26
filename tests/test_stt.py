import pytest

from src.escalation import detect_call
from src.stt_elevenlabs import STTError, transcribe
from src.transcript_stage import run as run_transcript
from src.types import StageResult


def _json_voice_uncertain():
    return (
        lambda path: StageResult("json", True, 0.60, 60.0),
        lambda *args, **kwargs: StageResult("voice", True, 0.55, 55.0),
    )


def test_transcribe_missing_key(monkeypatch):
    monkeypatch.setattr("src.stt_elevenlabs.ELEVENLABS_API_KEY", "")
    with pytest.raises(STTError) as exc:
        transcribe("audio.wav")
    assert exc.value.code == "stt_unconfigured"


def test_transcribe_success(monkeypatch, tmp_path):
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr("src.stt_elevenlabs.ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setattr("src.stt_elevenlabs._load_cache", lambda path: None)
    monkeypatch.setattr("src.stt_elevenlabs._save_cache", lambda path, payload: None)
    monkeypatch.setattr(
        "src.stt_elevenlabs._call_scribe",
        lambda path: {"text": "mande no tengo eso", "words": []},
    )
    payload = transcribe(str(audio))
    assert payload["text"] == "mande no tengo eso"


def test_transcribe_unauthorized(monkeypatch, tmp_path):
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr("src.stt_elevenlabs.ELEVENLABS_API_KEY", "bad-key")
    monkeypatch.setattr("src.stt_elevenlabs._load_cache", lambda path: None)

    class Boom(Exception):
        status_code = 401

    def fake_call(path):
        raise Boom("unauthorized")

    monkeypatch.setattr("src.stt_elevenlabs._call_scribe", fake_call)
    with pytest.raises(STTError) as exc:
        transcribe(str(audio))
    assert exc.value.code == "stt_failed"


def test_empty_text_returns_error(monkeypatch):
    monkeypatch.setattr(
        "src.transcript_stage.transcribe",
        lambda path: {"text": "", "words": []},
    )
    result = run_transcript("audio.wav", None)
    assert result.error == "stt_failed"


def test_orchestrator_keeps_voice_when_stt_fails(monkeypatch):
    json_fn, voice_fn = _json_voice_uncertain()
    monkeypatch.setattr("src.escalation.run_json", json_fn)
    monkeypatch.setattr("src.escalation.run_voice", voice_fn)
    monkeypatch.setattr(
        "src.escalation.run_stt",
        lambda *args, **kwargs: StageResult("stt", False, 0.5, 0.0, error="stt_failed"),
    )
    payload = detect_call("audio.wav", turns_path="turns.json")
    assert payload["stopped_at"] == "voice"
    assert payload["confidence"] == 0.55
