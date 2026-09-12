import base64

from fastapi.testclient import TestClient

from api.app import app
from src.types import StageResult


client = TestClient(app)


def test_detect_response_shape(monkeypatch):
    wav = b"RIFF" + b"\x00" * 44
    monkeypatch.setattr(
        "api.app.detect_call",
        lambda audio_path, turns_path=None: {
            "is_synthetic": True,
            "confidence": 0.87,
            "stopped_at": "json",
        },
    )
    response = client.post("/detect", json={"audio": base64.b64encode(wav).decode("ascii")})
    assert response.status_code == 200
    body = response.json()
    assert body["is_synthetic"] is True
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body.keys()) == {"is_synthetic", "confidence"}


def test_detect_rejects_empty_payload():
    response = client.post("/detect", json={})
    assert response.status_code == 400


def test_detect_all_stages_failed(monkeypatch):
    wav = b"RIFF" + b"\x00" * 44
    monkeypatch.setattr(
        "api.app.detect_call",
        lambda audio_path, turns_path=None: {
            "is_synthetic": False,
            "confidence": 0.5,
            "error": "all_stages_failed",
        },
    )
    response = client.post("/detect", json={"audio": base64.b64encode(wav).decode("ascii")})
    assert response.status_code == 503


def test_stage_result_confidence_bounds():
    result = StageResult("json", True, 0.87, 87.0)
    assert 0.0 <= result.p_synthetic <= 1.0
