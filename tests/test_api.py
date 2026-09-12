import base64
import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from api.app import app
from src.types import StageResult


client = TestClient(app)


def _wav_base64() -> str:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros((800, 2), dtype=np.float32), 8000, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_detect_response_shape(monkeypatch):
    monkeypatch.setattr(
        "api.app.detect_call",
        lambda audio_path: {
            "is_synthetic": True,
            "confidence": 0.87,
            "stopped_at": "json",
        },
    )
    response = client.post("/detect", json={"audio": _wav_base64()})
    assert response.status_code == 200
    body = response.json()
    assert body["is_synthetic"] is True
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body.keys()) == {"is_synthetic", "confidence"}


def test_detect_rejects_empty_payload():
    response = client.post("/detect", json={})
    assert response.status_code == 400


def test_detect_all_stages_failed(monkeypatch):
    monkeypatch.setattr(
        "api.app.detect_call",
        lambda audio_path: {
            "is_synthetic": False,
            "confidence": 0.5,
            "error": "all_stages_failed",
        },
    )
    response = client.post("/detect", json={"audio": _wav_base64()})
    assert response.status_code == 503


def test_detect_rejects_malformed_audio():
    response = client.post(
        "/detect",
        json={"audio": base64.b64encode(b"not a wav payload" * 4).decode("ascii")},
    )
    assert response.status_code == 400


def test_stage_result_confidence_bounds():
    result = StageResult("json", True, 0.87, 87.0)
    assert 0.0 <= result.p_synthetic <= 1.0
