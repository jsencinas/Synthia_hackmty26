from __future__ import annotations

import base64
import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.escalation import detect_call


app = FastAPI(title="HackMTY caller detector")


class DetectRequest(BaseModel):
    audio: str | None = None
    wav_base64: str | None = None
    audio_base64: str | None = None
    turns_path: str | None = Field(default=None, description="Optional local turns JSON")


def _decode_wav(request: DetectRequest) -> bytes:
    payload = request.audio or request.wav_base64 or request.audio_base64
    if not payload:
        raise HTTPException(status_code=400, detail="Missing base64 WAV in audio, wav_base64, or audio_base64.")
    try:
        return base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 WAV: {exc}") from exc


@app.post("/detect")
def detect(request: DetectRequest) -> dict:
    wav_bytes = _decode_wav(request)
    if len(wav_bytes) < 44:
        raise HTTPException(status_code=400, detail="WAV payload is too small.")

    fd, audio_path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(wav_bytes)
        result = detect_call(audio_path, turns_path=request.turns_path)
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass

    if result.get("error") == "all_stages_failed":
        raise HTTPException(status_code=503, detail="All detection stages failed.")

    return {
        "is_synthetic": bool(result["is_synthetic"]),
        "confidence": float(result["confidence"]),
    }
