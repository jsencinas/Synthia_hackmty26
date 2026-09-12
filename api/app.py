"""POST /detect: human-vs-synthetic caller detector.

Design goals for judging: always answer HTTP 200 with a boolean when the body
is a request at all, warm every model and JIT path at start-up so the first
call is as fast as the rest, and never exceed the per-call time budget.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ELEVENLABS_API_KEY, STT_ENABLED  # noqa: E402
from src.detect import detect_audio  # noqa: E402
from src.features import CallAudio, load_audio  # noqa: E402
from src.models import warm_up  # noqa: E402

logger = logging.getLogger("detector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FALLBACK = {"is_synthetic": False, "confidence": 0.5}
STATE = {"ready": False, "artifact_error": None}


def _warm_inference() -> None:
    """Run one fake call through the pipeline to trigger numba/librosa JIT."""
    rng = np.random.default_rng(0)
    sr = 8000
    seconds = 12
    audio = np.zeros((sr * seconds, 2), dtype=np.float32)
    # a few bursts of noise on each channel so the VAD finds "turns"
    for ch, offsets in ((1, (0.5, 5.0)), (0, (2.5, 8.0))):
        for start in offsets:
            s, e = int(start * sr), int((start + 1.5) * sr)
            audio[s:e, ch] = (rng.standard_normal(e - s) * 0.1).astype(np.float32)
    detect_audio(CallAudio(samples=audio, sample_rate=sr))


@asynccontextmanager
async def lifespan(_: FastAPI):
    started = time.perf_counter()
    try:
        warm_up()
        _warm_inference()
        STATE["ready"] = True
        logger.info("models loaded and warmed in %.2fs", time.perf_counter() - started)
    except Exception as exc:  # keep serving; every answer will be the fallback
        STATE["artifact_error"] = str(exc)
        logger.error("start-up warm-up failed: %s", exc)
    yield


app = FastAPI(title="HackMTY caller detector", lifespan=lifespan)


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    call_id: str | None = None
    audio: str | None = None
    wav_base64: str | None = None
    audio_base64: str | None = None
    sample_rate: int | None = None
    channels: int | None = None


def _decode_wav(request: DetectRequest) -> bytes | None:
    payload = request.audio_base64 or request.wav_base64 or request.audio
    if not payload:
        return None
    try:
        return base64.b64decode(payload, validate=True)
    except Exception:
        try:
            return base64.b64decode(payload)
        except Exception:
            return None


def _stt_possible() -> bool:
    return bool(STT_ENABLED and ELEVENLABS_API_KEY)


@app.get("/health")
def health() -> dict:
    return {"ready": STATE["ready"], "error": STATE["artifact_error"]}


@app.post("/detect")
def detect(request: DetectRequest) -> dict:
    started = time.perf_counter()
    call_id = request.call_id or "-"

    wav_bytes = _decode_wav(request)
    if not wav_bytes or len(wav_bytes) < 44:
        logger.warning("%s: missing or invalid base64 payload; answering fallback", call_id)
        return dict(FALLBACK)

    try:
        audio = load_audio(io.BytesIO(wav_bytes))
    except Exception as exc:
        logger.warning("%s: undecodable WAV (%s); answering fallback", call_id, exc)
        return dict(FALLBACK)

    if not STATE["ready"]:
        logger.error("%s: models not ready (%s); answering fallback", call_id, STATE["artifact_error"])
        return dict(FALLBACK)

    audio_path = None
    try:
        if _stt_possible():
            # The transcript head needs a file on disk; only pay for it if STT can run.
            fd, audio_path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as handle:
                handle.write(wav_bytes)
        result = detect_audio(audio, audio_path=audio_path, started_at=started)
    except Exception as exc:
        logger.exception("%s: detection crashed: %s", call_id, exc)
        return dict(FALLBACK)
    finally:
        if audio_path:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    elapsed = time.perf_counter() - started
    logger.info(
        "%s: is_synthetic=%s confidence=%.3f stopped_at=%s error=%s %.2fs",
        call_id,
        result["is_synthetic"],
        result["confidence"],
        result.get("stopped_at"),
        result.get("error"),
        elapsed,
    )
    return {
        "is_synthetic": bool(result["is_synthetic"]),
        "confidence": float(min(max(result["confidence"], 0.0), 1.0)),
    }
