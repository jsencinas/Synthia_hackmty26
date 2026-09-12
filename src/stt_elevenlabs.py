from __future__ import annotations

import json
import os
import time
import hashlib
from pathlib import Path

from src.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_KEYTERMS,
    ELEVENLABS_LANGUAGE,
    ELEVENLABS_MODEL_ID,
    TRANSCRIPT_CACHE_DIR,
)


class STTError(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


def _cache_path(audio_path: str) -> Path:
    digest = hashlib.sha256()
    with open(audio_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return Path(TRANSCRIPT_CACHE_DIR) / f"{digest.hexdigest()}.json"


def _load_cache(audio_path: str) -> dict | None:
    path = _cache_path(audio_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_cache(audio_path: str, payload: dict) -> None:
    path = _cache_path(audio_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _normalize_response(result) -> dict:
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    text = getattr(result, "text", "")
    words = getattr(result, "words", [])
    return {"text": text, "words": words}


def _call_scribe(audio_path: str):
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY, timeout=60)
    with open(audio_path, "rb") as audio_file:
        return client.speech_to_text.convert(
            file=audio_file,
            model_id=ELEVENLABS_MODEL_ID,
            language_code=ELEVENLABS_LANGUAGE,
            timestamps_granularity="word",
            use_multi_channel=True,
            keyterms=ELEVENLABS_KEYTERMS,
        )


def transcribe(audio_path: str) -> dict:
    if not ELEVENLABS_API_KEY:
        raise STTError("stt_unconfigured")
    if not audio_path or not os.path.exists(audio_path):
        raise STTError("stt_audio_missing")

    cached = _load_cache(audio_path)
    if cached:
        return cached

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            result = _call_scribe(audio_path)
            payload = _normalize_response(result)
            if not payload.get("text") and not payload.get("words"):
                raise STTError("stt_failed", "empty transcript")
            _save_cache(audio_path, payload)
            return payload
        except STTError:
            raise
        except Exception as exc:
            last_error = exc
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            if status in {401}:
                raise STTError("stt_failed", "unauthorized") from exc
            if status in {429, 500, 502, 503} and attempt == 0:
                time.sleep(1)
                continue
            raise STTError("stt_failed", str(exc)) from exc

    raise STTError("stt_failed", str(last_error) if last_error else "unknown")
