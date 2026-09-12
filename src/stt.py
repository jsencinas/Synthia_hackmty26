"""Step 3: transcribe with ElevenLabs and score the transcript."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from src.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_KEYTERMS,
    ELEVENLABS_LANGUAGE,
    ELEVENLABS_MODEL_ID,
    TRANSCRIPT_CACHE_DIR,
)
from src.detect import StageResult, certainty_from_probability, label_from_probability


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


REPEAT_BACK = ("repita", "confirme", "digame de nuevo", "dígame de nuevo")
HESITATION = ("mande", "no tengo", "como", "cómo")
FILLERS = ("bueno", "este", "o sea")
DIGIT_RE = re.compile(r"\d")


def _word_text(word: Any) -> str:
    if isinstance(word, dict):
        return str(word.get("text", ""))
    return str(getattr(word, "text", "") or "")


def _word_start(word: Any) -> float | None:
    if isinstance(word, dict):
        value = word.get("start")
    else:
        value = getattr(word, "start", None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _channel_for_time(start: float | None, turns: list[dict]) -> int | None:
    if start is None:
        return None
    for turn in turns:
        if turn["start"] <= start <= turn["end"]:
            return int(turn["channel"])
    return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_transcript(payload: dict, turns: list[dict]) -> tuple[float, dict]:
    words = payload.get("words") or []
    full_text = _normalize(payload.get("text") or "")
    if not full_text and words:
        full_text = _normalize("".join(_word_text(word) for word in words))

    caller_bits = []
    agent_bits = []
    for word in words:
        text = _word_text(word)
        if not text.strip() or text.isspace():
            continue
        channel = _channel_for_time(_word_start(word), turns)
        if channel == 0:
            caller_bits.append(text)
        elif channel == 1:
            agent_bits.append(text)

    caller_text = _normalize("".join(caller_bits)) or full_text
    agent_text = _normalize("".join(agent_bits))

    repeat_cue = any(cue in agent_text or cue in full_text for cue in REPEAT_BACK)
    hesitation = any(cue in caller_text for cue in HESITATION)
    fillers = sum(caller_text.count(cue) for cue in FILLERS)
    caller_digits = "".join(DIGIT_RE.findall(caller_text))
    agent_digits = "".join(DIGIT_RE.findall(agent_text))
    exact_digit_replay = (
        bool(caller_digits)
        and bool(agent_digits)
        and caller_digits == agent_digits
        and repeat_cue
    )

    score = 0.45
    if exact_digit_replay:
        score += 0.25
    if repeat_cue and not hesitation:
        score += 0.10
    if hesitation:
        score -= 0.20
    if fillers:
        score -= min(0.15, 0.05 * fillers)
    if not caller_text:
        score += 0.10

    p_synthetic = min(0.95, max(0.05, score))
    features = {
        "repeat_cue": repeat_cue,
        "hesitation": hesitation,
        "fillers": fillers,
        "exact_digit_replay": exact_digit_replay,
        "caller_chars": len(caller_text),
    }
    return p_synthetic, features


def run(audio_path: str, turns_payload: dict | None) -> StageResult:
    try:
        payload = transcribe(audio_path)
        turns = (turns_payload or {}).get("turns", [])
        p_synthetic, features = score_transcript(payload, turns)
        if not payload.get("text") and not payload.get("words"):
            return StageResult(
                stage="stt",
                is_synthetic=False,
                p_synthetic=0.5,
                certainty_pct=0.0,
                features=features,
                error="stt_failed",
            )
        return StageResult(
            stage="stt",
            is_synthetic=label_from_probability(p_synthetic),
            p_synthetic=p_synthetic,
            certainty_pct=certainty_from_probability(p_synthetic),
            features=features,
        )
    except STTError as exc:
        return StageResult(
            stage="stt",
            is_synthetic=False,
            p_synthetic=0.5,
            certainty_pct=0.0,
            features={},
            error=exc.code,
        )
    except Exception:
        return StageResult(
            stage="stt",
            is_synthetic=False,
            p_synthetic=0.5,
            certainty_pct=0.0,
            features={},
            error="stt_failed",
        )
