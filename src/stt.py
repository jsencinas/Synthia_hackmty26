"""Step 3: transcribe with ElevenLabs and score the transcript.

Only used as a tie-breaker when the acoustic heads are uncertain. Every call is
bounded by an explicit timeout so the request never blows the judge's budget.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from src.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_KEYTERMS,
    ELEVENLABS_LANGUAGE,
    ELEVENLABS_MODEL_ID,
    STT_TIMEOUT_S,
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
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _save_cache(audio_path: str, payload: dict) -> None:
    try:
        path = _cache_path(audio_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    except OSError:
        pass


def _to_dict(result) -> dict:
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"text": getattr(result, "text", ""), "words": getattr(result, "words", [])}


def _normalize_response(result) -> dict:
    """Flatten single- or multi-channel Scribe responses to one shape.

    Returns ``{"text", "words", "channels": {0: {"text", "words"}, 1: {...}}}``.
    With ``use_multi_channel=True`` ElevenLabs returns ``{"transcripts": [...]}``,
    one entry per channel with a ``channel_index``.
    """
    data = _to_dict(result)
    channels: dict[int, dict] = {}
    if isinstance(data.get("transcripts"), list):
        for entry in data["transcripts"]:
            entry = _to_dict(entry)
            index = entry.get("channel_index")
            try:
                index = int(index)
            except (TypeError, ValueError):
                index = len(channels)
            channels[index] = {
                "text": entry.get("text") or "",
                "words": [_to_dict(w) for w in (entry.get("words") or [])],
            }
        text = " ".join(channels[k]["text"] for k in sorted(channels))
        words = [w for k in sorted(channels) for w in channels[k]["words"]]
    else:
        text = data.get("text") or ""
        words = [_to_dict(w) for w in (data.get("words") or [])]
    return {
        "text": text,
        "words": words,
        "channels": {str(k): v for k, v in channels.items()},
    }


def _call_scribe(audio_path: str, timeout_s: float):
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY, timeout=max(1.0, float(timeout_s)))
    with open(audio_path, "rb") as audio_file:
        return client.speech_to_text.convert(
            file=audio_file,
            model_id=ELEVENLABS_MODEL_ID,
            language_code=ELEVENLABS_LANGUAGE,
            timestamps_granularity="word",
            use_multi_channel=True,
            keyterms=ELEVENLABS_KEYTERMS,
        )


def transcribe(audio_path: str, timeout_s: float = STT_TIMEOUT_S) -> dict:
    if not ELEVENLABS_API_KEY:
        raise STTError("stt_unconfigured")
    if not audio_path or not os.path.exists(audio_path):
        raise STTError("stt_audio_missing")

    cached = _load_cache(audio_path)
    if cached and (cached.get("text") or cached.get("words")):
        return cached

    try:
        result = _call_scribe(audio_path, timeout_s)
    except Exception as exc:
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if status == 401:
            raise STTError("stt_failed", "unauthorized") from exc
        raise STTError("stt_failed", str(exc)) from exc

    payload = _normalize_response(result)
    if not payload.get("text") and not payload.get("words"):
        raise STTError("stt_failed", "empty transcript")
    _save_cache(audio_path, payload)
    return payload


# Cues observed on the dataset: the agent asks to repeat in *every* call, so the
# repeat cue by itself carries no information; hesitations/disfluencies on the
# caller side are the useful human signal. Numbers are transcribed as words, so
# a digit replay check is only meaningful when digits do appear.
REPEAT_BACK = ("repita", "repite", "confirme", "confirma", "digame de nuevo", "dígame de nuevo", "de nuevo")
HESITATION_RE = re.compile(
    r"\b(mande|c[oó]mo|perd[oó]n|no tengo|no s[eé]|eh+|mm+|ah+|em+|a ver)\b|(\w+)-- "
)
FILLER_RE = re.compile(r"\b(bueno|este|o sea|pues|digo)\b")
DIGIT_RE = re.compile(r"\d")


def _word_text(word: Any) -> str:
    if isinstance(word, dict):
        return str(word.get("text", ""))
    return str(getattr(word, "text", "") or "")


def _word_start(word: Any) -> float | None:
    value = word.get("start") if isinstance(word, dict) else getattr(word, "start", None)
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


def _split_channels(payload: dict, turns: list[dict]) -> tuple[str, str, str]:
    full_text = _normalize(payload.get("text") or "")
    words = payload.get("words") or []
    if not full_text and words:
        full_text = _normalize("".join(_word_text(word) for word in words))

    channels = payload.get("channels") or {}
    if "0" in channels or "1" in channels:
        caller_text = _normalize(channels.get("0", {}).get("text") or "")
        agent_text = _normalize(channels.get("1", {}).get("text") or "")
        return full_text, caller_text, agent_text

    caller_bits, agent_bits = [], []
    for word in words:
        text = _word_text(word)
        if not text.strip():
            continue
        channel = _channel_for_time(_word_start(word), turns)
        if channel == 0:
            caller_bits.append(text)
        elif channel == 1:
            agent_bits.append(text)
    return full_text, _normalize("".join(caller_bits)), _normalize("".join(agent_bits))


def score_transcript(payload: dict, turns: list[dict]) -> tuple[float, dict]:
    full_text, caller_text, agent_text = _split_channels(payload, turns)
    if not caller_text and not agent_text:
        caller_text = full_text

    repeat_cue = any(cue in agent_text for cue in REPEAT_BACK)
    hesitations = len(HESITATION_RE.findall(caller_text))
    fillers = len(FILLER_RE.findall(caller_text))
    caller_digits = "".join(DIGIT_RE.findall(caller_text))
    agent_digits = "".join(DIGIT_RE.findall(agent_text))
    exact_digit_replay = (
        bool(caller_digits)
        and bool(agent_digits)
        and repeat_cue
        and (caller_digits in agent_digits or agent_digits in caller_digits)
    )

    score = 0.55
    if exact_digit_replay:
        score += 0.15
    if hesitations:
        score -= min(0.3, 0.12 * hesitations)
    if fillers:
        score -= min(0.06, 0.03 * fillers)
    if not caller_text:
        score += 0.1

    p_synthetic = min(0.85, max(0.15, score))
    features = {
        "repeat_cue": repeat_cue,
        "hesitations": hesitations,
        "fillers": fillers,
        "exact_digit_replay": exact_digit_replay,
        "caller_chars": len(caller_text),
    }
    return p_synthetic, features


def run(audio_path: str, turns_payload: dict | None, timeout_s: float = STT_TIMEOUT_S) -> StageResult:
    try:
        payload = transcribe(audio_path, timeout_s=timeout_s)
        turns = (turns_payload or {}).get("turns", [])
        p_synthetic, features = score_transcript(payload, turns)
        return StageResult(
            stage="stt",
            is_synthetic=label_from_probability(p_synthetic),
            p_synthetic=p_synthetic,
            certainty_pct=certainty_from_probability(p_synthetic),
            features=features,
        )
    except STTError as exc:
        return StageResult.failed("stt", exc.code)
    except Exception:
        return StageResult.failed("stt", "stt_failed")
