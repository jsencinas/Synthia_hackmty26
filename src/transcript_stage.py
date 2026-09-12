from __future__ import annotations

import re
from typing import Any

from src.stt_elevenlabs import STTError, transcribe
from src.types import StageResult, certainty_from_probability, label_from_probability


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
