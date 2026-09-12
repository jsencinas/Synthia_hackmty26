from __future__ import annotations

from src.config import ESCALATION_THRESHOLD_PCT
from src.json_stage import run as run_json
from src.transcript_stage import run as run_stt
from src.turns_from_wav import write_temp_json
from src.types import StageResult
from src.voice_stage import run as run_voice


def is_certain(result: StageResult, threshold_pct: float = ESCALATION_THRESHOLD_PCT) -> bool:
    return result.error is None and result.certainty_pct >= threshold_pct


def finalize(result: StageResult | None, stages: list[StageResult] | None = None) -> dict:
    if result is None:
        return {
            "is_synthetic": False,
            "confidence": 0.5,
            "stopped_at": None,
            "error": "all_stages_failed",
            "stages": [],
        }
    return {
        "is_synthetic": bool(result.p_synthetic >= 0.5),
        "confidence": float(result.p_synthetic),
        "stopped_at": result.stage,
        "certainty_pct": result.certainty_pct,
        "error": result.error,
        "stages": [
            {
                "stage": item.stage,
                "p_synthetic": item.p_synthetic,
                "certainty_pct": item.certainty_pct,
                "error": item.error,
            }
            for item in (stages or [])
        ],
    }


def detect_call(audio_path: str, turns_path: str | None = None) -> dict:
    generated_turns = False
    if turns_path is None:
        try:
            turns_path = write_temp_json(audio_path)
            generated_turns = True
        except Exception:
            turns_path = None

    stages: list[StageResult] = []
    last_good: StageResult | None = None

    json_result = run_json(turns_path)
    stages.append(json_result)
    if json_result.error is None:
        last_good = json_result
        if is_certain(json_result):
            return finalize(json_result, stages)

    if audio_path and turns_path:
        voice_result = run_voice(audio_path, turns_path)
        stages.append(voice_result)
        if voice_result.error is None:
            last_good = voice_result
            if is_certain(voice_result):
                return finalize(voice_result, stages)
    else:
        voice_result = StageResult(
            stage="voice",
            is_synthetic=False,
            p_synthetic=0.5,
            certainty_pct=0.0,
            features={},
            error="voice_turns_missing" if not turns_path else "voice_audio_missing",
        )
        stages.append(voice_result)

    stt_result = run_stt(audio_path, turns_path)
    stages.append(stt_result)
    if stt_result.error is None:
        last_good = stt_result

    _ = generated_turns
    return finalize(last_good, stages)
