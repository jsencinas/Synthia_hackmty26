from __future__ import annotations

from src.fusion import fuse, heads_disagree
from src.json_stage import run as run_timing
from src.transcript_stage import run as run_stt
from src.turns_from_wav import extract_turns
from src.types import StageResult
from src.voice_stage import run as run_voice


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
        "is_synthetic": result.is_synthetic,
        "confidence": max(result.p_synthetic, 1.0 - result.p_synthetic),
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


def detect_call(audio_path: str) -> dict:
    try:
        turns_payload = extract_turns(audio_path)
        if not turns_payload["turns"]:
            turns_payload = None
    except Exception:
        turns_payload = None

    stages: list[StageResult] = []
    last_good: StageResult | None = None

    timing_result = run_timing(turns_payload)
    stages.append(timing_result)
    if timing_result.error is None:
        last_good = timing_result

    voice_result = run_voice(audio_path, turns_payload)
    stages.append(voice_result)
    if voice_result.error is None:
        last_good = voice_result

    fused_result: StageResult | None = None
    if timing_result.error is None and voice_result.error is None:
        fused_result = fuse(timing_result, voice_result)
        stages.append(fused_result)
        last_good = fused_result
        if not heads_disagree(timing_result, voice_result):
            return finalize(fused_result, stages)

    stt_result = run_stt(audio_path, turns_payload)
    stages.append(stt_result)
    if stt_result.error is None:
        return finalize(stt_result, stages)
    if fused_result is not None:
        return finalize(fused_result, stages)
    return finalize(last_good, stages)
