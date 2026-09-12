"""Run the detection steps and fuse their scores.

Order: one pass of feature extraction (turns, timing, voice), the timing head,
the voice head, then their fusion. The transcript (STT) head is consulted only
when the fused acoustic decision is uncertain and there is time budget left; it
nudges the fused score rather than replacing it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.config import (
    ELEVENLABS_API_KEY,
    REQUEST_BUDGET_S,
    STT_ENABLED,
    STT_TIMEOUT_S,
    STT_UNCERTAINTY_BAND,
)
from src.features import CallAudio, analyze_audio, load_audio
from src.models import fuse_probabilities, logit, sigmoid

# Weight of the transcript head's logit when it is blended into the fused score.
STT_WEIGHT = 0.5
# Minimum remaining budget needed before we even try the STT stage.
STT_MIN_REMAINING_S = 4.0


@dataclass
class StageResult:
    stage: str
    is_synthetic: bool
    p_synthetic: float
    certainty_pct: float
    features: dict = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        self.p_synthetic = float(self.p_synthetic)
        if not 0.0 <= self.p_synthetic <= 1.0:
            raise ValueError("p_synthetic must be between 0 and 1.")
        self.is_synthetic = label_from_probability(self.p_synthetic)
        self.certainty_pct = certainty_from_probability(self.p_synthetic)

    @classmethod
    def failed(cls, stage: str, error: str) -> "StageResult":
        return cls(
            stage=stage,
            is_synthetic=False,
            p_synthetic=0.5,
            certainty_pct=50.0,
            features={},
            error=error,
        )


def certainty_from_probability(p_synthetic: float) -> float:
    return max(p_synthetic, 1.0 - p_synthetic) * 100.0


def label_from_probability(p_synthetic: float) -> bool:
    return p_synthetic >= 0.5


def heads_disagree(timing_result: StageResult, voice_result: StageResult) -> bool:
    if timing_result.error or voice_result.error:
        return False
    return timing_result.is_synthetic != voice_result.is_synthetic


def fuse(timing_result: StageResult, voice_result: StageResult, stacker=None) -> StageResult:
    try:
        p = fuse_probabilities(timing_result.p_synthetic, voice_result.p_synthetic, stacker=stacker)
    except FileNotFoundError:
        return StageResult.failed("fusion", "stacker_missing")
    except Exception as exc:
        return StageResult.failed("fusion", f"fusion_failed:{exc}")

    return StageResult(
        stage="fusion",
        is_synthetic=label_from_probability(p),
        p_synthetic=p,
        certainty_pct=certainty_from_probability(p),
        features={
            "p_timing": timing_result.p_synthetic,
            "p_voice": voice_result.p_synthetic,
            "disagree": heads_disagree(timing_result, voice_result),
        },
    )


def blend_with_stt(fused: StageResult, stt: StageResult) -> StageResult:
    z = float(logit(fused.p_synthetic)) + STT_WEIGHT * float(logit(stt.p_synthetic))
    p = float(sigmoid(z))
    return StageResult(
        stage="fusion+stt",
        is_synthetic=label_from_probability(p),
        p_synthetic=p,
        certainty_pct=certainty_from_probability(p),
        features={"p_fused": fused.p_synthetic, "p_stt": stt.p_synthetic, **stt.features},
    )


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


def _stt_is_worth_it(fused: StageResult, started_at: float) -> bool:
    if not (STT_ENABLED and ELEVENLABS_API_KEY):
        return False
    if abs(fused.p_synthetic - 0.5) > STT_UNCERTAINTY_BAND:
        return False
    remaining = REQUEST_BUDGET_S - (time.perf_counter() - started_at)
    return remaining >= STT_MIN_REMAINING_S


def detect_audio(audio: CallAudio, audio_path: str | None = None, started_at: float | None = None) -> dict:
    from src.stt import run as run_stt
    from src.timing import score_features as score_timing
    from src.voice import score_features as score_voice

    started_at = time.perf_counter() if started_at is None else started_at
    stages: list[StageResult] = []

    try:
        analysis = analyze_audio(audio)
    except Exception as exc:
        stages.append(StageResult.failed("features", f"features_failed:{exc}"))
        return finalize(None, stages)

    timing_result = score_timing(analysis.timing)
    stages.append(timing_result)
    voice_result = score_voice(analysis.voice)
    stages.append(voice_result)

    good_heads = [r for r in (timing_result, voice_result) if r.error is None]
    if not good_heads:
        return finalize(None, stages)
    if len(good_heads) == 1:
        # One head failed: answer with the other, no fusion possible.
        return finalize(good_heads[0], stages)

    fused_result = fuse(timing_result, voice_result)
    stages.append(fused_result)
    if fused_result.error is not None:
        # Fall back to a plain average of the two heads.
        p = 0.5 * (timing_result.p_synthetic + voice_result.p_synthetic)
        fused_result = StageResult("fusion_avg", label_from_probability(p), p, 0.0)
        stages.append(fused_result)

    if not _stt_is_worth_it(fused_result, started_at) or audio_path is None:
        return finalize(fused_result, stages)

    remaining = REQUEST_BUDGET_S - (time.perf_counter() - started_at)
    stt_result = run_stt(audio_path, analysis.turns, timeout_s=min(STT_TIMEOUT_S, remaining))
    stages.append(stt_result)
    if stt_result.error is None:
        blended = blend_with_stt(fused_result, stt_result)
        stages.append(blended)
        return finalize(blended, stages)
    return finalize(fused_result, stages)


def detect_call(audio_path: str) -> dict:
    started_at = time.perf_counter()
    try:
        audio = load_audio(audio_path)
    except Exception as exc:
        return finalize(None, [StageResult.failed("load", f"load_failed:{exc}")])
    return detect_audio(audio, audio_path=audio_path, started_at=started_at)
