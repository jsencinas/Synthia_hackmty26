"""Run the three detection steps and fuse their scores.

Order: timing, voice, fuse those two; call STT only if they disagree
or one of them failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.features import extract_turns
from src.models import load_stacker, stack_features


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
        if stacker is None:
            stacker = load_stacker()
        p = float(
            stacker.predict_proba(
                [stack_features(timing_result.p_synthetic, voice_result.p_synthetic)]
            )[0, 1]
        )
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
            "p_json": timing_result.p_synthetic,
            "p_voice": voice_result.p_synthetic,
            "disagree": heads_disagree(timing_result, voice_result),
        },
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


def detect_call(audio_path: str) -> dict:
    from src.stt import run as run_stt
    from src.timing import run as run_timing
    from src.voice import run as run_voice

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
        if fused_result.error is None:
            last_good = fused_result
            if not heads_disagree(timing_result, voice_result):
                return finalize(fused_result, stages)
        else:
            fused_result = None

    stt_result = run_stt(audio_path, turns_payload)
    stages.append(stt_result)
    if stt_result.error is None:
        return finalize(stt_result, stages)
    if fused_result is not None:
        return finalize(fused_result, stages)
    return finalize(last_good, stages)
