"""Step 1: score caller/agent turn timing."""

from __future__ import annotations

import pandas as pd

from src.detect import StageResult, certainty_from_probability, label_from_probability
from src.features import TIMING_FEATURES, timing_features
from src.models import load_timing_model


def score_features(features: dict) -> StageResult:
    try:
        model = load_timing_model()
        probability = float(
            model.predict_proba(pd.DataFrame([features], columns=TIMING_FEATURES))[0, 1]
        )
        return StageResult(
            stage="timing",
            is_synthetic=label_from_probability(probability),
            p_synthetic=probability,
            certainty_pct=certainty_from_probability(probability),
            features=features,
        )
    except FileNotFoundError:
        return StageResult.failed("timing", "timing_model_missing")
    except Exception as exc:
        return StageResult.failed("timing", f"timing_failed:{exc}")


def run(turns_payload: dict | None, duration: float | None = None) -> StageResult:
    if not turns_payload or not turns_payload.get("turns"):
        return StageResult.failed("timing", "timing_turns_missing")
    try:
        features = timing_features(turns_payload, duration=duration)
    except Exception as exc:
        return StageResult.failed("timing", f"timing_failed:{exc}")
    return score_features(features)
