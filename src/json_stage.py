from __future__ import annotations

import pandas as pd

from src.artifacts import load_timing_model
from src.calibration import apply_temperature, load_temperature
from src.features import TIMING_FEATURES, timing_features
from src.types import StageResult, certainty_from_probability, label_from_probability


def run(turns_payload: dict | None) -> StageResult:
    if not turns_payload:
        return StageResult.failed("timing", "timing_turns_missing")
    try:
        features = timing_features(turns_payload)
        model = load_timing_model()
        raw_probability = float(
            model.predict_proba(pd.DataFrame([features], columns=TIMING_FEATURES))[0, 1]
        )
        probability = apply_temperature(raw_probability, load_temperature())
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
