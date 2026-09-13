"""Step 2: score caller voice acoustics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.detect import StageResult, certainty_from_probability, label_from_probability
from src.features import VOICE_FEATURES, voice_features
from src.models import load_voice_model


def score_features(features: dict) -> StageResult:
    try:
        model = load_voice_model()
        probability = float(
            model.predict_proba(pd.DataFrame([features], columns=VOICE_FEATURES))[0, 1]
        )
        return StageResult(
            stage="voice",
            is_synthetic=label_from_probability(probability),
            p_synthetic=probability,
            certainty_pct=certainty_from_probability(probability),
            features=features,
        )
    except FileNotFoundError:
        return StageResult.failed("voice", "voice_model_missing")
    except Exception as exc:
        return StageResult.failed("voice", f"voice_failed:{exc}")


def run(audio_path: str, turns_payload: dict | None) -> StageResult:
    if not audio_path or not Path(audio_path).is_file():
        return StageResult.failed("voice", "voice_audio_missing")
    if not turns_payload or not turns_payload.get("turns"):
        return StageResult.failed("voice", "voice_turns_missing")
    try:
        features = voice_features(audio_path, turns_payload)
    except Exception as exc:
        return StageResult.failed("voice", f"voice_failed:{exc}")
    return score_features(features)
