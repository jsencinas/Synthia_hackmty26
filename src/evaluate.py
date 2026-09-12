from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)

from src.artifacts import (
    artifact_hashes,
    load_metadata,
    load_stacker,
    load_timing_model,
    load_voice_model,
    require_artifacts,
)
from src.calibration import apply_temperature, load_temperature
from src.data import audio_path_for, load_split, validate_audio_files
from src.features import TIMING_FEATURES, VOICE_FEATURES, extract_features
from src.fusion import stack_features


def build_validation_dataset(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if set(rows["split"]) != {"val"}:
        raise ValueError("Evaluation accepts VAL rows only.")
    timing_rows = []
    voice_rows = []
    labels = []
    for row in rows.itertuples(index=False):
        timing, voice = extract_features(audio_path_for(str(row.anon_id)))
        timing_rows.append(timing)
        voice_rows.append(voice)
        labels.append(1 if row.label == "synthetic" else 0)
    return (
        pd.DataFrame(timing_rows, columns=TIMING_FEATURES),
        pd.DataFrame(voice_rows, columns=VOICE_FEATURES),
        np.asarray(labels, dtype=int),
    )


def evaluate() -> dict:
    require_artifacts()
    hashes_before = artifact_hashes()
    metadata = load_metadata()
    if metadata.get("trained_on") != "train_only":
        raise ValueError("Refusing to evaluate an artifact without train-only provenance.")

    rows = load_split("val")
    validate_audio_files(rows)
    timing_features, voice_features, labels = build_validation_dataset(rows)
    timing_model = load_timing_model()
    voice_model = load_voice_model()
    stacker = load_stacker()
    temperature = load_temperature()

    timing_probability = np.asarray(
        [
            apply_temperature(value, temperature)
            for value in timing_model.predict_proba(timing_features)[:, 1]
        ]
    )
    voice_probability = voice_model.predict_proba(voice_features)[:, 1]
    fused_probability = stacker.predict_proba(
        [
            stack_features(timing, voice)
            for timing, voice in zip(timing_probability, voice_probability)
        ]
    )[:, 1]
    predictions = (fused_probability >= 0.5).astype(int)
    metrics = {
        "split": "val",
        "rows": len(rows),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, fused_probability)),
        "brier": float(brier_score_loss(labels, fused_probability)),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(labels, predictions, target_names=["human", "synthetic"]))
    if artifact_hashes() != hashes_before:
        raise RuntimeError("Evaluation modified model artifacts.")
    return metrics


if __name__ == "__main__":
    evaluate()
