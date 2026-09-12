"""Fit timing, voice, and fusion models on the train split only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import (
    CALIBRATION_PATH,
    METADATA_PATH,
    STACKER_PATH,
    TIMING_MODEL_PATH,
    VOICE_MODEL_PATH,
)
from src.data import audio_path_for, dataset_fingerprint, load_split, validate_audio_files
from src.features import TIMING_FEATURES, VOICE_FEATURES, analyze_call
from src.models import (
    RANDOM_STATE,
    TIMING_XGB_PARAMS,
    VOICE_XGB_PARAMS,
    apply_temperature,
    atomic_joblib_dump,
    atomic_json_dump,
    build_head,
    fit_stacker,
    fit_temperature,
    stack_features,
)

FOLDS = 5


def build_dataset(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if set(rows["split"]) != {"train"}:
        raise ValueError("Training accepts train rows only.")
    timing_rows = []
    voice_rows = []
    labels = []
    for index, row in enumerate(rows.itertuples(index=False), start=1):
        analysis = analyze_call(audio_path_for(str(row.anon_id)))
        timing_rows.append(analysis.timing)
        voice_rows.append(analysis.voice)
        labels.append(1 if row.label == "synthetic" else 0)
        print(f"[{index}/{len(rows)}] extracted features: {row.anon_id}")
    return (
        pd.DataFrame(timing_rows, columns=TIMING_FEATURES),
        pd.DataFrame(voice_rows, columns=VOICE_FEATURES),
        np.asarray(labels, dtype=int),
    )


def out_of_fold_proba(xgb_params: dict, features: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    class_counts = np.bincount(labels)
    if len(class_counts) < 2 or class_counts.min() < FOLDS:
        raise ValueError(f"Each class needs at least {FOLDS} training rows.")
    probabilities = np.zeros(len(labels), dtype=float)
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for train_indices, holdout_indices in folds.split(features, labels):
        model = build_head(xgb_params, labels[train_indices])
        model.fit(features.iloc[train_indices], labels[train_indices])
        probabilities[holdout_indices] = model.predict_proba(features.iloc[holdout_indices])[:, 1]
    return probabilities


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, probabilities >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def train() -> dict:
    rows = load_split("train")
    validate_audio_files(rows)
    timing_features, voice_features, labels = build_dataset(rows)

    # Out-of-fold head probabilities so the stacker never sees fitted-on scores.
    timing_oof = out_of_fold_proba(TIMING_XGB_PARAMS, timing_features, labels)
    voice_oof = out_of_fold_proba(VOICE_XGB_PARAMS, voice_features, labels)
    stacker = fit_stacker(timing_oof, voice_oof, labels)

    # Temperature for the fused output, fitted on cross-validated stacker scores.
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fused_oof = cross_val_predict(
        fit_stacker(timing_oof, voice_oof, labels),
        stack_features(timing_oof, voice_oof),
        labels,
        cv=folds,
        method="predict_proba",
    )[:, 1]
    temperature = fit_temperature(fused_oof, labels)
    calibrated_oof = apply_temperature(fused_oof, temperature)

    timing_model = build_head(TIMING_XGB_PARAMS, labels)
    timing_model.fit(timing_features, labels)
    voice_model = build_head(VOICE_XGB_PARAMS, labels)
    voice_model.fit(voice_features, labels)

    fingerprint = dataset_fingerprint(rows)
    metadata = {
        "schema_version": 2,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_on": "train_only",
        "training_rows": int(len(rows)),
        "class_counts": {"human": int((labels == 0).sum()), "synthetic": int((labels == 1).sum())},
        "source_fingerprint": fingerprint,
        "random_state": RANDOM_STATE,
        "folds": FOLDS,
        "turn_source": "adaptive_energy_vad_from_wav",
        "head_model": "soft_vote(xgboost, standardized_logistic_regression)",
        "fusion": "logistic_regression_on_head_logits + temperature",
        "timing_features": TIMING_FEATURES,
        "voice_features": VOICE_FEATURES,
        "class_mapping": {"0": "human", "1": "synthetic"},
        "train_oof_metrics": {
            "timing": _metrics(labels, timing_oof),
            "voice": _metrics(labels, voice_oof),
            "fused": _metrics(labels, calibrated_oof),
        },
    }
    calibration = {
        "temperature": float(temperature),
        "applies_to": "fused_probability",
        "trained_on": "train_oof_only",
    }

    atomic_joblib_dump(timing_model, TIMING_MODEL_PATH)
    atomic_joblib_dump(voice_model, VOICE_MODEL_PATH)
    atomic_joblib_dump(stacker, STACKER_PATH)
    atomic_json_dump(calibration, CALIBRATION_PATH)
    atomic_json_dump(metadata, METADATA_PATH)
    print("Train OOF metrics:", metadata["train_oof_metrics"])
    print(f"Saved clean train-only artifacts; fingerprint={fingerprint}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train production models on train only.")
    parser.parse_args()
    train()


if __name__ == "__main__":
    main()
