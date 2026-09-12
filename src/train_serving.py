from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.artifacts import atomic_joblib_dump, atomic_json_dump, atomic_xgb_save
from src.calibration import apply_temperature, fit_temperature
from src.config import (
    CALIBRATION_PATH,
    JSON_METADATA_PATH,
    JSON_MODEL_PATH,
    STACKER_PATH,
    VOICE_MODEL_PATH,
)
from src.data import audio_path_for, dataset_fingerprint, load_split, validate_audio_files
from src.features import TIMING_FEATURES, VOICE_FEATURES, extract_features
from src.fusion import fit_stacker


RANDOM_STATE = 42
FOLDS = 5
TIMING_PARAMS = {
    "n_estimators": 50,
    "max_depth": 2,
    "learning_rate": 0.05,
    "min_child_weight": 6,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_lambda": 3.0,
    "reg_alpha": 0.5,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
}
VOICE_PARAMS = {
    "n_estimators": 40,
    "max_depth": 2,
    "learning_rate": 0.05,
    "min_child_weight": 8,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_lambda": 4.0,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
}


def build_dataset(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if set(rows["split"]) != {"train"}:
        raise ValueError("Training accepts train rows only.")
    timing_rows = []
    voice_rows = []
    labels = []
    for row in rows.itertuples(index=False):
        timing, voice = extract_features(audio_path_for(str(row.anon_id)))
        timing_rows.append(timing)
        voice_rows.append(voice)
        labels.append(1 if row.label == "synthetic" else 0)
        print(f"Extracted train features: {row.anon_id}")
    return (
        pd.DataFrame(timing_rows, columns=TIMING_FEATURES),
        pd.DataFrame(voice_rows, columns=VOICE_FEATURES),
        np.asarray(labels, dtype=int),
    )


def out_of_fold_proba(model_params: dict, features: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    class_counts = np.bincount(labels)
    if len(class_counts) < 2 or class_counts.min() < FOLDS:
        raise ValueError(f"Each class needs at least {FOLDS} training rows.")
    probabilities = np.zeros(len(labels), dtype=float)
    folds = StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    for train_indices, holdout_indices in folds.split(features, labels):
        model = XGBClassifier(**model_params)
        model.fit(features.iloc[train_indices], labels[train_indices])
        probabilities[holdout_indices] = model.predict_proba(
            features.iloc[holdout_indices]
        )[:, 1]
    return probabilities


def train() -> dict:
    rows = load_split("train")
    validate_audio_files(rows)
    timing_features, voice_features, labels = build_dataset(rows)

    timing_oof = out_of_fold_proba(TIMING_PARAMS, timing_features, labels)
    voice_oof = out_of_fold_proba(VOICE_PARAMS, voice_features, labels)
    temperature = fit_temperature(timing_oof, labels)
    calibrated_timing_oof = np.asarray(
        [apply_temperature(value, temperature) for value in timing_oof]
    )
    stacker = fit_stacker(calibrated_timing_oof, voice_oof, labels)

    timing_model = XGBClassifier(**TIMING_PARAMS)
    timing_model.fit(timing_features, labels)
    voice_model = XGBClassifier(**VOICE_PARAMS)
    voice_model.fit(voice_features, labels)

    fingerprint = dataset_fingerprint(rows)
    metadata = {
        "schema_version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_on": "train_only",
        "training_rows": len(rows),
        "source_fingerprint": fingerprint,
        "random_state": RANDOM_STATE,
        "folds": FOLDS,
        "turn_source": "energy_vad_from_wav",
        "timing_features": TIMING_FEATURES,
        "voice_features": VOICE_FEATURES,
        "class_mapping": {"0": "human", "1": "synthetic"},
    }
    calibration = {
        "temperature": float(temperature),
        "trained_on": "train_oof_only",
    }

    atomic_joblib_dump(timing_model, JSON_MODEL_PATH)
    atomic_xgb_save(voice_model, VOICE_MODEL_PATH)
    atomic_joblib_dump(stacker, STACKER_PATH)
    atomic_json_dump(calibration, CALIBRATION_PATH)
    atomic_json_dump(metadata, JSON_METADATA_PATH)
    print(f"Saved clean train-only artifacts; fingerprint={fingerprint}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train production models on train only.")
    parser.parse_args()
    train()


if __name__ == "__main__":
    main()
