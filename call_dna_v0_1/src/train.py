from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from .config import (
    FEATURE_COLUMNS_PATH,
    FEATURES_PATH,
    HUMAN_STATS_PATH,
    MANIFEST_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    TURNS_DIR,
    RANDOM_STATE,
)
from .feature_engineering import FEATURE_NAMES, build_dataset
from .utils import ensure_directories, save_json


def calcular_metricas(y_true, y_pred):
    """Calcula métricas a mano sin usar scikit-learn."""
    y_t = list(y_true)
    y_p = list(y_pred)
    
    tp = sum(1 for t, p in zip(y_t, y_p) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_t, y_p) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_t, y_p) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_t, y_p) if t == 1 and p == 0)
    
    total = len(y_t)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate metrics for the 'human' class (class 0)
    h_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    h_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    h_f1 = 2 * (h_precision * h_recall) / (h_precision + h_recall) if (h_precision + h_recall) > 0 else 0.0
    
    confusion_matrix = [[tn, fp], [fn, tp]]

    return accuracy, precision, recall, f1, h_precision, h_recall, h_f1, confusion_matrix

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return model features without identifiers, labels or split metadata."""
    X = df[FEATURE_NAMES].copy()
    return X.replace([np.inf, -np.inf], np.nan)


def _human_reference_stats(
    train_df: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    """Calculate human-only reference distributions using TRAIN only."""
    human = train_df[train_df["label"] == "human"]
    stats: dict[str, dict[str, float | int]] = {}

    for feature in FEATURE_NAMES:
        values = pd.to_numeric(human[feature], errors="coerce").dropna()
        stats[feature] = {
            "count": int(values.shape[0]),
            "mean": float(values.mean()) if not values.empty else 0.0,
            "std": float(values.std(ddof=0)) if not values.empty else 0.0,
            "min": float(values.min()) if not values.empty else 0.0,
            "max": float(values.max()) if not values.empty else 0.0,
        }

    return stats


def train_model() -> dict:
    ensure_directories()

    dataset = build_dataset(MANIFEST_PATH, TURNS_DIR)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(FEATURES_PATH, index=False)

    train_df = dataset[dataset["split"] == "train"].copy()
    val_df = dataset[dataset["split"] == "val"].copy()

    if train_df.empty:
        raise ValueError("No training rows found (split == 'train').")
    if val_df.empty:
        raise ValueError("No validation rows found (split == 'val').")

    X_train = _prepare_features(train_df)
    X_val = _prepare_features(val_df)

    y_train = train_df["label"].map({"human": 0, "synthetic": 1}).astype(int)
    y_val = val_df["label"].map({"human": 0, "synthetic": 1}).astype(int)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        objective="binary:logistic",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    acc, prec, rec, f1, h_prec, h_rec, h_f1, conf_matrix = calcular_metricas(y_val, predictions)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": None,  # ROC AUC calculation manually is complex, omitting for simplicity as discussed
        "human": {
            "precision": h_prec,
            "recall": h_rec,
            "f1": h_f1,
        },
        "synthetic": {
            "precision": prec,
            "recall": rec,
            "f1": f1,
        },
        "confusion_matrix": conf_matrix,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "model_version": "0.1.0",
        "feature_count": len(FEATURE_NAMES),
    }

    joblib.dump(model, MODEL_PATH)
    save_json(FEATURE_NAMES, FEATURE_COLUMNS_PATH)

    reference_stats = _human_reference_stats(train_df)
    save_json(reference_stats, HUMAN_STATS_PATH)
    save_json(metrics, METRICS_PATH)

    print(json.dumps(metrics, indent=2))
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Features saved to: {FEATURES_PATH}")
    return metrics


if __name__ == "__main__":
    train_model()