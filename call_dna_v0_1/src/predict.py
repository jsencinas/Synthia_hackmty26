from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS_PATH, MODEL_PATH
from .utils import load_json


def _load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train.py` first."
        )
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Feature columns not found at {FEATURE_COLUMNS_PATH}."
        )

    model = joblib.load(MODEL_PATH)
    feature_columns = load_json(FEATURE_COLUMNS_PATH)
    return model, feature_columns


def predict_from_features(features: dict[str, Any]) -> dict[str, Any]:
    """
    Predict human/synthetic from a feature dictionary.

    Missing features are represented as NaN. XGBoost supports missing values,
    but the API should normally provide the complete feature vector.
    """
    model, feature_columns = _load_artifacts()

    row = {
        feature: features.get(feature, np.nan)
        for feature in feature_columns
    }
    X = pd.DataFrame([row], columns=feature_columns)
    probabilities = model.predict_proba(X)[0]

    synthetic_probability = float(probabilities[1])
    human_probability = float(probabilities[0])

    if synthetic_probability >= human_probability:
        classification = "synthetic"
        confidence = synthetic_probability
    else:
        classification = "human"
        confidence = human_probability

    return {
        "classification": classification,
        "is_synthetic": classification == "synthetic",
        "confidence": confidence,
        "probabilities": {
            "human": human_probability,
            "synthetic": synthetic_probability,
        },
    }


if __name__ == "__main__":
    from .config import FEATURES_PATH

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} does not exist. Run `python train.py` first."
        )

    df = pd.read_csv(FEATURES_PATH)
    sample = df.iloc[0].to_dict()
    result = predict_from_features(sample)
    print(result)
