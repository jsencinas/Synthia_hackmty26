"""Trained artifacts: build/load/save heads, fuse them, temperature-calibrate."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    ARTIFACT_DIR,
    CALIBRATION_PATH,
    METADATA_PATH,
    STACKER_PATH,
    TIMING_MODEL_PATH,
    VOICE_MODEL_PATH,
)

RANDOM_STATE = 42

# Each head is a soft-voting ensemble of a shallow gradient-boosted model and a
# standardised logistic regression: the trees capture thresholds/interactions,
# the linear model extrapolates smoothly to unseen callers and voices.
TIMING_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_lambda": 2.0,
    "reg_alpha": 0.2,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "n_jobs": 4,
}
VOICE_XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 2,
    "learning_rate": 0.02,
    "min_child_weight": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.4,
    "reg_lambda": 2.0,
    "reg_alpha": 0.2,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "n_jobs": 4,
}
LR_C = 0.1


def build_head(xgb_params: dict, labels: np.ndarray) -> VotingClassifier:
    labels = np.asarray(labels)
    negatives = max(int((labels == 0).sum()), 1)
    positives = max(int((labels == 1).sum()), 1)
    xgb = XGBClassifier(scale_pos_weight=negatives / positives, **xgb_params)
    linear = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=LR_C,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return VotingClassifier([("xgb", xgb), ("lr", linear)], voting="soft")


def artifact_paths() -> list[Path]:
    return [
        Path(TIMING_MODEL_PATH),
        Path(VOICE_MODEL_PATH),
        Path(STACKER_PATH),
        Path(CALIBRATION_PATH),
        Path(METADATA_PATH),
    ]


def artifact_hashes() -> dict[str, str]:
    hashes = {}
    for path in artifact_paths():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    return hashes


def _temporary_path(destination: str | Path, suffix: str = ".tmp") -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=suffix,
        dir=destination.parent,
    )
    os.close(fd)
    return Path(name)


def atomic_joblib_dump(value, destination: str | Path) -> None:
    temporary = _temporary_path(destination)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_dump(payload: dict, destination: str | Path) -> None:
    temporary = _temporary_path(destination, suffix=".json")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def load_timing_model():
    return joblib.load(TIMING_MODEL_PATH)


@lru_cache(maxsize=1)
def load_voice_model():
    return joblib.load(VOICE_MODEL_PATH)


@lru_cache(maxsize=1)
def load_stacker():
    return joblib.load(STACKER_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    return json.loads(Path(METADATA_PATH).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_temperature() -> float:
    calibration_path = Path(CALIBRATION_PATH)
    if not calibration_path.exists():
        return 1.0
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    return float(data.get("temperature", 1.0))


def clear_model_cache() -> None:
    for loader in (load_timing_model, load_voice_model, load_stacker, load_metadata, load_temperature):
        loader.cache_clear()


def require_artifacts() -> None:
    required = artifact_paths()
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {missing}")
    Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)


def warm_up() -> None:
    """Load every artifact into memory (call once at server start-up)."""
    require_artifacts()
    load_timing_model()
    load_voice_model()
    load_stacker()
    load_metadata()
    load_temperature()


# --- probability helpers --------------------------------------------------------

def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def apply_temperature(p_synthetic, temperature: float):
    temperature = max(float(temperature), 0.05)
    out = sigmoid(logit(p_synthetic) / temperature)
    return float(out) if np.ndim(out) == 0 else out


def fit_temperature(probabilities, labels) -> float:
    """Temperature that minimises NLL, restricted to softening (T >= 1).

    When the training scores are (nearly) separable the NLL optimum is T -> 0,
    i.e. sharpening towards 0/1; that is over-confidence that does not survive
    unseen callers and voices, so we never allow it.
    """
    z = logit(probabilities)
    y = np.asarray(labels, dtype=float)

    def nll(temperature: float) -> float:
        calibrated = np.clip(sigmoid(z / temperature), 1e-6, 1.0 - 1e-6)
        return float(-np.mean(y * np.log(calibrated) + (1.0 - y) * np.log(1.0 - calibrated)))

    grid = np.linspace(1.0, 3.0, 41)
    scores = [nll(float(t)) for t in grid]
    return float(grid[int(np.argmin(scores))])


# --- fusion -----------------------------------------------------------------------

def stack_features(p_timing, p_voice) -> np.ndarray:
    """Stacker input: the two head logits (works for scalars or arrays)."""
    return np.column_stack([logit(p_timing), logit(p_voice)])


def fit_stacker(p_timing, p_voice, labels) -> LogisticRegression:
    X = stack_features(p_timing, p_voice)
    y = np.asarray(labels, dtype=int)
    # Moderate regularisation: the head logits are already well separated on
    # train, and an unregularised stacker would just inflate them.
    model = LogisticRegression(
        C=0.3,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    return model


def fuse_probabilities(p_timing, p_voice, stacker=None, temperature: float | None = None):
    """Fused, temperature-calibrated P(synthetic) from the two head probabilities."""
    if stacker is None:
        stacker = load_stacker()
    if temperature is None:
        temperature = load_temperature()
    fused = stacker.predict_proba(stack_features(p_timing, p_voice))[:, 1]
    fused = apply_temperature(fused, temperature)
    return float(fused[0]) if np.ndim(p_timing) == 0 else fused
