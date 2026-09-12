"""Trained artifacts: load/save models, temperature-calibrate timing scores, fuse heads."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import (
    ARTIFACT_DIR,
    CALIBRATION_PATH,
    METADATA_PATH,
    STACKER_PATH,
    TIMING_MODEL_PATH,
    VOICE_MODEL_PATH,
)


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


def atomic_xgb_save(model: XGBClassifier, destination: str | Path) -> None:
    temporary = _temporary_path(destination, suffix=".json")
    try:
        model.save_model(temporary)
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
def load_voice_model() -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(VOICE_MODEL_PATH)
    return model


@lru_cache(maxsize=1)
def load_stacker():
    return joblib.load(STACKER_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    return json.loads(Path(METADATA_PATH).read_text(encoding="utf-8"))


def require_artifacts() -> None:
    required = artifact_paths()
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {missing}")
    Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)


def apply_temperature(p_synthetic: float, temperature: float) -> float:
    p = min(max(float(p_synthetic), 1e-6), 1.0 - 1e-6)
    temperature = max(float(temperature), 0.05)
    logit = np.log(p / (1.0 - p))
    return float(1.0 / (1.0 + np.exp(-logit / temperature)))


def fit_temperature(probabilities, labels) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(labels, dtype=float)

    def nll(temperature: float) -> float:
        calibrated = np.array([apply_temperature(v, temperature) for v in p])
        calibrated = np.clip(calibrated, 1e-6, 1.0 - 1e-6)
        return float(-np.mean(y * np.log(calibrated) + (1.0 - y) * np.log(1.0 - calibrated)))

    grid = np.linspace(0.2, 4.0, 39)
    best_t = 1.0
    best = nll(best_t)
    for temperature in grid:
        score = nll(float(temperature))
        if score < best:
            best = score
            best_t = float(temperature)
    return best_t


def load_temperature(path: str = CALIBRATION_PATH) -> float:
    calibration_path = Path(path)
    if not calibration_path.exists():
        return 1.0
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    return float(data.get("temperature", 1.0))


def save_calibration(temperature: float, extra: dict | None = None, path: str = CALIBRATION_PATH) -> None:
    payload = {"temperature": float(temperature)}
    if extra:
        payload.update(extra)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stack_features(p_json: float, p_voice: float) -> list[float]:
    return [
        float(p_json),
        float(p_voice),
        abs(float(p_json) - float(p_voice)),
    ]


def fit_stacker(p_json, p_voice, labels) -> LogisticRegression:
    X = np.array(
        [stack_features(j, v) for j, v in zip(p_json, p_voice)],
        dtype=float,
    )
    y = np.asarray(labels, dtype=int)
    model = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X, y)
    return model
