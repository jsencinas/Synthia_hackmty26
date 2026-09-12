from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import CALIBRATION_PATH


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
