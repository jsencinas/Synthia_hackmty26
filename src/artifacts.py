from __future__ import annotations

import json
import os
import tempfile
import hashlib
from functools import lru_cache
from pathlib import Path

import joblib
from xgboost import XGBClassifier

from src.config import (
    ARTIFACT_DIR,
    CALIBRATION_PATH,
    JSON_METADATA_PATH,
    JSON_MODEL_PATH,
    STACKER_PATH,
    VOICE_MODEL_PATH,
)


def artifact_paths() -> list[Path]:
    return [
        Path(JSON_MODEL_PATH),
        Path(VOICE_MODEL_PATH),
        Path(STACKER_PATH),
        Path(CALIBRATION_PATH),
        Path(JSON_METADATA_PATH),
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
    return joblib.load(JSON_MODEL_PATH)


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
    return json.loads(Path(JSON_METADATA_PATH).read_text(encoding="utf-8"))


def require_artifacts() -> None:
    required = artifact_paths()
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {missing}")
    Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
