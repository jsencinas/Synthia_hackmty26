from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import soundfile as sf

from src.config import AUDIO_DIR, MANIFEST_PATH


REQUIRED_COLUMNS = {"anon_id", "label", "split", "duration_s"}
ALLOWED_LABELS = {"human", "synthetic"}
ALLOWED_SPLITS = {"train", "val"}


def validate_manifest(manifest: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("Manifest is empty.")
    if manifest["anon_id"].isna().any() or manifest["anon_id"].duplicated().any():
        raise ValueError("Manifest anon_id values must be non-empty and unique.")
    if not set(manifest["label"].dropna()).issubset(ALLOWED_LABELS):
        raise ValueError("Manifest contains an unsupported label.")
    if not set(manifest["split"].dropna()).issubset(ALLOWED_SPLITS):
        raise ValueError("Manifest contains an unsupported split.")
    if manifest[["label", "split", "duration_s"]].isna().any().any():
        raise ValueError("Manifest contains missing required values.")
    durations = pd.to_numeric(manifest["duration_s"], errors="coerce")
    if durations.isna().any() or (durations <= 0).any():
        raise ValueError("Manifest durations must be positive.")


def load_split(split: str, manifest_path: str | Path = MANIFEST_PATH) -> pd.DataFrame:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    manifest = pd.read_csv(manifest_path)
    validate_manifest(manifest)
    selected = manifest.loc[manifest["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"Manifest has no rows for split={split!r}.")
    if set(selected["split"]) != {split}:
        raise RuntimeError(f"Split isolation failed for {split!r}.")
    return selected.reset_index(drop=True)


def audio_path_for(anon_id: str, audio_dir: str | Path = AUDIO_DIR) -> Path:
    return Path(audio_dir) / f"{anon_id}.wav"


def validate_audio_files(
    rows: pd.DataFrame,
    audio_dir: str | Path = AUDIO_DIR,
    *,
    expected_sample_rate: int = 8000,
) -> None:
    for anon_id in rows["anon_id"]:
        path = audio_path_for(str(anon_id), audio_dir)
        if not path.is_file():
            raise FileNotFoundError(f"Missing audio: {path}")
        info = sf.info(path)
        if info.channels != 2:
            raise ValueError(f"Expected stereo audio: {path}")
        if info.samplerate != expected_sample_rate:
            raise ValueError(
                f"Expected {expected_sample_rate} Hz audio, got {info.samplerate}: {path}"
            )
        if info.frames <= 0:
            raise ValueError(f"Audio is empty: {path}")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_no_cross_split_duplicates(
    train: pd.DataFrame,
    val: pd.DataFrame,
    audio_dir: str | Path = AUDIO_DIR,
) -> None:
    overlap = set(train["anon_id"]) & set(val["anon_id"])
    if overlap:
        raise ValueError(f"Train/VAL ID overlap: {sorted(overlap)[:5]}")
    train_hashes = {
        file_sha256(audio_path_for(str(anon_id), audio_dir)): str(anon_id)
        for anon_id in train["anon_id"]
    }
    for anon_id in val["anon_id"]:
        digest = file_sha256(audio_path_for(str(anon_id), audio_dir))
        if digest in train_hashes:
            raise ValueError(
                f"Train/VAL audio duplicate: {train_hashes[digest]} and {anon_id}"
            )


def dataset_fingerprint(
    rows: pd.DataFrame,
    audio_dir: str | Path = AUDIO_DIR,
) -> str:
    digest = hashlib.sha256()
    for row in rows.sort_values("anon_id").itertuples(index=False):
        digest.update(
            f"{row.anon_id}\0{row.label}\0{row.split}\0{row.duration_s}\n".encode()
        )
        digest.update(file_sha256(audio_path_for(str(row.anon_id), audio_dir)).encode())
    return digest.hexdigest()
