from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load and validate the dataset manifest."""
    df = pd.read_csv(manifest_path)

    required = {"anon_id", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv is missing columns: {sorted(missing)}")

    df["anon_id"] = df["anon_id"].astype(str)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["split"] = df["split"].astype(str).str.lower().str.strip()

    invalid_labels = set(df["label"]) - {"human", "synthetic"}
    if invalid_labels:
        raise ValueError(f"Unsupported labels: {sorted(invalid_labels)}")

    invalid_splits = set(df["split"]) - {"train", "val"}
    if invalid_splits:
        raise ValueError(f"Unsupported splits: {sorted(invalid_splits)}")

    return df


def load_turns(turns_path: Path, anon_id: str) -> list[dict[str, Any]]:
    """Load turns for one call."""
    json_path = turns_path / f"{anon_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Turns file not found: {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    turns = payload.get("turns", [])

    if not isinstance(turns, list):
        raise ValueError(f"'turns' must be a list in {json_path}")

    normalized = []
    for index, turn in enumerate(turns):
        if not all(key in turn for key in ("channel", "start", "end")):
            raise ValueError(
                f"Turn {index} in {json_path} must contain channel, start and end"
            )

        start = float(turn["start"])
        end = float(turn["end"])
        channel = int(turn["channel"])

        if end < start:
            raise ValueError(f"Turn {index} has end < start in {json_path}")

        normalized.append(
            {"channel": channel, "start": start, "end": end}
        )

    return normalized
