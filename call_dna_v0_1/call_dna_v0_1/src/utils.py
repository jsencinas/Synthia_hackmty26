from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directories() -> None:
    """Create runtime directories used by the project."""
    from .config import OUTPUTS_DIR, PLOTS_DIR, MODELS_DIR
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_divide(numerator: float, denominator: float) -> float:
    """Return 0.0 for a zero denominator."""
    if denominator == 0:
        return 0.0
    return numerator / denominator
