from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def mini_turns_path() -> str:
    return str(ROOT / "tests" / "fixtures" / "mini_turns.json")


@pytest.fixture
def mini_turns_payload(mini_turns_path) -> dict:
    return json.loads(Path(mini_turns_path).read_text(encoding="utf-8"))


@pytest.fixture
def mini_audio_path(tmp_path) -> str:
    sample_rate = 8000
    time = np.arange(sample_rate * 7) / sample_rate
    caller = 0.15 * np.sin(2 * np.pi * 180 * time)
    agent = 0.12 * np.sin(2 * np.pi * 240 * time)
    audio = np.column_stack([caller, agent]).astype(np.float32)
    path = tmp_path / "mini.wav"
    sf.write(path, audio, sample_rate)
    return str(path)
