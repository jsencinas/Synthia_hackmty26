from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def mini_turns_path() -> str:
    return str(ROOT / "tests" / "fixtures" / "mini_turns.json")
