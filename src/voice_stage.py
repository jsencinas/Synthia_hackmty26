from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import modeloVozFINAL
import voces
from src.types import StageResult, certainty_from_probability, label_from_probability


def run(audio_path: str, turns_path: str) -> StageResult:
    if not audio_path or not os.path.exists(audio_path):
        return StageResult(
            stage="voice",
            is_synthetic=False,
            p_synthetic=0.5,
            certainty_pct=0.0,
            features={},
            error="voice_audio_missing",
        )

    if not turns_path or not os.path.exists(turns_path):
        return StageResult(
            stage="voice",
            is_synthetic=False,
            p_synthetic=0.5,
            certainty_pct=0.0,
            features={},
            error="voice_turns_missing",
        )

    try:
        feats = voces.analizar_llamada(audio_path, turns_path)
    except Exception as exc:
        return StageResult(
            stage="voice",
            is_synthetic=False,
            p_synthetic=0.5,
            certainty_pct=0.0,
            features={},
            error=f"voice_failed:{exc}",
        )

    if feats is None:
        return StageResult(
            stage="voice",
            is_synthetic=False,
            p_synthetic=0.5,
            certainty_pct=0.0,
            features={},
            error="voice_no_caller_slices",
        )

    try:
        decision, certeza = modeloVozFINAL.clasificar_llamada(
            feats["mfcc_1_std"],
            feats["shimmer"],
            feats["pitch_delta_std"],
            feats["jitter"],
        )
        is_synthetic = decision == "synthetic"
        p_synthetic = float(certeza) if is_synthetic else 1.0 - float(certeza)
        return StageResult(
            stage="voice",
            is_synthetic=label_from_probability(p_synthetic),
            p_synthetic=p_synthetic,
            certainty_pct=certainty_from_probability(p_synthetic),
            features=feats,
        )
    except Exception as exc:
        return StageResult(
            stage="voice",
            is_synthetic=False,
            p_synthetic=0.5,
            certainty_pct=0.0,
            features=feats,
            error=f"voice_failed:{exc}",
        )
