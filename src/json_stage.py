from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import predict
import turns
from src.config import JSON_MODEL_PATH
from src.types import StageResult, certainty_from_probability, label_from_probability


def run(turns_path: str) -> StageResult:
    if not os.path.exists(JSON_MODEL_PATH):
        return StageResult(
            stage="json",
            is_synthetic=False,
            p_synthetic=0.5,
            certainty_pct=0.0,
            features={},
            error="json_model_missing",
        )

    if not turns_path or not os.path.exists(turns_path):
        return StageResult(
            stage="json",
            is_synthetic=False,
            p_synthetic=0.5,
            certainty_pct=0.0,
            features={},
            error="json_turns_missing",
        )

    try:
        metricas = turns.analizar_llamada(turns_path)
        pred = predict.predecir_muestra(metricas, usar_umbral_optimo=False)
        p_synthetic = float(pred["probabilidad_synthetic"])
        return StageResult(
            stage="json",
            is_synthetic=label_from_probability(p_synthetic),
            p_synthetic=p_synthetic,
            certainty_pct=certainty_from_probability(p_synthetic),
            features=metricas,
        )
    except FileNotFoundError:
        return StageResult(
            stage="json",
            is_synthetic=False,
            p_synthetic=0.5,
            certainty_pct=0.0,
            features={},
            error="json_model_missing",
        )
    except Exception as exc:
        return StageResult(
            stage="json",
            is_synthetic=False,
            p_synthetic=0.5,
            certainty_pct=0.0,
            features={},
            error=f"json_failed:{exc}",
        )
