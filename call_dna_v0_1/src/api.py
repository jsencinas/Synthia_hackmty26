from __future__ import annotations

from typing import Any

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import (
    API_VERSION,
    FEATURE_COLUMNS_PATH,
    HUMAN_STATS_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    RESPONSE_VERSION,
)
from .predict import predict_from_features
from .risk_engine import build_risk_result
from .schemas import DetectRequest, DetectResponse, HealthResponse
from .utils import load_json


app = FastAPI(
    title="CallDNA API",
    version=MODEL_VERSION,
    description="Baseline synthetic-voice detection from call turn metadata.",
)


def _load_runtime_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact missing. Run `python train.py` first."
        )
    if not HUMAN_STATS_PATH.exists():
        raise FileNotFoundError(
            "Human reference statistics missing. Run `python train.py` first."
        )

    model = joblib.load(MODEL_PATH)
    human_stats = load_json(HUMAN_STATS_PATH)
    feature_columns = load_json(FEATURE_COLUMNS_PATH)
    feature_importances = dict(
        zip(feature_columns, model.feature_importances_)
    )
    return human_stats, feature_importances


def _build_response(
    call_id: str,
    features: dict[str, Any],
) -> dict[str, Any]:
    prediction = predict_from_features(features)
    human_stats, feature_importances = _load_runtime_artifacts()

    risk_result = build_risk_result(
        prediction,
        features,
        human_stats,
        feature_importances,
    )

    return {
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "response_version": RESPONSE_VERSION,
        "call_id": call_id,
        "classification": prediction["classification"],
        "is_synthetic": prediction["is_synthetic"],
        "confidence": round(float(prediction["confidence"]), 4),
        "risk": risk_result["risk"],
        "scores": risk_result["scores"],
        "features": features,
        "red_flags": risk_result["red_flags"],
        "top_indicators": risk_result["top_indicators"],
        "recommendations": risk_result["recommendations"],
    }


@app.get("/health", response_model=HealthResponse)
@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model="call_dna_xgboost",
        version=MODEL_VERSION,
    )


@app.post("/detect", response_model=DetectResponse)
@app.post("/v1/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    try:
        result = _build_response(request.call_id, request.features)
        return DetectResponse.model_validate(result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {exc}",
        ) from exc


@app.post("/analyze")
@app.post("/v1/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Architecture placeholder for the audio pipeline.

    v0.1 intentionally does NOT infer turn boundaries from WAV. Automatic
    VAD/diarization/Whisper is planned for a later level.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="v0.1 accepts WAV files only.",
        )

    raise HTTPException(
        status_code=501,
        detail=(
            "Audio turn extraction is not implemented in v0.1. "
            "Use /detect with JSON turn-derived features. "
            "The future pipeline is WAV -> channel separation -> "
            "turn extraction -> feature extraction -> XGBoost."
        ),
    )
