from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    call_id: str = Field(..., min_length=1)
    features: dict[str, float | int | None]


class RedFlag(BaseModel):
    code: str
    title: str
    severity: str
    description: str
    feature: str | None = None
    z_score: float | None = None


class Indicator(BaseModel):
    feature: str
    value: float | None = None
    signal: str
    importance: float
    z_score: float | None = None


class Risk(BaseModel):
    level: str
    score: float


class DetectResponse(BaseModel):
    api_version: str
    model_version: str
    response_version: str
    call_id: str
    classification: str
    is_synthetic: bool
    confidence: float
    risk: Risk
    scores: dict[str, float]
    features: dict[str, Any]
    red_flags: list[RedFlag]
    top_indicators: list[Indicator]
    recommendations: list[str]


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
