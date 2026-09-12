from __future__ import annotations

from typing import Any

import numpy as np


# Initial heuristic: z > 2 is intentionally documented as a baseline.
# Human reference statistics MUST be calculated from TRAIN only.
Z_THRESHOLD = 2.0

INTERACTION_FEATURES = {
    "numero_interrupciones",
    "interruption_rate",
    "overlap_ratio",
    "latencia_promedio",
    "latencia_std",
    "latencia_min",
    "latencia_max",
}

BEHAVIORAL_FEATURES = {
    "duracion_promedio_caller",
    "duracion_std_caller",
    "duracion_min_caller",
    "duracion_max_caller",
    "pausa_promedio_caller",
    "pausa_std_caller",
    "pausa_min_caller",
    "pausa_max_caller",
    "numero_pausas",
}


def _z_score(value: float, mean: float, std: float) -> float:
    if std <= 1e-12:
        return 0.0
    return abs(value - mean) / std


def generate_red_flags(
    features: dict[str, Any],
    human_stats: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Generate explainable anomaly flags against TRAIN human distributions."""
    flag_specs = {
        "latencia_std": (
            "CONSISTENT_LATENCY",
            "Highly consistent response timing",
            "Response latency showed unusually low variability compared with the human reference distribution.",
        ),
        "interruption_rate": (
            "LOW_INTERRUPTION_RATE",
            "Low interruption frequency",
            "Caller-agent overlap occurred at an unusual rate compared with the human reference distribution.",
        ),
        "duracion_std_caller": (
            "REGULAR_TURN_DURATION",
            "Highly regular caller turn duration",
            "Caller turn durations showed unusual variability compared with the human reference distribution.",
        ),
        "pausa_std_caller": (
            "LOW_PAUSE_VARIABILITY",
            "Low pause variability",
            "Caller pause durations showed unusually low variability compared with the human reference distribution.",
        ),
        "overlap_ratio": (
            "UNUSUAL_OVERLAP_PATTERN",
            "Unusual overlap pattern",
            "Temporal overlap between caller and agent differed substantially from the human reference distribution.",
        ),
    }

    flags = []
    for feature, (code, title, description) in flag_specs.items():
        if feature not in features or feature not in human_stats:
            continue

        value = features.get(feature)
        stats = human_stats[feature]

        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 0.0))
        z = _z_score(value, mean, std)

        if z > Z_THRESHOLD:
            severity = "high" if z > 3.0 else "medium"
            flags.append(
                {
                    "code": code,
                    "title": title,
                    "severity": severity,
                    "description": description,
                    "feature": feature,
                    "z_score": round(z, 3),
                }
            )

    return flags


def calculate_risk(
    classification: str,
    confidence: float,
    red_flags: list[dict[str, Any]],
) -> dict[str, float | str]:
    """
    Separate AI detection from risk assessment.

    Synthetic does NOT mean fraud. Risk is an operational triage score, not
    a statement that fraud occurred.
    """
    flag_score = min(1.0, len(red_flags) / 3.0)

    if classification == "synthetic":
        # Confidence contributes, but flags also matter.
        score = 0.70 * confidence + 0.30 * flag_score
        if score >= 0.80:
            level = "high"
        elif score >= 0.50:
            level = "medium"
        else:
            level = "low"
    else:
        # A human classification should generally reduce this score, while
        # leaving room for interaction anomalies to increase operational risk.
        score = 0.25 * confidence + 0.35 * flag_score
        if score >= 0.65:
            level = "high"
        elif score >= 0.35:
            level = "medium"
        else:
            level = "low"

    return {"level": level, "score": round(float(score), 4)}


def _importance_map(
    feature_importances: dict[str, float],
) -> dict[str, float]:
    total = sum(max(0.0, value) for value in feature_importances.values())
    if total <= 0:
        return {key: 0.0 for key in feature_importances}
    return {
        key: max(0.0, value) / total
        for key, value in feature_importances.items()
    }


def build_top_indicators(
    features: dict[str, Any],
    human_stats: dict[str, dict[str, float]],
    feature_importances: dict[str, float],
    classification: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Combine XGBoost importance with deviation from the human reference.

    The importance is model feature importance, not a causal explanation.
    """
    normalized_importance = _importance_map(feature_importances)
    candidates = []

    for feature, importance in normalized_importance.items():
        if feature not in features or feature not in human_stats:
            continue

        try:
            value = float(features[feature])
        except (TypeError, ValueError):
            continue

        stats = human_stats[feature]
        z = _z_score(
            value,
            float(stats.get("mean", 0.0)),
            float(stats.get("std", 0.0)),
        )

        if z == 0:
            continue

        signal = "synthetic_like" if classification == "synthetic" else "human_like"
        candidates.append(
            {
                "feature": feature,
                "value": value,
                "signal": signal,
                "importance": round(float(importance), 4),
                "z_score": round(float(z), 3),
            }
        )

    candidates.sort(
        key=lambda item: item["importance"] * max(1.0, item["z_score"]),
        reverse=True,
    )
    return candidates[:limit]


def build_recommendations(
    classification: str,
    risk_level: str,
) -> list[str]:
    if classification == "synthetic" and risk_level == "high":
        return [
            "Avoid sharing sensitive financial information.",
            "Verify the caller through an official communication channel.",
            "Do not provide passwords, PINs or authentication codes.",
            "If fraud is suspected, contact the institution directly.",
        ]

    if classification == "synthetic":
        return [
            "Treat the call with additional caution.",
            "Verify the caller through an official communication channel.",
            "Do not provide passwords, PINs or authentication codes.",
        ]

    return [
        "No strong synthetic-voice indicators were detected.",
        "This result does not guarantee that the call is legitimate.",
        "Continue following normal identity verification procedures.",
    ]


def build_risk_result(
    prediction: dict[str, Any],
    features: dict[str, Any],
    human_stats: dict[str, dict[str, float]],
    feature_importances: dict[str, float],
) -> dict[str, Any]:
    flags = generate_red_flags(features, human_stats)
    risk = calculate_risk(
        prediction["classification"],
        prediction["confidence"],
        flags,
    )
    top_indicators = build_top_indicators(
        features,
        human_stats,
        feature_importances,
        prediction["classification"],
    )

    behavioral_values = []
    interaction_values = []
    for flag in flags:
        feature = flag.get("feature")
        if feature in BEHAVIORAL_FEATURES:
            behavioral_values.append(flag["z_score"])
        if feature in INTERACTION_FEATURES:
            interaction_values.append(flag["z_score"])

    # These scores are heuristic indicators, not probabilities.
    behavioral_anomaly = min(
        1.0, float(np.mean(behavioral_values) / 4.0)
    ) if behavioral_values else 0.0
    interaction_anomaly = min(
        1.0, float(np.mean(interaction_values) / 4.0)
    ) if interaction_values else 0.0

    if prediction["classification"] == "synthetic":
        behavioral_score = max(
            prediction["confidence"], behavioral_anomaly
        )
        interaction_score = max(
            prediction["confidence"], interaction_anomaly
        )
    else:
        behavioral_score = 1.0 - behavioral_anomaly
        interaction_score = 1.0 - interaction_anomaly

    return {
        "risk": risk,
        "scores": {
            "behavioral": round(float(behavioral_score), 4),
            "interaction": round(float(interaction_score), 4),
        },
        "red_flags": flags,
        "top_indicators": top_indicators,
        "recommendations": build_recommendations(
            prediction["classification"],
            risk["level"],
        ),
    }
