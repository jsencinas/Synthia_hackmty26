from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.artifacts import load_stacker
from src.types import StageResult, certainty_from_probability, label_from_probability


def stack_features(p_json: float, p_voice: float) -> list[float]:
    return [
        float(p_json),
        float(p_voice),
        abs(float(p_json) - float(p_voice)),
    ]


def heads_disagree(json_result: StageResult, voice_result: StageResult) -> bool:
    if json_result.error or voice_result.error:
        return False
    return json_result.is_synthetic != voice_result.is_synthetic


def fit_stacker(p_json, p_voice, labels) -> LogisticRegression:
    X = np.array(
        [stack_features(j, v) for j, v in zip(p_json, p_voice)],
        dtype=float,
    )
    y = np.asarray(labels, dtype=int)
    model = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X, y)
    return model


def fuse(json_result: StageResult, voice_result: StageResult, stacker=None) -> StageResult:
    if stacker is None:
        stacker = load_stacker()
    p = float(
        stacker.predict_proba(
            [stack_features(json_result.p_synthetic, voice_result.p_synthetic)]
        )[0, 1]
    )

    return StageResult(
        stage="fusion",
        is_synthetic=label_from_probability(p),
        p_synthetic=p,
        certainty_pct=certainty_from_probability(p),
        features={
            "p_json": json_result.p_synthetic,
            "p_voice": voice_result.p_synthetic,
            "disagree": heads_disagree(json_result, voice_result),
        },
    )
