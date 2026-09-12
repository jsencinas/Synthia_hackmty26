"""Score frozen train-only artifacts on the val split. Does not write models.

Reports each head, the fused model, and the end-to-end ``detect_audio`` path
(STT disabled) with per-call latency so the numbers match what the API serves.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)

from src import detect as detect_module
from src.data import audio_path_for, load_split, validate_audio_files
from src.features import TIMING_FEATURES, VOICE_FEATURES, analyze_audio, load_audio
from src.models import (
    artifact_hashes,
    fuse_probabilities,
    load_metadata,
    load_timing_model,
    load_voice_model,
    require_artifacts,
    warm_up,
)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def evaluate() -> dict:
    require_artifacts()
    hashes_before = artifact_hashes()
    metadata = load_metadata()
    if metadata.get("trained_on") != "train_only":
        raise ValueError("Refusing to evaluate an artifact without train-only provenance.")
    warm_up()

    rows = load_split("val")
    if set(rows["split"]) != {"val"}:
        raise ValueError("Evaluation accepts VAL rows only.")
    validate_audio_files(rows)

    # Force the acoustic-only path so results are deterministic and offline.
    detect_module.STT_ENABLED = False

    timing_rows, voice_rows, labels, e2e_probability, latencies, stopped_at = [], [], [], [], [], []
    for row in rows.itertuples(index=False):
        audio = load_audio(audio_path_for(str(row.anon_id)))
        analysis = analyze_audio(audio)
        timing_rows.append(analysis.timing)
        voice_rows.append(analysis.voice)
        labels.append(1 if row.label == "synthetic" else 0)

        started = time.perf_counter()
        result = detect_module.detect_audio(audio)
        latencies.append(time.perf_counter() - started)
        p = result["confidence"] if result["is_synthetic"] else 1.0 - result["confidence"]
        e2e_probability.append(p)
        stopped_at.append(result["stopped_at"])

    labels = np.asarray(labels, dtype=int)
    timing_features = pd.DataFrame(timing_rows, columns=TIMING_FEATURES)
    voice_features = pd.DataFrame(voice_rows, columns=VOICE_FEATURES)

    timing_probability = load_timing_model().predict_proba(timing_features)[:, 1]
    voice_probability = load_voice_model().predict_proba(voice_features)[:, 1]
    fused_probability = np.asarray(fuse_probabilities(timing_probability, voice_probability))
    e2e_probability = np.asarray(e2e_probability)
    predictions = (e2e_probability >= 0.5).astype(int)

    metrics = {
        "split": "val",
        "rows": int(len(rows)),
        "timing_head": _metrics(labels, timing_probability),
        "voice_head": _metrics(labels, voice_probability),
        "fused": _metrics(labels, fused_probability),
        "end_to_end": _metrics(labels, e2e_probability),
        "heads_disagree_frac": float(
            np.mean((timing_probability >= 0.5) != (voice_probability >= 0.5))
        ),
        "uncertain_frac": float(
            np.mean(np.abs(fused_probability - 0.5) <= detect_module.STT_UNCERTAINTY_BAND)
        ),
        "latency_s": {
            "mean": float(np.mean(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(np.max(latencies)),
        },
        "stopped_at": {stage: int(stopped_at.count(stage)) for stage in set(stopped_at)},
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(labels, predictions, target_names=["human", "synthetic"]))
    wrong = rows.loc[predictions != labels, "anon_id"].tolist()
    if wrong:
        print("Misclassified:", wrong)
    if artifact_hashes() != hashes_before:
        raise RuntimeError("Evaluation modified model artifacts.")
    return metrics


if __name__ == "__main__":
    evaluate()
