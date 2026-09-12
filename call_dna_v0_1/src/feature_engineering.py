from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import AGENT_CHANNEL, CALLER_CHANNEL
from .data_loader import load_manifest, load_turns
from .utils import safe_divide


FEATURE_NAMES = [
    "numero_turnos_caller",
    "duracion_total_llamada",
    "duracion_promedio_caller",
    "duracion_std_caller",
    "duracion_min_caller",
    "duracion_max_caller",
    "pausa_promedio_caller",
    "pausa_std_caller",
    "pausa_min_caller",
    "pausa_max_caller",
    "numero_pausas",
    "latencia_promedio",
    "latencia_std",
    "latencia_min",
    "latencia_max",
    "numero_interrupciones",
    "duracion_total_interrupciones",
    "duracion_promedio_interrupcion",
    "interrupcion_std",
    "interruption_rate",
    "overlap_ratio",
]


def _clean_turns(turns: Iterable[dict[str, Any]]) -> list[dict[str, float]]:
    """Sort valid turns and normalize numeric values."""
    cleaned = []
    for turn in turns:
        start = float(turn["start"])
        end = float(turn["end"])
        if end < start:
            continue
        cleaned.append(
            {
                "channel": int(turn["channel"]),
                "start": start,
                "end": end,
            }
        )
    return sorted(cleaned, key=lambda x: (x["start"], x["end"]))


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Merge overlapping or contiguous intervals.

    This prevents multiple agent segments overlapping the same caller segment
    from being counted as multiple interruption events.
    """
    intervals = sorted(
        [(float(start), float(end)) for start, end in intervals if end > start],
        key=lambda x: (x[0], x[1]),
    )
    if not intervals:
        return []

    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_overlap(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Return overlap duration between two intervals."""
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _caller_turns(turns: list[dict[str, float]]) -> list[dict[str, float]]:
    return [t for t in turns if t["channel"] == CALLER_CHANNEL]


def _agent_turns(turns: list[dict[str, float]]) -> list[dict[str, float]]:
    return [t for t in turns if t["channel"] == AGENT_CHANNEL]


def _duration_stats(durations: list[float]) -> tuple[float, float, float, float]:
    if not durations:
        return (0.0, 0.0, 0.0, 0.0)
    values = np.asarray(durations, dtype=float)
    return (
        float(values.mean()),
        float(values.std(ddof=0)),
        float(values.min()),
        float(values.max()),
    )


def calculate_latency_features(
    caller_turns: list[dict[str, float]],
    agent_turns: list[dict[str, float]],
) -> list[float]:
    """
    For every caller turn, find the latest agent turn that ended before it.

    Negative latencies are never produced. The first caller turn without a
    previous agent turn is ignored.
    """
    agent_ends = sorted(t["end"] for t in agent_turns)
    latencies: list[float] = []

    for caller in caller_turns:
        previous_ends = [end for end in agent_ends if end <= caller["start"]]
        if not previous_ends:
            continue
        latency = caller["start"] - max(previous_ends)
        if latency >= 0:
            latencies.append(latency)

    return latencies


def calculate_interruption_features(
    caller_turns: list[dict[str, float]],
    agent_turns: list[dict[str, float]],
) -> tuple[int, float, float, float]:
    """
    Calculate interruption events from temporal overlap.

    Agent intervals are first merged, so multiple agent segments overlapping
    the same caller segment do not become multiple interruption events.
    """
    merged_agent = merge_intervals(
        (t["start"], t["end"]) for t in agent_turns
    )

    interruption_events: list[tuple[float, float]] = []

    for caller in caller_turns:
        caller_interval = (caller["start"], caller["end"])
        for agent_interval in merged_agent:
            overlap_start = max(caller_interval[0], agent_interval[0])
            overlap_end = min(caller_interval[1], agent_interval[1])
            if overlap_end > overlap_start:
                interruption_events.append((overlap_start, overlap_end))

    # Merge again because overlaps can touch across adjacent segments.
    interruption_events = merge_intervals(interruption_events)

    durations = [
        end - start for start, end in interruption_events
    ]
    if not durations:
        return 0, 0.0, 0.0, 0.0

    values = np.asarray(durations, dtype=float)
    return (
        len(durations),
        float(values.sum()),
        float(values.mean()),
        float(values.std(ddof=0)),
    )


def calculate_features(
    turns: list[dict[str, Any]],
    duration_s: float | None = None,
) -> dict[str, float]:
    """
    Convert turn-level annotations into one row of call-level features.

    Only JSON turn metadata is used. No audio, transcript, embedding or LLM
    feature is used in this baseline.
    """
    turns = _clean_turns(turns)
    caller = _caller_turns(turns)
    agent = _agent_turns(turns)

    caller_durations = [
        t["end"] - t["start"] for t in caller if t["end"] > t["start"]
    ]
    duration_mean, duration_std, duration_min, duration_max = _duration_stats(
        caller_durations
    )

    pauses = []
    for previous, current in zip(caller, caller[1:]):
        pause = max(0.0, current["start"] - previous["end"])
        pauses.append(pause)

    pause_mean, pause_std, pause_min, pause_max = _duration_stats(pauses)

    latencies = calculate_latency_features(caller, agent)
    latency_mean, latency_std, latency_min, latency_max = _duration_stats(
        latencies
    )

    interruption_count, interruption_total, interruption_mean, interruption_std = (
        calculate_interruption_features(caller, agent)
    )

    # Use manifest duration when available. Otherwise use the temporal span
    # covered by the annotations.
    if duration_s is not None and np.isfinite(duration_s):
        total_call_duration = float(duration_s)
    elif turns:
        total_call_duration = max(t["end"] for t in turns) - min(
            t["start"] for t in turns
        )
    else:
        total_call_duration = 0.0

    # Compute total overlap from merged interruption events.
    overlap_ratio = safe_divide(
        interruption_total,
        total_call_duration,
    )

    features = {
        "numero_turnos_caller": float(len(caller)),
        "duracion_total_llamada": total_call_duration,
        "duracion_promedio_caller": duration_mean,
        "duracion_std_caller": duration_std,
        "duracion_min_caller": duration_min,
        "duracion_max_caller": duration_max,
        "pausa_promedio_caller": pause_mean,
        "pausa_std_caller": pause_std,
        "pausa_min_caller": pause_min,
        "pausa_max_caller": pause_max,
        "numero_pausas": float(len(pauses)),
        "latencia_promedio": latency_mean,
        "latencia_std": latency_std,
        "latencia_min": latency_min,
        "latencia_max": latency_max,
        "numero_interrupciones": float(interruption_count),
        "duracion_total_interrupciones": interruption_total,
        "duracion_promedio_interrupcion": interruption_mean,
        "interrupcion_std": interruption_std,
        "interruption_rate": safe_divide(
            interruption_count, len(caller)
        ),
        "overlap_ratio": overlap_ratio,
    }

    return features


def build_dataset(
    manifest_path,
    turns_dir,
) -> pd.DataFrame:
    """Build one feature row per call and merge labels/splits from manifest."""
    manifest = load_manifest(manifest_path)
    rows = []

    for row in manifest.itertuples(index=False):
        turns = load_turns(turns_dir, row.anon_id)
        duration = getattr(row, "duration_s", None)
        features = calculate_features(turns, duration_s=duration)
        rows.append({"anon_id": row.anon_id, **features})

    features_df = pd.DataFrame(rows)

    dataset = manifest[["anon_id", "label", "split"]].merge(
        features_df,
        on="anon_id",
        how="left",
        validate="one_to_one",
    )

    ordered_columns = ["anon_id"] + FEATURE_NAMES + ["label", "split"]
    return dataset[ordered_columns]
