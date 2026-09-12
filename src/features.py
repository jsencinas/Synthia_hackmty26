"""Speaker turns from WAV energy, then timing and voice features for the models."""

from __future__ import annotations

import statistics
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


FRAME_MS = 30
HANGOVER_FRAMES = 8
MIN_TURN_S = 0.15
ENERGY_PERCENTILE = 60

TIMING_BASE_FEATURES = [
    "numero_turnos_caller",
    "duracion_promedio_caller",
    "pausa_promedio_caller",
    "desviacion_estandar_latencia",
    "interrupciones_agente",
    "duracion_total_interrupciones",
    "duracion_promedio_interrupcion",
]
TIMING_FEATURES = TIMING_BASE_FEATURES + [
    "tasa_interrupciones",
    "duracion_interrupciones_por_turno",
    "ratio_habla_pausa",
]
VOICE_FEATURES = ["mfcc_1_std", "shimmer", "pitch_delta_std", "jitter"]


def _frame_energy(samples: np.ndarray, frame_len: int) -> np.ndarray:
    pad = (-len(samples)) % frame_len
    if pad:
        samples = np.pad(samples, (0, pad))
    frames = samples.reshape(-1, frame_len)
    return np.sqrt(np.mean(frames * frames, axis=1))


def _speech_mask(energy: np.ndarray) -> np.ndarray:
    threshold = float(np.percentile(energy, ENERGY_PERCENTILE))
    threshold = max(threshold, 1e-4)
    raw = energy > threshold
    mask = raw.copy()
    hangover = 0
    for i, voiced in enumerate(raw):
        if voiced:
            hangover = HANGOVER_FRAMES
            mask[i] = True
        elif hangover > 0:
            hangover -= 1
            mask[i] = True
        else:
            mask[i] = False
    return mask


def _mask_to_turns(mask: np.ndarray, frame_s: float, channel: int) -> list[dict]:
    turns = []
    start = None
    for i, spoken in enumerate(mask):
        if spoken and start is None:
            start = i
        elif not spoken and start is not None:
            end = i
            if (end - start) * frame_s >= MIN_TURN_S:
                turns.append({
                    "channel": channel,
                    "start": round(start * frame_s, 4),
                    "end": round(end * frame_s, 4),
                })
            start = None
    if start is not None:
        end = len(mask)
        if (end - start) * frame_s >= MIN_TURN_S:
            turns.append({
                "channel": channel,
                "start": round(start * frame_s, 4),
                "end": round(end * frame_s, 4),
            })
    return turns


def extract_turns(audio_path: str) -> dict:
    audio, sample_rate = sf.read(audio_path, always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)

    frame_len = max(int(sample_rate * FRAME_MS / 1000), 1)
    frame_s = frame_len / float(sample_rate)
    turns = []
    for channel in (0, 1):
        samples = np.asarray(audio[:, channel], dtype=np.float64)
        energy = _frame_energy(samples, frame_len)
        mask = _speech_mask(energy)
        turns.extend(_mask_to_turns(mask, frame_s, channel))

    turns.sort(key=lambda t: (t["start"], t["channel"]))
    return {"turns": turns}


def _validated_turns(payload: dict) -> list[dict]:
    turns = payload.get("turns")
    if not isinstance(turns, list):
        raise ValueError("Turns payload must contain a list.")
    validated = []
    for turn in turns:
        channel = int(turn["channel"])
        start = float(turn["start"])
        end = float(turn["end"])
        if channel not in {0, 1} or not np.isfinite([start, end]).all() or end <= start:
            raise ValueError(f"Invalid turn: {turn}")
        validated.append({"channel": channel, "start": start, "end": end})
    return sorted(validated, key=lambda item: (item["start"], item["channel"]))


def timing_features(turns_payload: dict) -> dict[str, float]:
    turns = _validated_turns(turns_payload)
    caller_turns = [turn for turn in turns if turn["channel"] == 0]
    agent_turns = [turn for turn in turns if turn["channel"] == 1]

    durations = [turn["end"] - turn["start"] for turn in caller_turns]
    pauses = [
        caller_turns[index + 1]["start"] - caller_turns[index]["end"]
        for index in range(len(caller_turns) - 1)
    ]
    latencies = []
    for caller in caller_turns:
        previous_agent_ends = [
            agent["end"] for agent in agent_turns if agent["end"] <= caller["start"]
        ]
        if previous_agent_ends:
            latencies.append(caller["start"] - max(previous_agent_ends))

    overlap_durations = []
    for caller in caller_turns:
        for agent in agent_turns:
            overlap = min(caller["end"], agent["end"]) - max(
                caller["start"], agent["start"]
            )
            if overlap > 0:
                overlap_durations.append(overlap)

    turn_count = len(caller_turns)
    total_overlap = float(sum(overlap_durations))
    base = {
        "numero_turnos_caller": float(turn_count),
        "duracion_promedio_caller": float(np.mean(durations)) if durations else 0.0,
        "pausa_promedio_caller": float(np.mean(pauses)) if pauses else 0.0,
        "desviacion_estandar_latencia": (
            float(statistics.stdev(latencies)) if len(latencies) > 1 else 0.0
        ),
        "interrupciones_agente": float(len(overlap_durations)),
        "duracion_total_interrupciones": total_overlap,
        "duracion_promedio_interrupcion": (
            total_overlap / len(overlap_durations) if overlap_durations else 0.0
        ),
    }
    epsilon = 1e-5
    base.update(
        {
            "tasa_interrupciones": base["interrupciones_agente"]
            / (turn_count + epsilon),
            "duracion_interrupciones_por_turno": total_overlap
            / (turn_count + epsilon),
            "ratio_habla_pausa": base["duracion_promedio_caller"]
            / (base["pausa_promedio_caller"] + epsilon),
        }
    )
    return base


def _caller_samples(
    audio_path: str | Path,
    turns_payload: dict,
) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
    caller = audio[:, 0]
    segments = []
    for turn in _validated_turns(turns_payload):
        if turn["channel"] != 0:
            continue
        start = max(0, int(turn["start"] * sample_rate))
        end = min(len(caller), int(turn["end"] * sample_rate))
        if end > start:
            segments.append(caller[start:end])
    if not segments:
        raise ValueError("No caller speech was found.")
    return np.concatenate(segments), int(sample_rate)


def voice_features(audio_path: str | Path, turns_payload: dict) -> dict[str, float]:
    caller, sample_rate = _caller_samples(audio_path, turns_payload)
    f0 = librosa.yin(caller, fmin=70, fmax=400, sr=sample_rate)
    valid_f0 = f0[np.isfinite(f0) & (f0 > 0)]

    pitch_delta_std = (
        float(np.std(np.abs(np.diff(valid_f0)))) if len(valid_f0) > 1 else 0.0
    )
    if len(valid_f0) > 1:
        periods = 1.0 / valid_f0
        mean_period = float(np.mean(periods))
        jitter = (
            float(np.mean(np.abs(np.diff(periods))) / mean_period)
            if mean_period > 0
            else 0.0
        )
    else:
        jitter = 0.0

    if len(caller) >= 1024:
        frames = librosa.util.frame(caller, frame_length=1024, hop_length=256)
        amplitudes = np.mean(np.abs(frames), axis=0)
        mean_amplitude = float(np.mean(amplitudes))
        shimmer = (
            float(np.mean(np.abs(np.diff(amplitudes))) / mean_amplitude)
            if len(amplitudes) > 1 and mean_amplitude > 0
            else 0.0
        )
    else:
        shimmer = 0.0

    mfcc = librosa.feature.mfcc(y=caller, sr=sample_rate, n_mfcc=13)
    result = {
        "mfcc_1_std": float(np.std(mfcc[0])),
        "shimmer": shimmer,
        "pitch_delta_std": pitch_delta_std,
        "jitter": jitter,
    }
    if not np.isfinite(list(result.values())).all():
        raise ValueError(f"Non-finite voice features for {audio_path}")
    return result


def extract_features(audio_path: str | Path) -> tuple[dict, dict]:
    turns_payload = extract_turns(str(audio_path))
    if not turns_payload["turns"]:
        raise ValueError(f"No speech turns found: {audio_path}")
    return (
        timing_features(turns_payload),
        voice_features(audio_path, turns_payload),
    )
