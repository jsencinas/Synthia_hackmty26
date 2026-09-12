from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import soundfile as sf


FRAME_MS = 30
HANGOVER_FRAMES = 8
MIN_TURN_S = 0.15
ENERGY_PERCENTILE = 60


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


def write_temp_json(audio_path: str) -> str:
    payload = extract_turns(audio_path)
    if not payload["turns"]:
        raise ValueError("empty_vad_turns")

    stem = os.path.splitext(os.path.basename(audio_path))[0] or "call"
    fd, path = tempfile.mkstemp(prefix=f"{stem}_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path
