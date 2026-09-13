"""Speaker turns from WAV energy, then timing and voice features for the models.

The whole analysis of one call is done from a single decoded audio buffer
(``CallAudio``) so the WAV is read from disk exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


# --- Voice activity detection -------------------------------------------------
# Adaptive energy VAD: a frame is speech when its RMS level (dB) is more than
# ``VAD_MARGIN_DB`` above the channel noise floor (a low percentile of frame
# levels). Parameters were fitted against the organiser-provided reference turns
# (frame IoU 0.95 caller / 0.97 agent).
FRAME_MS = 20
VAD_FLOOR_PERCENTILE = 20
VAD_MARGIN_DB = 12.0
VAD_HANGOVER_S = 0.1
VAD_MIN_SILENCE_S = 0.2
VAD_MIN_SPEECH_S = 0.3

# Voice features are computed on at most this much caller speech (latency cap).
MAX_VOICE_SECONDS = 90.0

_STAT_SUFFIXES = ("mean", "std", "median", "max", "min", "cv")


def _stat_names(prefix: str) -> list[str]:
    return [f"{prefix}_{suffix}" for suffix in _STAT_SUFFIXES]


TIMING_FEATURES = (
    [
        "n_caller_turns",
        "n_agent_turns",
        "turn_ratio",
    ]
    + _stat_names("caller_dur")
    + [
        "agent_dur_mean",
        "caller_speech_frac",
        "agent_speech_frac",
        "caller_short_turn_frac",
        "caller_long_turn_frac",
        "caller_turns_per_min",
    ]
    + _stat_names("resp_lat")
    + [
        "resp_lat_n",
        "resp_lat_frac_fast",
        "resp_lat_frac_slow",
        "resp_lat_iqr",
    ]
    + _stat_names("agent_lat")
    + _stat_names("caller_pause")
    + [
        "caller_pause_n",
        "overlap_n",
        "overlap_total",
        "overlap_mean",
        "overlap_rate",
        "overlap_per_min",
        "caller_interrupt_frac",
        "duration",
    ]
)

VOICE_FEATURES = (
    [
        # channel level / noise floor / cross-talk
        "noise_floor_db",
        "speech_level_db",
        "snr_db",
        "db_std_all",
        "nonspeech_db_mean",
        "nonspeech_db_std",
        "nonspeech_db_p5",
        "speech_db_std",
        "zero_sample_frac",
        "min_abs_nonzero",
        "clip_frac",
        "peak_abs",
        "crosstalk_db",
        "xcorr_db_agent",
        # pitch
        "f0_voiced_frac",
        "f0_mean",
        "f0_std",
        "f0_cv",
        "f0_range",
        "pitch_delta_std",
        "pitch_delta_mean",
        "pitch_delta_median",
        "jitter",
        # amplitude envelope
        "shimmer",
        "amp_cv",
        # spectrum
        "band_lo_ratio",
        "band_mid_ratio",
        "band_hi_ratio",
        "band_vhi_ratio",
        "band_vhi_ratio_log",
        "centroid_mean",
        "centroid_std",
        "bandwidth_mean",
        "bandwidth_std",
        "rolloff_mean",
        "rolloff_std",
        "flatness_mean",
        "flatness_std",
        "zcr_mean",
        "zcr_std",
    ]
    # MFCC spread only (means are speaker/voice specific and do not transfer
    # to unseen voices).
    + [f"mfcc{i}_std" for i in range(13)]
    + [
        "mfcc_delta_std_mean",
        "onset_rate",
        "onset_strength_cv",
        "flux_mean",
        "flux_cv",
    ]
)


@dataclass
class CallAudio:
    """Decoded stereo call: channel 0 caller, channel 1 agent."""

    samples: np.ndarray  # shape (n, 2), float32
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    @property
    def caller(self) -> np.ndarray:
        return self.samples[:, 0]

    @property
    def agent(self) -> np.ndarray:
        return self.samples[:, 1]


@dataclass
class ChannelActivity:
    frame_db: np.ndarray
    speech_mask: np.ndarray
    noise_floor_db: float
    threshold_db: float


def load_audio(source, target_sample_rate: int = 8000) -> CallAudio:
    """Read a WAV (path or file-like) into a stereo float32 buffer at 8 kHz."""
    audio, sample_rate = sf.read(source, always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    if sample_rate != target_sample_rate:
        audio = np.stack(
            [
                librosa.resample(
                    np.ascontiguousarray(audio[:, ch]),
                    orig_sr=sample_rate,
                    target_sr=target_sample_rate,
                )
                for ch in range(2)
            ],
            axis=1,
        ).astype(np.float32)
        sample_rate = target_sample_rate
    return CallAudio(samples=np.ascontiguousarray(audio), sample_rate=int(sample_rate))


# --- VAD ----------------------------------------------------------------------

def _frame_db(samples: np.ndarray, frame_len: int) -> np.ndarray:
    pad = (-len(samples)) % frame_len
    if pad:
        samples = np.pad(samples, (0, pad))
    frames = samples.reshape(-1, frame_len)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return 20.0 * np.log10(rms + 1e-8)


def _channel_activity(samples: np.ndarray, sample_rate: int) -> tuple[ChannelActivity, list[tuple[float, float]]]:
    frame_len = max(int(sample_rate * FRAME_MS / 1000), 1)
    frame_s = frame_len / float(sample_rate)
    db = _frame_db(np.asarray(samples, dtype=np.float64), frame_len)

    floor = float(np.percentile(db, VAD_FLOOR_PERCENTILE))
    threshold = floor + VAD_MARGIN_DB
    raw = db > threshold

    mask = raw.copy()
    hangover = int(round(VAD_HANGOVER_S / frame_s))
    voiced_idx = np.flatnonzero(raw)
    for k in range(1, hangover + 1):
        shifted = voiced_idx + k
        mask[shifted[shifted < len(mask)]] = True

    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    merged: list[list[int]] = []
    for start, end in zip(starts, ends):
        if merged and (start - merged[-1][1]) * frame_s < VAD_MIN_SILENCE_S:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    segments = []
    final_mask = np.zeros_like(mask)
    for start, end in merged:
        if (end - start) * frame_s >= VAD_MIN_SPEECH_S:
            segments.append((start * frame_s, end * frame_s))
            final_mask[start:end] = True

    return (
        ChannelActivity(
            frame_db=db,
            speech_mask=final_mask,
            noise_floor_db=floor,
            threshold_db=threshold,
        ),
        segments,
    )


def detect_turns(audio: CallAudio) -> tuple[dict, dict[int, ChannelActivity]]:
    turns = []
    activity: dict[int, ChannelActivity] = {}
    for channel in (0, 1):
        act, segments = _channel_activity(audio.samples[:, channel], audio.sample_rate)
        activity[channel] = act
        turns.extend(
            {"channel": channel, "start": round(s, 3), "end": round(e, 3)}
            for s, e in segments
        )
    turns.sort(key=lambda t: (t["start"], t["channel"]))
    return {"turns": turns}, activity


def extract_turns(audio_path: str) -> dict:
    """Turns payload ``{"turns": [{"channel", "start", "end"}, ...]}`` from a WAV."""
    turns_payload, _ = detect_turns(load_audio(audio_path))
    return turns_payload


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


# --- Timing features ----------------------------------------------------------

def _stats(values, prefix: str) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        out = {f"{prefix}_{s}": 0.0 for s in ("mean", "std", "median", "max", "min")}
    else:
        out = {
            f"{prefix}_mean": float(v.mean()),
            f"{prefix}_std": float(v.std()) if len(v) > 1 else 0.0,
            f"{prefix}_median": float(np.median(v)),
            f"{prefix}_max": float(v.max()),
            f"{prefix}_min": float(v.min()),
        }
    out[f"{prefix}_cv"] = out[f"{prefix}_std"] / (abs(out[f"{prefix}_mean"]) + 1e-3)
    return out


def timing_features(turns_payload: dict, duration: float | None = None) -> dict[str, float]:
    turns = _validated_turns(turns_payload)
    caller = [t for t in turns if t["channel"] == 0]
    agent = [t for t in turns if t["channel"] == 1]
    if duration is None:
        duration = max((t["end"] for t in turns), default=0.0)
    duration = max(float(duration), 1e-3)
    minutes = max(duration / 60.0, 1e-3)

    f: dict[str, float] = {}
    f["n_caller_turns"] = float(len(caller))
    f["n_agent_turns"] = float(len(agent))
    f["turn_ratio"] = len(caller) / (len(agent) + 1e-3)

    dur_c = [t["end"] - t["start"] for t in caller]
    dur_a = [t["end"] - t["start"] for t in agent]
    f.update(_stats(dur_c, "caller_dur"))
    f["agent_dur_mean"] = float(np.mean(dur_a)) if dur_a else 0.0
    f["caller_speech_frac"] = sum(dur_c) / duration
    f["agent_speech_frac"] = sum(dur_a) / duration
    f["caller_short_turn_frac"] = float(np.mean([d < 1.0 for d in dur_c])) if dur_c else 0.0
    f["caller_long_turn_frac"] = float(np.mean([d > 8.0 for d in dur_c])) if dur_c else 0.0
    f["caller_turns_per_min"] = len(caller) / minutes

    # Response latency: gap between the end of an agent turn and the start of
    # the first caller turn that follows it (how long the caller "thinks").
    agent_ends = np.array([a["end"] for a in agent])
    caller_starts = np.array([c["start"] for c in caller])
    latencies = []
    for c in caller:
        prev = agent_ends[agent_ends <= c["start"]]
        if len(prev) == 0:
            continue
        last_agent_end = float(prev.max())
        between = np.any((caller_starts >= last_agent_end) & (caller_starts < c["start"]))
        if not between:
            latencies.append(c["start"] - last_agent_end)
    f.update(_stats(latencies, "resp_lat"))
    f["resp_lat_n"] = float(len(latencies))
    f["resp_lat_frac_fast"] = float(np.mean([l < 0.5 for l in latencies])) if latencies else 0.0
    f["resp_lat_frac_slow"] = float(np.mean([l > 2.0 for l in latencies])) if latencies else 0.0
    f["resp_lat_iqr"] = (
        float(np.subtract(*np.percentile(latencies, [75, 25]))) if len(latencies) > 1 else 0.0
    )

    # Agent latency: how fast the agent's endpointing reacts to the caller.
    caller_ends = np.array([c["end"] for c in caller])
    agent_starts = np.array([a["start"] for a in agent])
    agent_lat = []
    for a in agent:
        prev = caller_ends[caller_ends <= a["start"]]
        if len(prev) == 0:
            continue
        last_caller_end = float(prev.max())
        between = np.any((agent_starts >= last_caller_end) & (agent_starts < a["start"]))
        if not between:
            agent_lat.append(a["start"] - last_caller_end)
    f.update(_stats(agent_lat, "agent_lat"))

    # Pauses inside the caller's own speech (no agent speech in the gap).
    pauses = []
    for i in range(len(caller) - 1):
        gap_start, gap_end = caller[i]["end"], caller[i + 1]["start"]
        if not any(a["start"] < gap_end and a["end"] > gap_start for a in agent):
            pauses.append(gap_end - gap_start)
    f.update(_stats(pauses, "caller_pause"))
    f["caller_pause_n"] = float(len(pauses))

    # Overlaps between caller and agent speech.
    overlaps = []
    caller_starts_inside_agent = 0
    for c in caller:
        for a in agent:
            overlap = min(c["end"], a["end"]) - max(c["start"], a["start"])
            if overlap > 0:
                overlaps.append(overlap)
            if a["start"] < c["start"] < a["end"]:
                caller_starts_inside_agent += 1
    f["overlap_n"] = float(len(overlaps))
    f["overlap_total"] = float(sum(overlaps))
    f["overlap_mean"] = float(np.mean(overlaps)) if overlaps else 0.0
    f["overlap_rate"] = len(overlaps) / (len(caller) + 1e-3)
    f["overlap_per_min"] = len(overlaps) / minutes
    f["caller_interrupt_frac"] = caller_starts_inside_agent / (len(caller) + 1e-3)
    f["duration"] = duration

    return {name: _finite(f[name]) for name in TIMING_FEATURES}


# --- Voice features -----------------------------------------------------------

def _finite(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _caller_speech(audio: CallAudio, turns: list[dict]) -> np.ndarray:
    caller = audio.caller
    sr = audio.sample_rate
    segments = []
    for turn in turns:
        if turn["channel"] != 0:
            continue
        start = max(0, int(turn["start"] * sr))
        end = min(len(caller), int(turn["end"] * sr))
        if end > start:
            segments.append(caller[start:end])
    if not segments:
        raise ValueError("No caller speech was found.")
    speech = np.concatenate(segments)
    limit = int(MAX_VOICE_SECONDS * sr)
    return speech[:limit] if len(speech) > limit else speech


def voice_features_from_audio(
    audio: CallAudio,
    turns_payload: dict,
    activity: dict[int, ChannelActivity] | None = None,
) -> dict[str, float]:
    turns = _validated_turns(turns_payload)
    if activity is None:
        _, activity = detect_turns(audio)
    sr = audio.sample_rate
    caller = audio.caller
    act0, act1 = activity[0], activity[1]
    db0, mask0 = act0.frame_db, act0.speech_mask
    db1, mask1 = act1.frame_db, act1.speech_mask

    f: dict[str, float] = {}

    # Channel level statistics: noise floor, SNR, behaviour in the silences.
    f["noise_floor_db"] = act0.noise_floor_db
    f["speech_level_db"] = float(np.percentile(db0, 90))
    f["snr_db"] = f["speech_level_db"] - f["noise_floor_db"]
    f["db_std_all"] = float(db0.std())
    nonspeech = db0[~mask0]
    f["nonspeech_db_mean"] = float(nonspeech.mean()) if len(nonspeech) else act0.noise_floor_db
    f["nonspeech_db_std"] = float(nonspeech.std()) if len(nonspeech) > 1 else 0.0
    f["nonspeech_db_p5"] = float(np.percentile(nonspeech, 5)) if len(nonspeech) else act0.noise_floor_db
    speech_db = db0[mask0]
    f["speech_db_std"] = float(speech_db.std()) if len(speech_db) > 1 else 0.0
    nonzero = caller[caller != 0.0]
    f["zero_sample_frac"] = float(np.mean(caller == 0.0)) if len(caller) else 0.0
    f["min_abs_nonzero"] = float(np.min(np.abs(nonzero))) if len(nonzero) else 0.0
    f["clip_frac"] = float(np.mean(np.abs(caller) > 0.99)) if len(caller) else 0.0
    f["peak_abs"] = float(np.max(np.abs(caller))) if len(caller) else 0.0

    # Cross-talk: does the agent's voice leak into the caller channel?
    n = min(len(db0), len(db1))
    agent_only = mask1[:n] & ~mask0[:n]
    both_silent = ~mask1[:n] & ~mask0[:n]
    if agent_only.sum() > 5 and both_silent.sum() > 5:
        f["crosstalk_db"] = float(db0[:n][agent_only].mean() - db0[:n][both_silent].mean())
    else:
        f["crosstalk_db"] = 0.0
    if mask1[:n].sum() > 5:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(db0[:n][mask1[:n]], db1[:n][mask1[:n]])[0, 1]
        f["xcorr_db_agent"] = float(corr) if np.isfinite(corr) else 0.0
    else:
        f["xcorr_db_agent"] = 0.0

    y = _caller_speech(audio, turns)

    # Pitch (YIN) statistics; frames pinned to the search bounds are unvoiced.
    f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr, frame_length=1024, hop_length=256)
    voiced = f0[np.isfinite(f0) & (f0 > 70.5) & (f0 < 399.5)]
    f["f0_voiced_frac"] = len(voiced) / max(len(f0), 1)
    f["f0_mean"] = float(voiced.mean()) if len(voiced) else 0.0
    f["f0_std"] = float(voiced.std()) if len(voiced) > 1 else 0.0
    f["f0_cv"] = f["f0_std"] / (f["f0_mean"] + 1e-3)
    f["f0_range"] = (
        float(np.percentile(voiced, 90) - np.percentile(voiced, 10)) if len(voiced) > 1 else 0.0
    )
    deltas = np.abs(np.diff(voiced)) if len(voiced) > 1 else np.zeros(1)
    f["pitch_delta_std"] = float(deltas.std())
    f["pitch_delta_mean"] = float(deltas.mean())
    f["pitch_delta_median"] = float(np.median(deltas))
    if len(voiced) > 1:
        periods = 1.0 / voiced
        f["jitter"] = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))
    else:
        f["jitter"] = 0.0

    # Amplitude envelope.
    if len(y) >= 1024:
        frames = librosa.util.frame(y, frame_length=1024, hop_length=256)
        amplitudes = np.mean(np.abs(frames), axis=0)
        f["shimmer"] = float(np.mean(np.abs(np.diff(amplitudes))) / (amplitudes.mean() + 1e-8))
        f["amp_cv"] = float(amplitudes.std() / (amplitudes.mean() + 1e-8))
    else:
        f["shimmer"] = 0.0
        f["amp_cv"] = 0.0

    # Spectrum (one STFT shared by every spectral feature).
    power = np.abs(librosa.stft(y, n_fft=512, hop_length=128)) ** 2
    magnitude = np.sqrt(power)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    total = power.sum(axis=0) + 1e-10

    def band_ratio(low: float, high: float) -> float:
        sel = (freqs >= low) & (freqs < high)
        return float((power[sel].sum(axis=0) / total).mean())

    f["band_lo_ratio"] = band_ratio(0, 300)
    f["band_mid_ratio"] = band_ratio(300, 2000)
    f["band_hi_ratio"] = band_ratio(2000, 3400)
    f["band_vhi_ratio"] = band_ratio(3400, 4001)
    f["band_vhi_ratio_log"] = float(np.log10(f["band_vhi_ratio"] + 1e-9))
    centroid = librosa.feature.spectral_centroid(S=power, sr=sr)[0]
    f["centroid_mean"], f["centroid_std"] = float(centroid.mean()), float(centroid.std())
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
    f["bandwidth_mean"], f["bandwidth_std"] = float(bandwidth.mean()), float(bandwidth.std())
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=0.95)[0]
    f["rolloff_mean"], f["rolloff_std"] = float(rolloff.mean()), float(rolloff.std())
    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    f["flatness_mean"], f["flatness_std"] = float(flatness.mean()), float(flatness.std())
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=128)[0]
    f["zcr_mean"], f["zcr_std"] = float(zcr.mean()), float(zcr.std())

    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(S=power, sr=sr, n_mels=40))
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=13)
    for i in range(13):
        f[f"mfcc{i}_std"] = float(mfcc[i].std())
    f["mfcc_delta_std_mean"] = float(np.diff(mfcc, axis=1).std(axis=1).mean())

    # Speech-rate proxy: onset peaks per second of caller speech.
    onset = librosa.onset.onset_strength(S=mel_db, sr=sr, hop_length=128)
    peaks = librosa.util.peak_pick(
        onset, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=3
    )
    f["onset_rate"] = len(peaks) / (len(y) / sr)
    f["onset_strength_cv"] = float(onset.std() / (onset.mean() + 1e-6))
    flux = np.sqrt((np.diff(magnitude, axis=1) ** 2).sum(axis=0))
    f["flux_mean"] = float(flux.mean())
    f["flux_cv"] = float(flux.std() / (flux.mean() + 1e-8))

    return {name: _finite(f[name]) for name in VOICE_FEATURES}


def voice_features(audio_path: str | Path, turns_payload: dict) -> dict[str, float]:
    return voice_features_from_audio(load_audio(audio_path), turns_payload)


# --- One-shot analysis --------------------------------------------------------

@dataclass
class CallAnalysis:
    turns: dict
    timing: dict[str, float]
    voice: dict[str, float]
    duration: float


def analyze_audio(audio: CallAudio) -> CallAnalysis:
    turns_payload, activity = detect_turns(audio)
    if not turns_payload["turns"]:
        raise ValueError("No speech turns found.")
    timing = timing_features(turns_payload, duration=audio.duration)
    voice = voice_features_from_audio(audio, turns_payload, activity)
    return CallAnalysis(turns=turns_payload, timing=timing, voice=voice, duration=audio.duration)


def analyze_call(audio_path: str | Path) -> CallAnalysis:
    return analyze_audio(load_audio(audio_path))


def extract_features(audio_path: str | Path) -> tuple[dict, dict]:
    analysis = analyze_call(audio_path)
    return analysis.timing, analysis.voice
