from src.features import (
    TIMING_FEATURES,
    VOICE_FEATURES,
    extract_features,
    timing_features,
    voice_features,
)


def test_combined_extractor_matches_stage_extractors(
    monkeypatch,
    mini_audio_path,
    mini_turns_payload,
):
    monkeypatch.setattr(
        "src.features.extract_turns",
        lambda path: mini_turns_payload,
    )
    timing, voice = extract_features(mini_audio_path)
    assert timing == timing_features(mini_turns_payload)
    assert voice == voice_features(mini_audio_path, mini_turns_payload)
    assert list(timing) == TIMING_FEATURES
    assert list(voice) == VOICE_FEATURES


def test_voice_extractor_handles_mono_audio(
    monkeypatch,
    tmp_path,
    mini_turns_payload,
):
    import numpy as np
    import soundfile as sf

    path = tmp_path / "mono.wav"
    time = np.arange(8000 * 7) / 8000
    sf.write(path, 0.1 * np.sin(2 * np.pi * 180 * time), 8000)
    features = voice_features(path, mini_turns_payload)
    assert set(features) == set(VOICE_FEATURES)
    assert all(np.isfinite(value) for value in features.values())
