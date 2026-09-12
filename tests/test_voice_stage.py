import numpy as np

from src.features import VOICE_FEATURES, voice_features
from src.voice_stage import run


class _VoiceModel:
    def predict_proba(self, frame):
        assert list(frame.columns) == VOICE_FEATURES
        return np.array([[0.7, 0.3]])


def test_voice_stage_uses_shared_features(
    monkeypatch,
    mini_audio_path,
    mini_turns_payload,
):
    monkeypatch.setattr("src.voice_stage.load_voice_model", lambda: _VoiceModel())
    result = run(mini_audio_path, mini_turns_payload)
    assert result.error is None
    assert result.p_synthetic == 0.3
    expected = voice_features(mini_audio_path, mini_turns_payload)
    assert result.features == expected


def test_voice_rejects_missing_inputs(mini_turns_payload):
    assert run("missing.wav", mini_turns_payload).error == "voice_audio_missing"
