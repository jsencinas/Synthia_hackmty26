import numpy as np

from src.features import TIMING_FEATURES, timing_features
from src.json_stage import run


class _TimingModel:
    def predict_proba(self, frame):
        assert list(frame.columns) == TIMING_FEATURES
        return np.array([[0.2, 0.8]])


def test_timing_feature_schema(mini_turns_payload):
    features = timing_features(mini_turns_payload)
    assert list(features) == TIMING_FEATURES
    assert features["numero_turnos_caller"] == 2


def test_timing_stage_uses_shared_features(monkeypatch, mini_turns_payload):
    monkeypatch.setattr("src.json_stage.load_timing_model", lambda: _TimingModel())
    monkeypatch.setattr("src.json_stage.load_temperature", lambda: 1.0)
    result = run(mini_turns_payload)
    assert result.error is None
    assert result.p_synthetic == 0.8
    assert list(result.features) == TIMING_FEATURES


def test_missing_model_returns_error(monkeypatch, mini_turns_payload):
    def missing():
        raise FileNotFoundError

    monkeypatch.setattr("src.json_stage.load_timing_model", missing)
    result = run(mini_turns_payload)
    assert result.error == "timing_model_missing"


def test_missing_turns_returns_error():
    assert run(None).error == "timing_turns_missing"
