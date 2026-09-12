import pandas as pd
import pytest

from src.data import assert_no_cross_split_duplicates, load_split
from src.train_serving import build_dataset


def test_manifest_splits_are_isolated():
    train = load_split("train")
    val = load_split("val")
    assert set(train["split"]) == {"train"}
    assert set(val["split"]) == {"val"}
    assert set(train["anon_id"]).isdisjoint(val["anon_id"])


def test_raw_audio_is_not_duplicated_across_splits():
    assert_no_cross_split_duplicates(load_split("train"), load_split("val"))


def test_training_dataset_rejects_validation_before_extraction(monkeypatch):
    rows = pd.DataFrame(
        [{"anon_id": "call_val", "label": "human", "split": "val", "duration_s": 1}]
    )
    monkeypatch.setattr(
        "src.train_serving.extract_features",
        lambda path: (_ for _ in ()).throw(AssertionError("VAL was extracted")),
    )
    with pytest.raises(ValueError, match="train rows only"):
        build_dataset(rows)
