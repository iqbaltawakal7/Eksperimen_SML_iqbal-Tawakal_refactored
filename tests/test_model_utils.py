"""Unit test dasar untuk model_utils.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Membangun_model"))
from model_utils import is_classification_target, prepare_features_and_target


def test_is_classification_target_for_categorical_string():
    y = pd.Series(["a", "b", "a", "b"])
    assert is_classification_target(y) is True


def test_is_classification_target_for_continuous_numeric():
    y = pd.Series(range(50))
    assert is_classification_target(y) is False


def test_prepare_features_and_target_with_explicit_target():
    df = pd.DataFrame({"fitur": [1, 2, 3], "kategori": ["a", "b", "a"], "target": [10, 20, 30]})
    X, y = prepare_features_and_target(df, target_col="target")
    assert "target" not in X.columns
    assert list(y) == [10, 20, 30]


def test_prepare_features_and_target_raises_for_unknown_column():
    df = pd.DataFrame({"fitur": [1, 2], "target": [1, 0]})
    with pytest.raises(ValueError):
        prepare_features_and_target(df, target_col="kolom_tidak_ada")
