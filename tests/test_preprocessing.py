"""Unit test dasar untuk fungsi preprocessing."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from importlib import import_module

preprocessing_mod = import_module("automate_Nama-siswa".replace("-", "_")) if False else None
# Modul punya tanda hubung di nama file sehingga tidak bisa di-import langsung
# dengan `import`; pakai importlib.util agar tetap bisa diuji.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "automate_preprocessing",
    Path(__file__).resolve().parents[1] / "preprocessing" / "automate_Nama-siswa.py",
)
automate_preprocessing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(automate_preprocessing)


def test_clean_and_preprocess_fills_missing_numeric_values():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", "z"]})
    result = automate_preprocessing.clean_and_preprocess(df)
    assert result["a"].isnull().sum() == 0
    assert result["a"].iloc[1] == pytest.approx(2.0)


def test_clean_and_preprocess_drops_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    result = automate_preprocessing.clean_and_preprocess(df)
    assert len(result) == 2


def test_load_data_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        automate_preprocessing.load_data("file_yang_tidak_ada.csv")
