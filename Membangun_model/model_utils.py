"""
Modul bersama untuk logic yang dipakai ulang oleh modelling.py dan
modelling_tuning.py: memuat data, memisahkan fitur/target, dan
menentukan apakah task-nya klasifikasi atau regresi.

Menyatukan logic ini di satu tempat supaya perubahan (mis. threshold
deteksi tipe target, cara split data) cukup dilakukan sekali dan
konsisten di semua skrip modelling.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Jumlah nilai unik maksimum agar target numerik masih dianggap kategori
# (klasifikasi) alih-alih kontinu (regresi).
CLASSIFICATION_UNIQUE_THRESHOLD = 10


def load_dataset(data_path: str) -> pd.DataFrame:
    """Memuat dataset bersih hasil preprocessing. Melempar error yang jelas jika file tidak ada."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data tidak ditemukan di: {path}")
    logger.info("Membaca data dari %s", path)
    return pd.read_csv(path)


def is_classification_target(y: pd.Series) -> bool:
    """Menentukan apakah target sebaiknya diperlakukan sebagai klasifikasi."""
    return y.dtype == "object" or y.nunique() < CLASSIFICATION_UNIQUE_THRESHOLD


def prepare_features_and_target(
    df: pd.DataFrame, target_col: str | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Memisahkan fitur (X) dan target (y), lalu meng-encode fitur kategorikal.

    Jika target_col tidak diberikan, secara default memakai kolom terakhir
    (perilaku lama) — tapi sebaiknya selalu diberikan secara eksplisit agar
    tidak bergantung pada urutan kolom dataset.
    """
    if target_col is None:
        target_col = df.columns[-1]
        logger.warning(
            "target_col tidak diberikan, menggunakan kolom terakhir '%s' sebagai target. "
            "Sebaiknya set target_col secara eksplisit.",
            target_col,
        )
    elif target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Wrapper tipis di atas train_test_split supaya parameter default konsisten di semua skrip."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_model(y_train: pd.Series, **model_kwargs):
    """Membuat model yang sesuai (classifier/regressor) berdasarkan tipe target."""
    if is_classification_target(y_train):
        logger.info("Target berupa kategori/klasifikasi -> RandomForestClassifier")
        return RandomForestClassifier(**model_kwargs)
    logger.info("Target berupa angka kontinu -> RandomForestRegressor")
    return RandomForestRegressor(**model_kwargs)
