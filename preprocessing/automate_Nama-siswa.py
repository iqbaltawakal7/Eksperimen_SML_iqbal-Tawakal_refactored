"""Otomatisasi preprocessing: load -> clean -> save."""

import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE = "namadataset_raw/dataset_pelatihan.csv"
OUTPUT_FOLDER = "preprocessing/namadataset_preprocessing"
OUTPUT_FILE = "dataset_clean.csv"


def load_data(file_path: str) -> pd.DataFrame:
    """Membaca dataset mentah."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {file_path}")
    logger.info("[1/3] Membaca data dari %s...", file_path)
    return pd.read_csv(file_path)


def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Fungsi pusat untuk semua logic preprocessing."""
    logger.info("[2/3] Menjalankan proses pembersihan data...")
    df_clean = df.copy()

    # Handling missing values: isi kolom numerik dengan mean
    kolom_numerik = df_clean.select_dtypes(include=[np.number]).columns
    for col in kolom_numerik:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    # Handling duplicates
    if df_clean.duplicated().sum() > 0:
        n_before = len(df_clean)
        df_clean.drop_duplicates(inplace=True)
        logger.info("    -> %d baris duplikat dihapus.", n_before - len(df_clean))

    return df_clean


def save_data(df: pd.DataFrame, output_folder: str, filename: str) -> str:
    """Membuat folder output (jika belum ada) dan menyimpan data hasil preprocessing."""
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    df.to_csv(output_path, index=False)
    logger.info("[3/3] Data sukses diproses & disimpan di: %s", output_path)
    return output_path


def run_pipeline(input_file: str = INPUT_FILE, output_folder: str = OUTPUT_FOLDER, output_file: str = OUTPUT_FILE) -> str:
    logger.info("=== MEMULAI AUTOMATISASI PREPROCESSING ===")
    raw_data = load_data(input_file)
    cleaned_data = clean_and_preprocess(raw_data)
    output_path = save_data(cleaned_data, output_folder, output_file)
    logger.info("=== PROSES SELESAI DENGAN SUKSES ===")
    return output_path


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as exc:
        logger.error("Terjadi kesalahan: %s", exc)
