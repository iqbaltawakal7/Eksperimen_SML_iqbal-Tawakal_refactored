# Membangun model

- `model_utils.py` — logic bersama (load data, split, deteksi tipe target) dipakai oleh kedua skrip di bawah.
- `modelling.py` — baseline model dengan MLflow autolog.
- `modelling_tuning.py` — model dengan hyperparameter tetap + manual MLflow logging.

Jalankan salah satu: `python modelling.py` atau `python modelling_tuning.py`.
Hasil tracking MLflow tersimpan di `mlruns/` (tidak di-commit ke git).
