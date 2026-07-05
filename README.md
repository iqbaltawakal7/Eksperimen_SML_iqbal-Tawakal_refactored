# Eksperimen SML — Iqbal Tawakal

Pipeline machine learning end-to-end: preprocessing data -> training model
(dengan MLflow) -> CI otomatis -> monitoring & logging.

## Struktur folder

1. `preprocessing/` — pembersihan data mentah menjadi data siap latih.
2. `Membangun_model/` — training model (baseline & tuning) dengan MLflow tracking lokal.
3. `Workflow-CI/` — MLflow Project yang sama, dikemas agar bisa dijalankan otomatis di CI (mis. GitHub Actions).
4. `Monitoring dan Logging/` — serving model tiruan, exporter metrik Prometheus, dan konfigurasi Grafana.
5. `tests/` — unit test untuk fungsi preprocessing dan modelling.

## Alur menjalankan pipeline secara lokal

```bash
pip install -r requirements.txt

# 1. Preprocessing
python preprocessing/automate_Nama-siswa.py

# 2. Training (baseline + tuning)
cd Membangun_model
python modelling.py
python modelling_tuning.py

# 3. Jalankan test
pytest tests/
```

## Catatan penting

- **Target column**: skrip modelling saat ini default memakai kolom terakhir
  dataset sebagai target jika `target_col` tidak diset eksplisit. Sebaiknya
  selalu set `target_col` sesuai dataset Anda (lihat `TARGET_COLUMN` di
  `Membangun_model/modelling.py`).
- **Monitoring**: metrik di `3.prometheus_exporter.py` masih berupa simulasi
  (angka acak), bukan pengukuran nyata — lihat catatan di file tersebut.
- `dicoding_env/`, `mlruns/`, dan file `*.pkl`/`*.skops` sengaja tidak
  disertakan di git (lihat `.gitignore`) karena merupakan artefak lokal yang
  regenerated saat pipeline dijalankan.
