# Workflow CI

`MLProject/` adalah MLflow Project yang self-contained (environment sendiri
via `conda.yaml`) supaya bisa dijalankan otomatis oleh CI (mis. GitHub
Actions) tanpa bergantung pada environment lokal developer.

Jalankan manual: `mlflow run MLProject/`

`model_utils.py` di folder ini adalah salinan dari
`Membangun_model/model_utils.py` — lihat catatan sinkronisasi di file
tersebut.
