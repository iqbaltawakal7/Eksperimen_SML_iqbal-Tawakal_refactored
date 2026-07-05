"""Model training dengan hyperparameter tetap, kombinasi autolog + manual MLflow logging.

Menggunakan mlflow.autolog() supaya folder artefak model yang dihasilkan
mengikuti struktur standar MLflow (MLmodel, conda.yaml, model.pkl,
python_env.yaml, requirements.txt, estimator.html, metric_info.json,
training_confusion_matrix.png) sesuai kriteria submission "skilled".
"""

import logging

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from model_utils import (
    build_model,
    is_classification_target,
    load_dataset,
    prepare_features_and_target,
    split_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "namadataset_preprocessing/dataset_clean.csv"
TARGET_COLUMN = None  # TODO: ganti dengan nama kolom target yang eksplisit
N_ESTIMATORS = 120
MAX_DEPTH = 10


def run_tuning_modelling(
    data_path: str = DATA_PATH,
    target_col: str | None = TARGET_COLUMN,
    n_estimators: int = N_ESTIMATORS,
    max_depth: int = MAX_DEPTH,
) -> None:
    logger.info("=== STARTING MLFLOW TUNING WITH AUTOLOG + MANUAL LOGGING ===")
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("Eksperimen_SML_Tuning")
    mlflow.autolog()  # menghasilkan estimator.html, metric_info.json, confusion matrix, dll.

    df = load_dataset(data_path)
    X, y = prepare_features_and_target(df, target_col)
    X_train, X_test, y_train, y_test = split_data(X, y)

    with mlflow.start_run(run_name="Tuned_Model_Autolog_Manual"):
        # Autolog sudah mencatat n_estimators & max_depth otomatis dari parameter model,
        # tapi tetap dicatat manual juga agar eksplisit terlihat di UI.
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = build_model(y_train, n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if is_classification_target(y_train):
            acc = accuracy_score(y_test, preds)
            logger.info("Tuned Model -> Accuracy: %.4f", acc)
            mlflow.log_metric("accuracy_manual", acc)
        else:
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            logger.info("Tuned Model -> R2: %.4f, MSE: %.4f", r2, mse)
            mlflow.log_metric("mse_manual", mse)
            mlflow.log_metric("r2_score_manual", r2)


if __name__ == "__main__":
    run_tuning_modelling()
