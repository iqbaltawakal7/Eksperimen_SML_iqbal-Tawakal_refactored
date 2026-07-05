"""Baseline model training dengan MLflow autolog."""

import logging
import os
import mlflow
import mlflow.sklearn

# Set kredensial DagsHub langsung ke memory environment
os.environ['MLFLOW_TRACKING_USERNAME'] = 'iqbaltawakal7'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'Prolink10'

# Arahkan tracking URI langsung ke repositori online DagsHub kamu
mlflow.set_tracking_uri("https://dagshub.com/iqbaltawakal7/Eksperimen_SML_iqbal-Tawakal.mlflow")
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, r2_score

from model_utils import build_model, load_dataset, prepare_features_and_target, split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "namadataset_preprocessing/dataset_clean.csv"
TARGET_COLUMN = None  # TODO: ganti dengan nama kolom target yang eksplisit, mis. "harga"


def run_basic_modelling(data_path: str = DATA_PATH, target_col: str | None = TARGET_COLUMN) -> None:
    logger.info("=== STARTING MLFLOW BASIC MODELLING WITH AUTOLOG ===")
    mlflow.set_experiment("Eksperimen_SML_Baseline")
    mlflow.autolog()

    df = load_dataset(data_path)
    X, y = prepare_features_and_target(df, target_col)
    X_train, X_test, y_train, y_test = split_data(X, y)

    with mlflow.start_run(run_name="Baseline_Model_Autolog"):
        model = build_model(y_train, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if hasattr(model, "predict_proba") or model.__class__.__name__.endswith("Classifier"):
            score = accuracy_score(y_test, preds)
            logger.info("Baseline Accuracy Score: %.4f", score)
        else:
            score = r2_score(y_test, preds)
            logger.info("Baseline R2 Score: %.4f", score)


if __name__ == "__main__":
    run_basic_modelling()
