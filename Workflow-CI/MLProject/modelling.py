"""Entry point training untuk MLflow Project (dipanggil via `mlflow run`)."""

import argparse
import logging

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, r2_score

from model_utils import build_model, is_classification_target, load_dataset, prepare_features_and_target, split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_basic_modelling(data_path: str, target_col: str | None = None) -> None:
    logger.info("=== STARTING MLFLOW BASIC MODELLING WITH AUTOLOG ===")
    mlflow.autolog()

    df = load_dataset(data_path)
    X, y = prepare_features_and_target(df, target_col)
    X_train, X_test, y_train, y_test = split_data(X, y)

    with mlflow.start_run(run_name="Baseline_Model_Autolog"):
        model = build_model(y_train, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if is_classification_target(y_train):
            logger.info("Baseline Accuracy Score: %.4f", accuracy_score(y_test, preds))
        else:
            logger.info("Baseline R2 Score: %.4f", r2_score(y_test, preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="namadataset_preprocessing/dataset_clean.csv")
    parser.add_argument("--target_col", type=str, default=None)
    args = parser.parse_args()

    run_basic_modelling(args.data_path, args.target_col or None)
