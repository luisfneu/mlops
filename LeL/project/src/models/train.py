"""Treino com MLflow tracking — registra parâmetros, métricas, modelo e artefatos."""
from __future__ import annotations

import json
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR, settings
from src.features.build_features import split_xy


@dataclass
class TrainConfig:
    n_estimators: int = 200
    max_depth: int | None = 8
    min_samples_split: int = 2
    random_state: int = settings.random_state


def build_pipeline(cfg: TrainConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=cfg.n_estimators,
                    max_depth=cfg.max_depth,
                    min_samples_split=cfg.min_samples_split,
                    random_state=cfg.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    preds = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1_macro": float(f1_score(y, preds, average="macro")),
        "precision_macro": float(precision_score(y, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, preds, average="macro")),
    }


def train(cfg: TrainConfig | None = None, register: bool = True) -> dict:
    """Treina o modelo, loga no MLflow e (opcional) registra no Model Registry."""
    cfg = cfg or TrainConfig()

    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "n_estimators": cfg.n_estimators,
                "max_depth": cfg.max_depth,
                "min_samples_split": cfg.min_samples_split,
                "random_state": cfg.random_state,
                "n_features": X_train.shape[1],
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )

        pipe = build_pipeline(cfg)
        pipe.fit(X_train, y_train)

        train_metrics = {f"train_{k}": v for k, v in evaluate(pipe, X_train, y_train).items()}
        test_metrics = {f"test_{k}": v for k, v in evaluate(pipe, X_test, y_test).items()}
        mlflow.log_metrics({**train_metrics, **test_metrics})

        signature = infer_signature(X_train, pipe.predict(X_train))
        input_example = X_train.head(3)

        log_kwargs = dict(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )
        if register:
            log_kwargs["registered_model_name"] = settings.model_name
        mlflow.sklearn.log_model(**log_kwargs)

        result = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "metrics": {**train_metrics, **test_metrics},
            "model_uri": f"runs:/{run.info.run_id}/model",
        }
        print(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    train()
