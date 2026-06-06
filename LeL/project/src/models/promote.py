"""Promoção de versão no MLflow Model Registry (Staging/Production)."""
from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from src.config import settings


def promote_latest(stage: str = "Production", archive_existing: bool = True) -> dict:
    """Promove a versão mais recente do modelo para o stage indicado."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{settings.model_name}'")
    if not versions:
        raise RuntimeError(f"Nenhuma versão encontrada para modelo {settings.model_name}")

    latest = max(versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=settings.model_name,
        version=latest.version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    return {"name": settings.model_name, "version": latest.version, "stage": stage}


if __name__ == "__main__":
    print(promote_latest())
