"""Carregamento do modelo do MLflow Model Registry com fallback."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    model: Any
    version: str
    stage: str


def load_model() -> LoadedModel | None:
    """Carrega o modelo do MLflow Registry pelo stage configurado."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    try:
        versions = client.get_latest_versions(settings.model_name, stages=[settings.model_stage])
        if not versions:
            logger.warning(
                "Nenhuma versão no stage %s — tentando 'None' (latest sem stage)",
                settings.model_stage,
            )
            versions = client.get_latest_versions(settings.model_name, stages=["None"])
        if not versions:
            logger.error("Modelo %s não encontrado no registry", settings.model_name)
            return None

        v = versions[0]
        uri = f"models:/{settings.model_name}/{v.version}"
        model = mlflow.pyfunc.load_model(uri)
        logger.info("Modelo carregado: %s v%s (%s)", settings.model_name, v.version, v.current_stage)
        return LoadedModel(model=model, version=v.version, stage=v.current_stage)
    except Exception as e:
        logger.exception("Falha ao carregar modelo: %s", e)
        return None
