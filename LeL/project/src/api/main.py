"""FastAPI para servir o modelo com métricas Prometheus + log de predições para drift."""
from __future__ import annotations

import csv
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.model_loader import LoadedModel, load_model
from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PredictionResponse,
    WineFeatures,
)
from src.config import REPORTS_DIR, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("api")

PREDICTION_LOG = REPORTS_DIR / "predictions_log.csv"

PREDICTIONS_TOTAL = Counter(
    "ml_predictions_total",
    "Total de predições atendidas",
    ["model_name", "model_version", "predicted_class"],
)
PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Latência da inferência (apenas predict)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTION_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Probabilidade máxima da predição (proxy de confiança)",
    buckets=tuple(i / 10 for i in range(1, 11)),
)
MODEL_LOAD_FAILURES = Counter("ml_model_load_failures_total", "Falhas ao carregar o modelo")


class State:
    loaded: LoadedModel | None = None


def _append_log(features: dict, prediction: int, probabilities: list[float]) -> None:
    new_file = not PREDICTION_LOG.exists()
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **features,
        "prediction": prediction,
        "proba_max": max(probabilities),
    }
    with PREDICTION_LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


@asynccontextmanager
async def lifespan(app: FastAPI):
    State.loaded = load_model()
    if State.loaded is None:
        MODEL_LOAD_FAILURES.inc()
        logger.warning("API subiu sem modelo — /predict retornará 503 até modelo estar disponível")
    yield


app = FastAPI(
    title="MLOps LeL — Wine Classifier API",
    version="0.1.0",
    description="API de inferência servindo modelo registrado no MLflow Registry.",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def _to_df(features: WineFeatures) -> pd.DataFrame:
    data = features.model_dump(by_alias=True)
    return pd.DataFrame([data])


def _predict_one(features: WineFeatures) -> PredictionResponse:
    if State.loaded is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    df = _to_df(features)
    with PREDICTION_LATENCY.time():
        raw = State.loaded.model.predict(df)

    pred = int(np.asarray(raw).ravel()[0])
    try:
        sk_model = State.loaded.model._model_impl.sklearn_model
        proba = sk_model.predict_proba(df)[0].tolist()
    except Exception:
        proba = [1.0 if i == pred else 0.0 for i in range(3)]

    PREDICTIONS_TOTAL.labels(
        model_name=settings.model_name,
        model_version=State.loaded.version,
        predicted_class=str(pred),
    ).inc()
    PREDICTION_CONFIDENCE.observe(max(proba))

    _append_log(df.iloc[0].to_dict(), pred, proba)

    return PredictionResponse(
        prediction=pred,
        probabilities=proba,
        model_version=State.loaded.version,
        model_stage=State.loaded.stage,
    )


@app.get("/", tags=["meta"])
def root():
    return {
        "name": "MLOps LeL — Wine Classifier API",
        "docs": "/docs",
        "metrics": "/metrics",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    loaded = State.loaded is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=settings.model_name,
        model_version=State.loaded.version if loaded else None,
    )


@app.post("/reload", tags=["meta"])
def reload_model():
    State.loaded = load_model()
    if State.loaded is None:
        MODEL_LOAD_FAILURES.inc()
        raise HTTPException(status_code=503, detail="Falha ao recarregar modelo")
    return {"reloaded": True, "version": State.loaded.version, "stage": State.loaded.stage}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(features: WineFeatures):
    return _predict_one(features)


@app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch(batch: BatchRequest):
    if not batch.instances:
        raise HTTPException(status_code=400, detail="instances vazio")
    return BatchResponse(predictions=[_predict_one(x) for x in batch.instances])
