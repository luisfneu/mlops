"""Configuração centralizada — lê variáveis de ambiente com defaults sensatos."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
REPORTS_DIR = ROOT_DIR / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, REFERENCE_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_s3_endpoint: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "wine-quality")
    model_name: str = os.getenv("MODEL_NAME", "wine-classifier")
    model_stage: str = os.getenv("MODEL_STAGE", "Production")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    aws_access_key: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    aws_secret_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    bucket_name: str = os.getenv("MLFLOW_BUCKET_NAME", "mlflow-artifacts")
    random_state: int = 42


settings = Settings()
