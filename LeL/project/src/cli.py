"""CLI unificada para operar o projeto (treino, drift, simulação de tráfego)."""
from __future__ import annotations

import json
import random
import time

import httpx
import typer
from rich import print as rprint

from src.config import settings
from src.data.load import load_raw
from src.data.preprocess import split_and_save
from src.models.promote import promote_latest
from src.models.train import TrainConfig, train
from src.monitoring.drift import run_drift_report

app = typer.Typer(help="MLOps LeL — utilitários CLI")


@app.command()
def prepare():
    """Carrega dataset bruto e gera splits train/test/reference."""
    df = load_raw()
    paths = split_and_save(df)
    rprint({"shape": df.shape, **paths})


@app.command()
def train_model(
    n_estimators: int = 200,
    max_depth: int = 8,
    min_samples_split: int = 2,
    register: bool = True,
):
    """Treina o modelo e (opcional) registra no MLflow Model Registry."""
    cfg = TrainConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )
    result = train(cfg, register=register)
    rprint(result)


@app.command()
def promote(stage: str = "Production"):
    """Promove a última versão para o stage indicado."""
    rprint(promote_latest(stage=stage))


@app.command()
def drift():
    """Roda relatório de data drift (Evidently)."""
    rprint(run_drift_report())


@app.command()
def simulate(
    n: int = 100,
    drift_factor: float = 1.0,
    delay_ms: int = 50,
    url: str = "http://localhost:8000/predict",
):
    """Envia chamadas à API a partir do test set, opcionalmente induzindo drift.

    drift_factor > 1.0 escala features numéricas para forçar drift detectável.
    """
    import pandas as pd

    from src.config import PROCESSED_DIR
    from src.features.build_features import feature_names

    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    feats = feature_names(test_df)
    rows = test_df[feats].sample(n=n, replace=True, random_state=random.randint(1, 1_000_000))

    if drift_factor != 1.0:
        rows = rows.copy()
        rows[feats] = rows[feats] * drift_factor

    ok = 0
    errs = 0
    with httpx.Client(timeout=10.0) as client:
        for _, row in rows.iterrows():
            payload = row.to_dict()
            try:
                r = client.post(url, json=payload)
                if r.status_code == 200:
                    ok += 1
                else:
                    errs += 1
                    rprint(f"[yellow]{r.status_code}[/yellow] {r.text[:120]}")
            except Exception as e:
                errs += 1
                rprint(f"[red]erro:[/red] {e}")
            if delay_ms:
                time.sleep(delay_ms / 1000)
    rprint({"sent": n, "ok": ok, "errors": errs, "drift_factor": drift_factor})


@app.command("info")
def info_cmd():
    """Mostra configuração efetiva."""
    rprint(json.loads(json.dumps(settings.__dict__, default=str)))


if __name__ == "__main__":
    app()
