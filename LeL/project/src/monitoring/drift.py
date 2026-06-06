"""Detecção de data drift e data quality usando Evidently.

Gera relatórios HTML + JSON comparando dados de referência (treino) vs.
dados "atuais" (predições logadas pela API, ou um teste).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report

from src.config import PROCESSED_DIR, REFERENCE_DIR, REPORTS_DIR
from src.features.build_features import TARGET, feature_names

PREDICTION_LOG = REPORTS_DIR / "predictions_log.csv"


def _load_reference() -> pd.DataFrame:
    ref_path = REFERENCE_DIR / "reference.parquet"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference dataset ausente: {ref_path}. Rode o training flow primeiro.")
    return pd.read_parquet(ref_path)


def _load_current() -> pd.DataFrame:
    """Tenta carregar predições logadas pela API; cai para test set se vazio."""
    if PREDICTION_LOG.exists() and PREDICTION_LOG.stat().st_size > 0:
        df = pd.read_csv(PREDICTION_LOG)
        cols_to_drop = [c for c in ("timestamp", "proba_max", "prediction") if c in df.columns]
        return df.drop(columns=cols_to_drop)
    test_path = PROCESSED_DIR / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError("Sem dados 'atuais' — rode o pipeline e/ou faça chamadas à API.")
    return pd.read_parquet(test_path).drop(columns=[TARGET], errors="ignore")


def run_drift_report() -> dict:
    """Gera relatório de drift e qualidade. Retorna sumário com share de features com drift."""
    reference = _load_reference()
    current = _load_current()

    feats = feature_names(reference)
    current = current.reindex(columns=feats)
    reference_x = reference[feats]

    mapping = ColumnMapping(numerical_features=feats, categorical_features=[])

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference_x, current_data=current, column_mapping=mapping)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = REPORTS_DIR / f"drift_report_{ts}.html"
    json_path = REPORTS_DIR / f"drift_report_{ts}.json"
    latest_html = REPORTS_DIR / "drift_report_latest.html"
    latest_json = REPORTS_DIR / "drift_report_latest.json"

    report.save_html(str(html_path))
    raw = report.as_dict()
    Path(json_path).write_text(json.dumps(raw, default=str, indent=2))

    latest_html.write_text(html_path.read_text())
    latest_json.write_text(json_path.read_text())

    drift_metric = next(
        (m for m in raw["metrics"] if m.get("metric") == "DatasetDriftMetric"),
        None,
    )
    summary = {
        "html_report": str(html_path),
        "json_report": str(json_path),
        "n_reference": len(reference_x),
        "n_current": len(current),
    }
    if drift_metric:
        result = drift_metric.get("result", {})
        summary.update(
            {
                "dataset_drift": bool(result.get("dataset_drift", False)),
                "share_of_drifted_columns": float(result.get("share_of_drifted_columns", 0.0)),
                "number_of_drifted_columns": int(result.get("number_of_drifted_columns", 0)),
            }
        )
    return summary


if __name__ == "__main__":
    print(json.dumps(run_drift_report(), indent=2))
