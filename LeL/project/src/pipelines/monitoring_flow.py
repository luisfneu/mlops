"""Flow Prefect que gera relatório de drift comparando dados de referência vs. produção."""
from __future__ import annotations

from prefect import flow, get_run_logger, task

from src.monitoring.drift import run_drift_report


@task
def t_drift_report() -> dict:
    return run_drift_report()


@flow(name="monitoring-flow", log_prints=True)
def monitoring_flow() -> dict:
    logger = get_run_logger()
    summary = t_drift_report()
    logger.info("Relatório de drift gerado: %s", summary)
    return summary


if __name__ == "__main__":
    monitoring_flow()
