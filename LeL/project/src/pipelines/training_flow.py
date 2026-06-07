"""Flow Prefect que orquestra: carga -> validação -> split -> treino -> promoção."""
from __future__ import annotations

from prefect import flow, get_run_logger, task

from src.data.load import load_raw
from src.data.preprocess import split_and_save, validate
from src.models.promote import promote_latest
from src.models.train import TrainConfig, train


@task(retries=2, retry_delay_seconds=5)
def t_load():
    logger = get_run_logger()
    df = load_raw()
    logger.info("Dataset carregado: %d linhas, %d colunas", *df.shape)
    return df


@task
def t_validate(df):
    return validate(df)


@task
def t_split(df, test_size: float = 0.2):
    return split_and_save(df, test_size=test_size)


@task(retries=1)
def t_train(cfg: dict, register: bool = True):
    return train(TrainConfig(**cfg), register=register)


@task
def t_promote(stage: str = "Production"):
    return promote_latest(stage=stage)


@flow(name="training-flow", log_prints=True)
def training_flow(
    n_estimators: int = 200,
    max_depth: int = 8,
    min_samples_split: int = 2,
    test_size: float = 0.2,
    promote_to: str = "Production",
):
    logger = get_run_logger()
    df = t_load()
    df = t_validate(df)
    paths = t_split(df, test_size=test_size)
    logger.info("Splits gravados em: %s", paths)
    result = t_train(
        cfg={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
        },
        register=True,
    )
    logger.info("Treino finalizado run=%s", result["run_id"])

    if promote_to:
        promo = t_promote(stage=promote_to)
        logger.info("Promoção: %s", promo)
        return {"train": result, "promotion": promo}
    return {"train": result}


if __name__ == "__main__":
    training_flow()
