"""Carregamento do dataset Wine (sklearn) — escolhido por rodar offline e ter classes balanceadas."""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_wine

from src.config import RAW_DIR


def load_wine_df() -> pd.DataFrame:
    """Retorna o dataset Wine como DataFrame com a coluna alvo 'target'."""
    bunch = load_wine(as_frame=True)
    df = bunch.frame.copy()
    df = df.rename(columns={"target": "target"})
    return df


def save_raw(df: pd.DataFrame, name: str = "wine.parquet") -> str:
    path = RAW_DIR / name
    df.to_parquet(path, index=False)
    return str(path)


def load_raw(name: str = "wine.parquet") -> pd.DataFrame:
    path = RAW_DIR / name
    if not path.exists():
        df = load_wine_df()
        save_raw(df, name)
        return df
    return pd.read_parquet(path)
