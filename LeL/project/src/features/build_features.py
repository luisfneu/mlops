"""Construção de features — separação X/y e definição da lista canônica de colunas."""
from __future__ import annotations

import pandas as pd

TARGET = "target"


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y


def feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET]
