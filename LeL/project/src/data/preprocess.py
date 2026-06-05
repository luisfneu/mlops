"""Pré-processamento + validação com Pandera."""
from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera.typing import Series
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, REFERENCE_DIR, settings


class WineSchema(pa.DataFrameModel):
    """Schema esperado para o dataset Wine após carregamento."""

    alcohol: Series[float] = pa.Field(ge=10.0, le=16.0)
    malic_acid: Series[float] = pa.Field(ge=0.0, le=6.0)
    ash: Series[float] = pa.Field(ge=1.0, le=4.0)
    alcalinity_of_ash: Series[float] = pa.Field(ge=10.0, le=35.0)
    magnesium: Series[float] = pa.Field(ge=70.0, le=170.0)
    total_phenols: Series[float] = pa.Field(ge=0.0, le=4.0)
    flavanoids: Series[float] = pa.Field(ge=0.0, le=6.0)
    nonflavanoid_phenols: Series[float] = pa.Field(ge=0.0, le=1.0)
    proanthocyanins: Series[float] = pa.Field(ge=0.0, le=4.0)
    color_intensity: Series[float] = pa.Field(ge=1.0, le=14.0)
    hue: Series[float] = pa.Field(ge=0.0, le=2.0)
    od280_od315_of_diluted_wines: Series[float] = pa.Field(alias="od280/od315_of_diluted_wines", ge=1.0, le=4.5)
    proline: Series[float] = pa.Field(ge=250.0, le=2000.0)
    target: Series[int] = pa.Field(isin=[0, 1, 2])

    class Config:
        strict = True
        coerce = True


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame contra WineSchema, levantando erro se inválido."""
    return WineSchema.validate(df, lazy=True)


def split_and_save(df: pd.DataFrame, test_size: float = 0.2) -> dict[str, str]:
    """Split treino/teste, persiste como parquet e salva referência para drift."""
    df = validate(df)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=settings.random_state,
        stratify=df["target"],
    )

    train_path = PROCESSED_DIR / "train.parquet"
    test_path = PROCESSED_DIR / "test.parquet"
    ref_path = REFERENCE_DIR / "reference.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    train_df.to_parquet(ref_path, index=False)

    return {
        "train": str(train_path),
        "test": str(test_path),
        "reference": str(ref_path),
    }
