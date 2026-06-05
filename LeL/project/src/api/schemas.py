"""Schemas Pydantic para a API de inferência."""
from __future__ import annotations

from pydantic import BaseModel, Field


class WineFeatures(BaseModel):
    """Features de uma amostra de vinho — nomes seguem o dataset original do sklearn."""

    alcohol: float = Field(..., ge=10.0, le=16.0, examples=[13.2])
    malic_acid: float = Field(..., ge=0.0, le=6.0, examples=[1.78])
    ash: float = Field(..., ge=1.0, le=4.0, examples=[2.14])
    alcalinity_of_ash: float = Field(..., ge=10.0, le=35.0, examples=[11.2])
    magnesium: float = Field(..., ge=70.0, le=170.0, examples=[100.0])
    total_phenols: float = Field(..., ge=0.0, le=4.0, examples=[2.65])
    flavanoids: float = Field(..., ge=0.0, le=6.0, examples=[2.76])
    nonflavanoid_phenols: float = Field(..., ge=0.0, le=1.0, examples=[0.26])
    proanthocyanins: float = Field(..., ge=0.0, le=4.0, examples=[1.28])
    color_intensity: float = Field(..., ge=1.0, le=14.0, examples=[4.38])
    hue: float = Field(..., ge=0.0, le=2.0, examples=[1.05])
    od280_od315_of_diluted_wines: float = Field(
        ..., ge=1.0, le=4.5, alias="od280/od315_of_diluted_wines", examples=[3.4]
    )
    proline: float = Field(..., ge=250.0, le=2000.0, examples=[1050.0])

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    prediction: int
    probabilities: list[float]
    model_version: str
    model_stage: str


class BatchRequest(BaseModel):
    instances: list[WineFeatures]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str | None = None
