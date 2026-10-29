"""Pydantic request and response models for the prediction API."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from house_price.config import training_config
from house_price.data.schema import (
    CORE_API_FEATURES,
    NUMERIC_CONTINUOUS,
    NUMERIC_DISCRETE,
    NUMERIC_RANGES,
    OPTIONAL_API_FEATURES,
)

NUMERIC_API_FIELDS = set(NUMERIC_CONTINUOUS) | set(NUMERIC_DISCRETE)

CORE_DESCRIPTIONS = {
    "OverallQual": "Overall material and finish quality (1-10)",
    "GrLivArea": "Above-grade living area in square feet",
    "TotalBsmtSF": "Total basement area in square feet",
    "1stFlrSF": "First floor area in square feet",
    "2ndFlrSF": "Second floor area in square feet",
    "GarageCars": "Garage capacity in cars",
    "GarageArea": "Garage area in square feet",
    "YearBuilt": "Original construction year",
    "YearRemodAdd": "Remodel year",
    "FullBath": "Full bathrooms above grade",
    "TotRmsAbvGrd": "Total rooms above grade (excluding bathrooms)",
    "Fireplaces": "Number of fireplaces",
    "LotArea": "Lot size in square feet",
    "Neighborhood": "Physical location within Ames",
    "MSZoning": "General zoning classification",
}


def _nan_to_none(cls: Any, data: Any) -> Any:
    """Accept NaN from pandas/json payloads as an omitted optional value."""
    if isinstance(data, dict):
        return {
            key: None if isinstance(value, float) and math.isnan(value) else value
            for key, value in data.items()
        }
    return data


def _field_definition(name: str, required: bool) -> tuple[Any, Any]:
    if name == "MSSubClass":
        return (int | None, Field(None, ge=0, le=500))
    if name in NUMERIC_API_FIELDS:
        bounds = NUMERIC_RANGES.get(name)
        constraints: dict[str, Any] = {}
        if bounds:
            constraints["ge"], constraints["le"] = bounds
        elif name not in {"GarageYrBlt"}:
            constraints["ge"] = 0
        if required:
            return (float, Field(..., description=CORE_DESCRIPTIONS.get(name), **constraints))
        return (
            float | None,
            Field(
                None,
                description=f"Optional. {CORE_DESCRIPTIONS.get(name, '')}".strip(),
                **constraints,
            ),
        )
    if required:
        return (str, Field(..., description=CORE_DESCRIPTIONS.get(name)))
    return (str | None, Field(None))


def _build_house_features() -> Any:
    fields: dict[str, Any] = {
        name: _field_definition(name, required=True) for name in CORE_API_FEATURES
    }
    fields.update({name: _field_definition(name, required=False) for name in OPTIONAL_API_FEATURES})
    return create_model(
        "HouseFeatures",
        __validators__={
            "nan_to_none": model_validator(mode="before")(_nan_to_none)  # type: ignore[dict-item]
        },
        __config__=ConfigDict(
            extra="forbid",
            json_schema_extra={
                "example": {
                    "OverallQual": 7,
                    "GrLivArea": 1710,
                    "TotalBsmtSF": 856,
                    "1stFlrSF": 856,
                    "2ndFlrSF": 854,
                    "GarageCars": 2,
                    "GarageArea": 548,
                    "YearBuilt": 2003,
                    "YearRemodAdd": 2003,
                    "FullBath": 2,
                    "TotRmsAbvGrd": 8,
                    "Fireplaces": 1,
                    "LotArea": 8450,
                    "Neighborhood": "CollgCr",
                    "MSZoning": "RL",
                }
            },
        ),
        **fields,
    )


HouseFeatures: Any = _build_house_features()

MAX_BATCH = int(training_config().serving.batch_limit)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[HouseFeatures] = Field(
        ..., min_length=1, max_length=MAX_BATCH, description="Up to 1000 feature objects"
    )


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted sale price in USD")
    currency: str = "USD"
    model_version: str
    request_id: str


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    count: int
    currency: str = "USD"
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    target: str = "SalePrice (USD)"
    training_rows: int
    feature_count: int
    cv_rmse_log: float | None = None
    test_metrics: dict[str, float] = Field(default_factory=dict)
    trained_at: str | None = None
    library_versions: dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
