from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from house_price.data.schema import (
    HIGH_CARDINALITY_COLUMNS,
    NOMINAL_COLUMNS,
    NUMERIC_CONTINUOUS,
    NUMERIC_DISCRETE,
    ORDINAL_COLUMNS,
    RAW_COLUMNS,
)

CATEGORY_POOL = {
    "Neighborhood": ["CollgCr", "NAmes", "Edwards"],
    "MSZoning": ["RL", "RM"],
    "CentralAir": ["Y", "N"],
    "Street": ["Pave", "Grvl"],
    "PavedDrive": ["Y", "N", "P"],
    "Functional": ["Typ", "Min1"],
    "KitchenQual": ["TA", "Gd", "Ex"],
    "ExterQual": ["TA", "Gd"],
    "ExterCond": ["TA", "Gd"],
    "HeatingQC": ["TA", "Gd"],
    "BsmtQual": ["TA", "Gd"],
    "BsmtCond": ["TA"],
    "GarageQual": ["TA"],
    "GarageCond": ["TA"],
    "FireplaceQu": ["TA"],
    "PoolQC": ["Ex"],
    "BsmtExposure": ["No", "Av"],
    "BsmtFinType1": ["Unf", "GLQ"],
    "BsmtFinType2": ["Unf"],
    "GarageFinish": ["Unf", "Fin"],
}


def _synthetic_frame(n: int = 160, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {"Id": np.arange(1, n + 1)}

    for column in NUMERIC_CONTINUOUS:
        data[column] = np.clip(rng.lognormal(mean=6.2, sigma=0.6, size=n), 1, None).round(1)
    for column in NUMERIC_DISCRETE:
        data[column] = rng.integers(0, 10, size=n)
    for column in ORDINAL_COLUMNS + NOMINAL_COLUMNS + HIGH_CARDINALITY_COLUMNS:
        pool = CATEGORY_POOL.get(column, ["A", "B", "C"])
        data[column] = rng.choice(pool, size=n)

    frame = pd.DataFrame(data)
    frame["YearBuilt"] = rng.integers(1900, 2010, size=n)
    frame["YearRemodAdd"] = frame["YearBuilt"] + rng.integers(0, 20, size=n)
    frame["YrSold"] = rng.integers(2006, 2011, size=n)
    frame["MoSold"] = rng.integers(1, 13, size=n)
    frame["GarageYrBlt"] = frame["YearBuilt"]
    frame["OverallQual"] = rng.integers(3, 10, size=n)
    frame["OverallCond"] = rng.integers(3, 10, size=n)
    frame["GrLivArea"] = rng.integers(700, 3000, size=n).astype(float)
    frame["TotalBsmtSF"] = rng.integers(0, 1800, size=n).astype(float)
    frame["1stFlrSF"] = rng.integers(500, 2000, size=n).astype(float)
    frame["2ndFlrSF"] = rng.integers(0, 1200, size=n).astype(float)
    frame["LotArea"] = rng.integers(2000, 20000, size=n).astype(float)
    frame["GarageArea"] = rng.integers(0, 800, size=n).astype(float)
    frame["FullBath"] = rng.integers(0, 4, size=n)
    frame["HalfBath"] = rng.integers(0, 3, size=n)
    frame["BsmtFullBath"] = rng.integers(0, 3, size=n)
    frame["BsmtHalfBath"] = rng.integers(0, 2, size=n)
    frame["TotRmsAbvGrd"] = rng.integers(3, 12, size=n)
    frame["Fireplaces"] = rng.integers(0, 3, size=n)
    frame["GarageCars"] = rng.integers(0, 4, size=n)
    frame["BedroomAbvGr"] = rng.integers(1, 6, size=n)
    frame["KitchenAbvGr"] = rng.integers(1, 3, size=n)

    base_price = (
        40_000
        + 60 * frame["GrLivArea"]
        + 12_000 * frame["OverallQual"]
        + rng.normal(0, 18_000, size=n)
    )
    frame["SalePrice"] = np.clip(base_price, 35_000, 800_000).round(0)

    for column in ("LotFrontage", "MasVnrArea", "GarageYrBlt"):
        frame.loc[frame.sample(frac=0.1, random_state=seed).index, column] = np.nan
    frame["Alley"] = np.nan

    ordered = [column for column in RAW_COLUMNS if column in frame.columns]
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    for column in missing:
        frame[column] = 0.0
    return frame.loc[:, ordered]


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    return _synthetic_frame()


@pytest.fixture(scope="session")
def fitted_pipeline(synthetic_frame: pd.DataFrame):
    from house_price.features.pipeline import build_feature_pipeline
    from house_price.models.train import prepare_xy
    from house_price.models.tune import build_model

    pipeline = build_feature_pipeline(
        build_model("lightgbm", {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 15})
    )
    X, y_log = prepare_xy(synthetic_frame)
    pipeline.fit(X, y_log)
    return pipeline


@pytest.fixture(scope="session")
def model_bundle(
    tmp_path_factory: pytest.TempPathFactory,
    synthetic_frame: pd.DataFrame,
    fitted_pipeline,
):
    pipeline = fitted_pipeline
    directory = tmp_path_factory.mktemp("bundle")
    model_path = directory / "model.joblib"
    metadata_path = directory / "metadata.json"
    joblib.dump(pipeline, model_path)
    metadata = {
        "model_name": "lightgbm",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "n_train": int(len(synthetic_frame)),
        "feature_count": 10,
        "cv_rmse_log": 0.12,
        "test_metrics": {"mae": 15000.0, "rmse": 21000.0, "r2": 0.9},
        "clip": [35_000.0, 800_000.0],
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return model_path, metadata_path


@pytest.fixture()
def predictor(model_bundle):
    from house_price.serving.predictor import Predictor

    model_path, metadata_path = model_bundle
    return Predictor(model_path, metadata_path, log_predictions=False)


@pytest.fixture()
def api_client(predictor, monkeypatch):
    from fastapi.testclient import TestClient

    import house_price.serving.api as api_module

    monkeypatch.setattr(api_module, "get_predictor", lambda: predictor)
    with TestClient(api_module.app) as client:
        yield client


@pytest.fixture()
def sample_payload() -> dict[str, object]:
    return {
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
