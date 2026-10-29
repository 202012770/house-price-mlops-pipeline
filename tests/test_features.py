from __future__ import annotations

import numpy as np
import pandas as pd

from house_price.config import data_config
from house_price.data.schema import ENGINEERED_CONTINUOUS
from house_price.features.pipeline import build_feature_pipeline, feature_names
from house_price.features.transformers import (
    FeatureEngineer,
    MappedOrdinalEncoder,
    OutOfFoldTargetEncoder,
)
from house_price.models.tune import build_model


def test_feature_engineer_adds_domain_columns(synthetic_frame: pd.DataFrame) -> None:
    engineered = FeatureEngineer().fit_transform(synthetic_frame)
    for column in ENGINEERED_CONTINUOUS:
        assert column in engineered.columns
    assert "Id" not in engineered.columns
    assert "SalePrice" not in engineered.columns
    expected_total = (
        synthetic_frame["TotalBsmtSF"] + synthetic_frame["1stFlrSF"] + synthetic_frame["2ndFlrSF"]
    )
    pd.testing.assert_series_equal(engineered["TotalSF"], expected_total, check_names=False)


def test_ordinal_encoder_maps_and_defaults() -> None:
    frame = pd.DataFrame({"ExterQual": ["Ex", "TA", None], "Functional": ["Typ", None, None]})
    config = data_config().features
    encoder = MappedOrdinalEncoder(
        mapping=dict(config.ordinal_mapping), defaults={"Functional": 7.0}
    )
    encoded = encoder.fit_transform(frame)
    assert encoded["ExterQual"].tolist() == [5.0, 3.0, 0.0]
    assert encoded["Functional"].tolist() == [7.0, 7.0, 7.0]


def test_target_encoder_handles_unseen_category() -> None:
    frame = pd.DataFrame({"Neighborhood": ["A", "A", "B", "B", "C", "C"]})
    target = np.array([1.0, 1.2, 2.0, 2.2, 3.0, 3.2])
    encoder = OutOfFoldTargetEncoder(smoothing=1.0, n_splits=3)
    train_encoded = encoder.fit_transform(frame, target)
    assert train_encoded.shape == (6, 1)
    assert train_encoded.notna().all().all()

    unseen = encoder.transform(pd.DataFrame({"Neighborhood": ["Z"]}))
    assert np.isclose(unseen.iloc[0, 0], encoder.prior_)


def test_pipeline_handles_missing_optional_fields(synthetic_frame: pd.DataFrame) -> None:
    pipeline = build_feature_pipeline(
        build_model("lightgbm", {"n_estimators": 20, "num_leaves": 7})
    )
    X = synthetic_frame.drop(columns=["SalePrice"])
    y = np.log1p(synthetic_frame["SalePrice"].to_numpy())
    pipeline.fit(X, y)

    core = [
        "OverallQual",
        "GrLivArea",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "GarageCars",
        "GarageArea",
        "YearBuilt",
        "YearRemodAdd",
        "FullBath",
        "TotRmsAbvGrd",
        "Fireplaces",
        "LotArea",
        "Neighborhood",
        "MSZoning",
    ]
    rows = [{column: X.iloc[0][column] for column in core}]
    sparse = pd.DataFrame(rows).reindex(columns=X.columns)
    predictions = pipeline.predict(sparse)
    assert np.all(np.isfinite(predictions))


def test_feature_names_are_unique(synthetic_frame: pd.DataFrame) -> None:
    pipeline = build_feature_pipeline(build_model("ridge", {"alpha": 1.0}))
    X = synthetic_frame.drop(columns=["SalePrice"])
    pipeline.fit(X, np.log1p(synthetic_frame["SalePrice"].to_numpy()))
    names = feature_names(pipeline)
    assert len(names) == len(set(names))
    assert len(names) > 60
