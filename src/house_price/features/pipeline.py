"""Assemble the preprocessing and modelling pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
)

from house_price.config import data_config
from house_price.data.schema import (
    ENGINEERED_CONTINUOUS,
    HIGH_CARDINALITY_COLUMNS,
    NOMINAL_COLUMNS,
    NUMERIC_CONTINUOUS,
    NUMERIC_DISCRETE,
    ORDINAL_COLUMNS,
    ORDINAL_DEFAULTS,
)
from house_price.features.transformers import (
    FeatureEngineer,
    MappedOrdinalEncoder,
    OutOfFoldTargetEncoder,
    cast_and_fill_missing,
)


def _cast_transformer() -> FunctionTransformer:
    return FunctionTransformer(
        func=cast_and_fill_missing,
        feature_names_out="one-to-one",
        validate=False,
    )


def build_preprocessor(
    *,
    engineered: bool = True,
    use_target_encoding: bool = True,
    include_ordinal: bool = True,
    include_nominal: bool = True,
) -> ColumnTransformer:
    features_cfg = data_config().features

    continuous = list(NUMERIC_CONTINUOUS) + (list(ENGINEERED_CONTINUOUS) if engineered else [])

    transformers: list[tuple[str, Any, list[str]]] = [
        (
            "continuous",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
                ]
            ),
            continuous,
        ),
        (
            "discrete",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            list(NUMERIC_DISCRETE),
        ),
    ]

    if include_ordinal:
        transformers.append(
            (
                "ordinal",
                MappedOrdinalEncoder(
                    mapping=dict(features_cfg.ordinal_mapping),
                    defaults=dict(ORDINAL_DEFAULTS),
                ),
                list(ORDINAL_COLUMNS),
            )
        )

    if include_nominal:
        transformers.append(
            (
                "nominal",
                Pipeline(
                    [
                        ("cast", _cast_transformer()),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(NOMINAL_COLUMNS),
            )
        )

    if use_target_encoding:
        transformers.append(
            (
                "target_encoded",
                Pipeline(
                    [
                        ("cast", _cast_transformer()),
                        (
                            "encode",
                            OutOfFoldTargetEncoder(
                                smoothing=float(features_cfg.target_encoding_smoothing)
                            ),
                        ),
                    ]
                ),
                list(HIGH_CARDINALITY_COLUMNS),
            )
        )
    else:
        transformers.append(
            (
                "high_cardinality",
                Pipeline(
                    [
                        ("cast", _cast_transformer()),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(HIGH_CARDINALITY_COLUMNS),
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline(
    model: BaseEstimator,
    *,
    engineered: bool = True,
    use_target_encoding: bool = True,
) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureEngineer() if engineered else "passthrough"),
            (
                "preprocessor",
                build_preprocessor(
                    engineered=engineered,
                    use_target_encoding=use_target_encoding,
                ),
            ),
            ("model", model),
        ]
    )


def feature_names(pipeline: Pipeline) -> list[str]:
    """Return the column names fed into the final estimator."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return [str(name) for name in preprocessor.get_feature_names_out()]
