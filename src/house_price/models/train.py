"""Fit a pipeline on the training split and score it on the holdout split."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from house_price.config import data_config
from house_price.data.schema import ID_COLUMN, TARGET
from house_price.evaluation.metrics import regression_metrics
from house_price.features.pipeline import build_feature_pipeline
from house_price.models.factory import build_model


def prepare_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    features = frame.drop(columns=[TARGET])
    return features, np.log1p(frame[TARGET].to_numpy(dtype="float64"))


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[TARGET])


def fit_final_model(
    name: str,
    params: dict[str, Any],
    train_df: pd.DataFrame,
) -> Pipeline:
    pipeline = build_feature_pipeline(build_model(name, params))
    features, y_log = prepare_xy(train_df)
    pipeline.fit(features, y_log)
    return pipeline


def predict_dollars(
    pipeline: Pipeline,
    features: pd.DataFrame,
    clip: tuple[float, float] | None = None,
) -> np.ndarray:
    predictions = np.expm1(pipeline.predict(features))
    if clip is not None:
        predictions = np.clip(predictions, clip[0], clip[1])
    return predictions


def evaluate_on_frame(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    clip: tuple[float, float] | None = None,
    prefix: str = "",
) -> dict[str, float]:
    features = prepare_features(frame)
    predictions = predict_dollars(pipeline, features, clip)
    return regression_metrics(frame[TARGET].to_numpy(), predictions, prefix=prefix)


if __name__ == "__main__":
    cfg = data_config()
    train = pd.read_csv(cfg.path("paths.processed_dir") / "train.csv")
    test = pd.read_csv(cfg.path("paths.processed_dir") / "test.csv")
    print(f"train={train.shape} test={test.shape} id_column={ID_COLUMN}")
    print("Use `hpp compare` to run tuning and model selection.")
