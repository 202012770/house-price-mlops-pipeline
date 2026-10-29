from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest

from house_price.evaluation.metrics import regression_metrics
from house_price.features.pipeline import build_feature_pipeline
from house_price.models.registry import (
    compute_clip_bounds,
    library_versions,
    load_bundle,
    save_bundle,
)
from house_price.models.train import (
    evaluate_on_frame,
    fit_final_model,
    predict_dollars,
    prepare_xy,
)
from house_price.models.tune import (
    build_model,
    cross_validated_rmse,
    suggest_params,
)


def test_regression_metrics() -> None:
    metrics = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["r2"] < 1.0
    assert metrics["mape"] > 0


@pytest.mark.parametrize("name", ["ridge", "random_forest", "xgboost", "lightgbm", "catboost"])
def test_build_model_returns_estimator(name: str) -> None:
    model = build_model(name, {})
    assert hasattr(model, "fit")


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError):
        build_model("magic", {})


@pytest.mark.parametrize("name", ["ridge", "random_forest", "xgboost", "lightgbm", "catboost"])
def test_suggest_params_covers_all_models(name: str) -> None:
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    params = suggest_params(trial, name, 500)
    assert isinstance(params, dict) and params


def test_cross_validated_rmse_runs(synthetic_frame: pd.DataFrame) -> None:
    X, y_log = prepare_xy(synthetic_frame)
    pipeline = build_feature_pipeline(build_model("ridge", {"alpha": 1.0}))
    mean, std = cross_validated_rmse(pipeline, X, y_log, n_splits=2)
    assert np.isfinite(mean) and std >= 0


def test_fit_final_model_and_predict(synthetic_frame: pd.DataFrame) -> None:
    pipeline = fit_final_model("lightgbm", {"n_estimators": 30, "num_leaves": 7}, synthetic_frame)
    X, _ = prepare_xy(synthetic_frame)
    predictions = predict_dollars(pipeline, X, clip=(40_000.0, 700_000.0))
    assert len(predictions) == len(synthetic_frame)
    assert predictions.min() >= 40_000.0
    assert predictions.max() <= 700_000.0

    metrics = evaluate_on_frame(pipeline, synthetic_frame)
    assert metrics["mae"] > 0
    assert metrics["r2"] > 0.5


def test_save_and_load_bundle(tmp_path, fitted_pipeline, synthetic_frame: pd.DataFrame) -> None:
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    save_bundle(fitted_pipeline, {"model_name": "lightgbm"}, model_path, metadata_path)
    loaded, metadata = load_bundle(model_path, metadata_path)
    assert metadata["model_name"] == "lightgbm"
    assert "library_versions" in metadata
    X, y_log = prepare_xy(synthetic_frame)
    assert np.allclose(loaded.predict(X), fitted_pipeline.predict(X))
    assert y_log.shape[0] == len(synthetic_frame)


def test_load_bundle_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "missing.joblib", tmp_path / "missing.json")


def test_clip_bounds_and_versions() -> None:
    low, high = compute_clip_bounds(np.arange(1, 101, dtype="float64"))
    assert low < high
    versions = library_versions()
    assert "scikit-learn" in versions and "python" in versions
