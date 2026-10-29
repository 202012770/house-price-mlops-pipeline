from __future__ import annotations

import numpy as np
import pandas as pd

from house_price.evaluation.error_analysis import (
    group_table,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals,
    price_band_table,
    worst_predictions,
)
from house_price.evaluation.explain import (
    compute_shap,
    permutation_importance_table,
    save_shap_plots,
    transform_features,
)


def _predictions(synthetic_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_true = synthetic_frame["SalePrice"].to_numpy(dtype="float64")
    rng = np.random.default_rng(3)
    y_pred = y_true * rng.normal(1.0, 0.08, size=len(y_true))
    return y_true, y_pred


def test_price_band_table(synthetic_frame: pd.DataFrame) -> None:
    y_true, y_pred = _predictions(synthetic_frame)
    table = price_band_table(y_true, y_pred)
    assert len(table) == 4
    assert {"mae", "rmse", "mape", "bias"} <= set(table.columns)


def test_group_table_and_worst(synthetic_frame: pd.DataFrame) -> None:
    y_true, y_pred = _predictions(synthetic_frame)
    groups = group_table(synthetic_frame, y_true, y_pred, "Neighborhood", min_count=5)
    assert not groups.empty
    worst = worst_predictions(synthetic_frame, y_true, y_pred, n=5)
    assert len(worst) == 5
    assert {"actual", "predicted", "error"} <= set(worst.columns)


def test_plots_are_written(synthetic_frame: pd.DataFrame, tmp_path) -> None:
    y_true, y_pred = _predictions(synthetic_frame)
    for path in (
        plot_residuals(y_true, y_pred, tmp_path / "residuals.png"),
        plot_actual_vs_predicted(y_true, y_pred, tmp_path / "actual.png"),
        plot_feature_importance(
            pd.Series({"a": 1.0, "b": 0.5, "c": 0.2}), tmp_path / "importance.png"
        ),
    ):
        assert path.exists() and path.stat().st_size > 0


def test_transform_features_shape(fitted_pipeline, synthetic_frame: pd.DataFrame) -> None:
    X = synthetic_frame.drop(columns=["SalePrice"])
    transformed, names = transform_features(fitted_pipeline, X)
    assert transformed.shape == (len(X), len(names))
    assert len(names) > 100


def test_shap_pipeline(fitted_pipeline, synthetic_frame: pd.DataFrame, tmp_path) -> None:
    X = synthetic_frame.drop(columns=["SalePrice"])
    values, X_trans, base_value = compute_shap(fitted_pipeline, X, max_samples=40)
    assert values.shape[0] == len(X_trans) == 40
    assert np.isfinite(base_value)
    paths = save_shap_plots(values, X_trans, base_value, tmp_path)
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0


def test_permutation_importance(fitted_pipeline, synthetic_frame: pd.DataFrame) -> None:
    X, y_log = (
        synthetic_frame.drop(columns=["SalePrice"]),
        np.log1p(synthetic_frame["SalePrice"].to_numpy(dtype="float64")),
    )
    table = permutation_importance_table(fitted_pipeline, X.head(60), y_log[:60], n_repeats=1)
    assert len(table) == X.shape[1]
    assert table["importance_mean"].notna().all()
