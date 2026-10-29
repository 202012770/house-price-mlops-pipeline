"""SHAP explanations and permutation importance for the fitted pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from house_price.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = pipeline.named_steps["features"].transform(X)
    transformed = pipeline.named_steps["preprocessor"].transform(features)
    names = [str(name) for name in pipeline.named_steps["preprocessor"].get_feature_names_out()]
    array = np.asarray(transformed, dtype="float64")
    return pd.DataFrame(array, columns=names, index=X.index), names


def _explainer(pipeline: Pipeline, X_trans: pd.DataFrame) -> Any:
    model = pipeline.named_steps["model"]
    model_type = type(model).__name__
    if model_type in {
        "RandomForestRegressor",
        "LGBMRegressor",
        "XGBRegressor",
        "CatBoostRegressor",
    }:
        return shap.TreeExplainer(model)
    return shap.LinearExplainer(model, X_trans)


def compute_shap(
    pipeline: Pipeline,
    X: pd.DataFrame,
    max_samples: int = 300,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    X_trans, _ = transform_features(pipeline, X)
    if len(X_trans) > max_samples:
        X_trans = X_trans.sample(max_samples, random_state=random_state).sort_index()

    explainer = _explainer(pipeline, X_trans)
    values = np.asarray(explainer.shap_values(X_trans), dtype="float64")
    if values.ndim == 3:
        values = values[..., 0]
    base_value = np.asarray(explainer.expected_value, dtype="float64").reshape(-1)[0]
    logger.info("Computed SHAP values for %d samples x %d features", *values.shape)
    return values, X_trans, float(base_value)


def save_shap_plots(
    values: np.ndarray,
    X_trans: pd.DataFrame,
    base_value: float,
    figures_dir: Path,
    worst_index: int | None = None,
) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    shap.summary_plot(values, X_trans, show=False, max_display=20)
    figure = plt.gcf()
    figure.set_size_inches(9, 7)
    figure.tight_layout()
    summary_path = figures_dir / "shap_summary.png"
    figure.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    outputs["shap_summary"] = summary_path

    shap.summary_plot(values, X_trans, plot_type="bar", show=False, max_display=20)
    figure = plt.gcf()
    figure.set_size_inches(9, 6)
    figure.tight_layout()
    bar_path = figures_dir / "shap_importance.png"
    figure.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    outputs["shap_importance"] = bar_path

    position = (
        0
        if worst_index is None
        else int(
            np.clip(np.searchsorted(X_trans.index.to_numpy(), worst_index), 0, len(X_trans) - 1)
        )
    )
    explanation = shap.Explanation(
        values=values[position],
        base_values=base_value,
        data=X_trans.iloc[position].to_numpy(),
        feature_names=list(X_trans.columns),
    )
    shap.plots.waterfall(explanation, max_display=12, show=False)
    figure = plt.gcf()
    figure.set_size_inches(9, 6)
    figure.tight_layout()
    waterfall_path = figures_dir / "shap_waterfall.png"
    figure.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    outputs["shap_waterfall"] = waterfall_path
    return outputs


def permutation_importance_table(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y_log: np.ndarray,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    result = permutation_importance(
        pipeline,
        X,
        y_log,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return table
