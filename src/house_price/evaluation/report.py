"""Generate the full evaluation report for the registered model."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from house_price.config import data_config
from house_price.data.ingest import load_raw_data, split_data
from house_price.data.schema import TARGET
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
)
from house_price.evaluation.metrics import regression_metrics
from house_price.logging_utils import get_logger
from house_price.models.registry import load_bundle
from house_price.models.train import predict_dollars

logger = get_logger(__name__)


def run_evaluation(*, with_shap: bool = True) -> dict:
    cfg = data_config()
    reports_dir = cfg.path("paths.reports_dir")
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    pipeline, metadata = load_bundle()
    frame = load_raw_data()
    _, test_df = split_data(frame)

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].to_numpy(dtype="float64")
    raw_clip = metadata.get("clip")
    clip: tuple[float, float] | None = (
        (float(raw_clip[0]), float(raw_clip[1])) if raw_clip else None
    )
    predictions = predict_dollars(pipeline, X_test, clip)

    metrics: dict[str, Any] = dict(regression_metrics(y_test, predictions))
    metrics["model"] = metadata.get("model_name", "unknown")
    metrics["cv_rmse_log"] = metadata.get("cv_rmse_log")
    metrics["n_test"] = int(len(test_df))
    (cfg.path("paths.metrics_file")).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bands = price_band_table(y_test, predictions)
    bands.to_csv(reports_dir / "error_by_price_band.csv", index=False)

    if "Neighborhood" in test_df.columns:
        groups = group_table(test_df, y_test, predictions, "Neighborhood")
        groups.to_csv(reports_dir / "error_by_neighborhood.csv", index=False)

    worst_predictions(test_df, y_test, predictions, n=10).to_csv(
        reports_dir / "worst_predictions.csv", index=False
    )

    plot_residuals(y_test, predictions, figures_dir / "residuals.png")
    plot_actual_vs_predicted(y_test, predictions, figures_dir / "actual_vs_predicted.png")

    report: dict = {"metrics": metrics, "price_bands": bands.to_dict(orient="records")}

    if with_shap:
        y_pred_series = pd.Series(predictions, index=X_test.index)
        residuals = np.abs(y_test - y_pred_series.to_numpy())
        worst_index = int(X_test.index[int(np.argmax(residuals))])
        values, X_trans, base_value = compute_shap(pipeline, X_test)
        paths = save_shap_plots(values, X_trans, base_value, figures_dir, worst_index)
        report["shap_figures"] = {key: str(path) for key, path in paths.items()}
        shap_importance = pd.DataFrame(
            {
                "feature": X_trans.columns,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        shap_importance.to_csv(reports_dir / "shap_feature_importance.csv", index=False)
        plot_feature_importance(
            shap_importance.set_index("feature")["mean_abs_shap"],
            figures_dir / "shap_feature_importance.png",
        )

        y_log = np.log1p(y_test)
        permutation = permutation_importance_table(pipeline, X_test, y_log)
        permutation.to_csv(reports_dir / "permutation_importance.csv", index=False)

    (reports_dir / "evaluation_summary.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )

    logger.info("Evaluation complete: %s", metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Reports written to {reports_dir}")
    return report


if __name__ == "__main__":
    run_evaluation()
