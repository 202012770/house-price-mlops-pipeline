"""Tune every enabled model, log to MLflow and persist the best pipeline."""

from __future__ import annotations

import json
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from house_price.config import PROJECT_ROOT, data_config, resolve_path, training_config
from house_price.data.ingest import load_raw_data, split_data
from house_price.data.schema import TARGET
from house_price.data.validate import validate_training_data
from house_price.evaluation.metrics import regression_metrics
from house_price.features.pipeline import feature_names
from house_price.logging_utils import get_logger
from house_price.models.registry import compute_clip_bounds, save_bundle
from house_price.models.train import fit_final_model, predict_dollars, prepare_xy
from house_price.models.tune import TuningResult, tune_model
from house_price.monitoring.drift import build_reference_stats, save_reference_stats

logger = get_logger(__name__)


def _configure_mlflow() -> str:
    db_path = resolve_path(training_config().mlflow.tracking_db)
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(training_config().mlflow.experiment)
    return tracking_uri


def _log_model(pipeline: Any) -> None:
    options = {
        "sk_model": pipeline,
        # custom transformers are not supported by skops
        "serialization_format": "cloudpickle",
    }
    try:
        mlflow.sklearn.log_model(**options, name="model")
    except TypeError:
        mlflow.sklearn.log_model(**options, artifact_path="model")


def run_comparison(
    *,
    models: list[str] | None = None,
    n_trials: int | None = None,
) -> pd.DataFrame:
    cfg = data_config()
    train_cfg = training_config()

    frame = load_raw_data()
    validate_training_data(frame)
    train_df, test_df = split_data(frame)

    X_train, y_log = prepare_xy(train_df)
    X_test = test_df.drop(columns=[TARGET])

    enabled = [
        name
        for name, options in train_cfg.models.items()
        if options.get("enabled", False) and (models is None or name in models)
    ]
    if not enabled:
        raise ValueError("No models enabled for comparison")

    _configure_mlflow()
    results: list[dict[str, Any]] = []
    tuned: dict[str, tuple[TuningResult, Any, dict[str, float]]] = {}

    for name in enabled:
        high = int(train_cfg.models[name].get("n_estimators_range", [200, 1200])[1])
        tuning = tune_model(name, X_train, y_log, n_trials=n_trials, n_estimators_high=high)
        pipeline = fit_final_model(name, tuning.best_params, train_df)
        test_metrics = regression_metrics(
            test_df[TARGET].to_numpy(), predict_dollars(pipeline, X_test)
        )

        with mlflow.start_run(run_name=f"{name}-tuned") as run:
            mlflow.log_params(tuning.best_params)
            mlflow.log_metrics(
                {
                    "cv_rmse_log": tuning.cv_rmse,
                    "cv_std_log": tuning.cv_std,
                    **test_metrics,
                }
            )
            _log_model(pipeline)
            run_id = run.info.run_id

        tuned[name] = (tuning, pipeline, test_metrics)
        results.append(
            {
                "model": name,
                "cv_rmse_log": round(tuning.cv_rmse, 5),
                "cv_std_log": round(tuning.cv_std, 5),
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_mape": test_metrics["mape"],
                "n_trials": tuning.n_trials,
                "seconds": round(tuning.seconds, 1),
                "mlflow_run_id": run_id,
            }
        )

    comparison = pd.DataFrame(results).sort_values("cv_rmse_log").reset_index(drop=True)
    comparison_path = cfg.path("paths.comparison_file")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)

    best_name = str(comparison.iloc[0]["model"])
    best_tuning, best_pipeline, best_test_metrics = tuned[best_name]

    y_raw = train_df[TARGET].to_numpy(dtype="float64")
    clip = compute_clip_bounds(y_raw)
    reference_stats = build_reference_stats(train_df)
    reference_path = save_reference_stats(reference_stats, cfg.path("paths.reference_stats_file"))

    metadata = {
        "model_name": best_name,
        "params": best_tuning.best_params,
        "cv_rmse_log": round(best_tuning.cv_rmse, 5),
        "cv_std_log": round(best_tuning.cv_std, 5),
        "test_metrics": best_test_metrics,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "target_transform": cfg.features.target_transform,
        "clip": [clip[0], clip[1]],
        "feature_count": len(feature_names(best_pipeline)),
        "feature_names": feature_names(best_pipeline),
        "reference_stats_file": str(reference_path.relative_to(PROJECT_ROOT)),
        "mlflow_run_id": str(comparison.iloc[0]["mlflow_run_id"]),
        "selection_metric": "cv_rmse_log",
    }
    model_path, metadata_path = save_bundle(best_pipeline, metadata)

    _register_best(best_name, str(comparison.iloc[0]["mlflow_run_id"]))

    params_path = cfg.path("paths.reports_dir") / "best_params.json"
    params_path.write_text(
        json.dumps(
            {
                "model": best_name,
                "params": best_tuning.best_params,
                "cv_rmse_log": best_tuning.cv_rmse,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Best model %s (cv_rmse_log=%.5f). Saved to %s",
        best_name,
        best_tuning.cv_rmse,
        model_path,
    )
    print(comparison.to_string(index=False))
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    return comparison


def _register_best(model_name: str, run_id: str) -> None:
    cfg = training_config()
    try:
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=cfg.mlflow.registered_model_name,
        )
        logger.info("Registered model %s in MLflow", model_name)
    except Exception as exc:  # registry is optional for local file setups
        logger.warning("MLflow model registration skipped: %s", exc)


if __name__ == "__main__":
    import numpy as np  # noqa: F401

    run_comparison()
