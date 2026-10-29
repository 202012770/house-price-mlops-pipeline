"""Quantify the contribution of each feature engineering step."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from house_price.config import data_config
from house_price.data.ingest import load_raw_data, split_data
from house_price.data.schema import TARGET
from house_price.evaluation.metrics import regression_metrics
from house_price.features.pipeline import build_feature_pipeline
from house_price.logging_utils import get_logger
from house_price.models.train import predict_dollars
from house_price.models.tune import build_model, cross_validated_rmse

logger = get_logger(__name__)

DEFAULT_LIGHTGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 8,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
}

VARIANTS: tuple[tuple[str, dict[str, bool]], ...] = (
    ("baseline_raw_onehot", {"engineered": False, "use_target_encoding": False}),
    ("plus_domain_features", {"engineered": True, "use_target_encoding": False}),
    ("plus_target_encoding", {"engineered": True, "use_target_encoding": True}),
)


def _load_best_params(reports_dir: Path) -> dict[str, Any]:
    path = reports_dir / "best_params.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model") == "lightgbm":
            return dict(payload["params"])
    return DEFAULT_LIGHTGBM_PARAMS


def run_ablation() -> pd.DataFrame:
    cfg = data_config()
    frame = load_raw_data()
    train_df, test_df = split_data(frame)

    X_train = train_df.drop(columns=[TARGET])
    X_test = test_df.drop(columns=[TARGET])
    y_log = np.log1p(train_df[TARGET].to_numpy(dtype="float64"))
    y_test = test_df[TARGET].to_numpy(dtype="float64")

    params = _load_best_params(cfg.path("paths.reports_dir"))
    rows: list[dict[str, Any]] = []

    for label, options in VARIANTS:
        pipeline = build_feature_pipeline(
            build_model("lightgbm", params),
            engineered=options["engineered"],
            use_target_encoding=options["use_target_encoding"],
        )
        cv_rmse, cv_std = cross_validated_rmse(pipeline, X_train, y_log)
        pipeline.fit(X_train, y_log)
        metrics = regression_metrics(y_test, predict_dollars(pipeline, X_test))
        rows.append(
            {
                "variant": label,
                "cv_rmse_log": round(cv_rmse, 5),
                "cv_std_log": round(cv_std, 5),
                "test_mae": metrics["mae"],
                "test_rmse": metrics["rmse"],
                "test_r2": metrics["r2"],
                "n_features": int(pipeline[:-1].transform(X_train.head(5)).shape[1]),
            }
        )
        logger.info("%s: cv_rmse_log=%.5f", label, cv_rmse)

    table = pd.DataFrame(rows)
    baseline = float(table.iloc[0]["cv_rmse_log"])
    table["improvement_vs_baseline_pct"] = (
        (baseline - table["cv_rmse_log"]) / baseline * 100
    ).round(2)

    out_path = cfg.path("paths.ablation_file")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print(f"Saved ablation results to {out_path}")
    return table


if __name__ == "__main__":
    run_ablation()
