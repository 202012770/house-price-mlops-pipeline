"""Optuna hyper-parameter search over cross-validated pipelines."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_score

from house_price.config import training_config
from house_price.features.pipeline import build_feature_pipeline
from house_price.logging_utils import get_logger
from house_price.models.factory import RANDOM_STATE
from house_price.models.factory import build_model as build_model

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    name: str
    best_params: dict[str, Any]
    cv_rmse: float
    cv_std: float
    n_trials: int
    seconds: float
    study: optuna.Study | None = field(default=None, repr=False)


def suggest_params(trial: optuna.Trial, name: str, high: int) -> dict[str, Any]:
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}

    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, min(high, 800), step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }

    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, high, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, high, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, high, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        }

    raise ValueError(f"Unknown model: {name}")


def cross_validated_rmse(
    pipeline: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    rmse_scores = -scores
    return float(rmse_scores.mean()), float(rmse_scores.std())


def tune_model(
    name: str,
    X_train: pd.DataFrame,
    y_log: np.ndarray,
    *,
    n_trials: int | None = None,
    timeout: int | None = None,
    n_estimators_high: int = 1200,
) -> TuningResult:
    cfg = training_config()
    n_trials = cfg.tuning.n_trials if n_trials is None else n_trials
    timeout = cfg.tuning.timeout_seconds if timeout is None else timeout
    n_splits = int(cfg.cv.n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, name, n_estimators_high)
        pipeline = build_feature_pipeline(build_model(name, params))
        mean_rmse, _ = cross_validated_rmse(pipeline, X_train, y_log, n_splits=n_splits)
        return mean_rmse

    started = time.perf_counter()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    seconds = time.perf_counter() - started

    best_params = dict(study.best_params)
    _, std = cross_validated_rmse(
        build_feature_pipeline(build_model(name, best_params)),
        X_train,
        y_log,
        n_splits=n_splits,
    )
    result = TuningResult(
        name=name,
        best_params=best_params,
        cv_rmse=float(study.best_value),
        cv_std=std,
        n_trials=len(study.trials),
        seconds=seconds,
        study=study,
    )
    logger.info(
        "Tuned %s: cv_rmse=%.5f (+/- %.5f) in %.1fs over %d trials",
        name,
        result.cv_rmse,
        result.cv_std,
        seconds,
        result.n_trials,
    )
    return result
