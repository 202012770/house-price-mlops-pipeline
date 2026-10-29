"""Estimator factory shared by training, tuning and serving."""

from __future__ import annotations

from typing import Any

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

RANDOM_STATE = 42


def build_model(name: str, params: dict[str, Any]) -> BaseEstimator:
    if name == "ridge":
        return Ridge(**params)
    if name == "random_forest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)
    if name == "xgboost":
        return XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
            **params,
        )
    if name == "lightgbm":
        return LGBMRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
            subsample_freq=1,
            **params,
        )
    if name == "catboost":
        return CatBoostRegressor(
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=1,
            **params,
        )
    raise ValueError(f"Unknown model: {name}")
