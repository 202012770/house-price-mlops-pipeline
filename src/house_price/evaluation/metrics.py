"""Regression metrics computed in dollar space."""

from __future__ import annotations

import numpy as np


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    mape = float(np.mean(np.abs(residual) / np.maximum(y_true, 1.0)))
    return {
        f"{prefix}mae": round(mae, 2),
        f"{prefix}rmse": round(rmse, 2),
        f"{prefix}r2": round(r2, 4),
        f"{prefix}mape": round(mape, 4),
        f"{prefix}median_ae": round(float(np.median(np.abs(residual))), 2),
    }
