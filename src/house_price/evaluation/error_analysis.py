"""Slice-level error analysis and diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from house_price.evaluation.metrics import regression_metrics  # noqa: E402


def price_band_table(y_true: np.ndarray, y_pred: np.ndarray, n_bands: int = 4) -> pd.DataFrame:
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    frame["band"] = pd.qcut(frame["y_true"], q=n_bands, labels=False, duplicates="drop")
    rows = []
    for band, group in frame.groupby("band"):
        metrics = regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        rows.append(
            {
                "price_band": f"Q{int(band) + 1}",
                "price_min": round(float(group["y_true"].min()), 0),
                "price_max": round(float(group["y_true"].max()), 0),
                "n": int(len(group)),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "bias": round(float((group["y_true"] - group["y_pred"]).mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def group_table(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_column: str,
    min_count: int = 20,
) -> pd.DataFrame:
    working = pd.DataFrame(
        {
            "group": frame[group_column].astype(str).to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    rows = []
    for group, values in working.groupby("group"):
        if len(values) < min_count:
            continue
        metrics = regression_metrics(values["y_true"].to_numpy(), values["y_pred"].to_numpy())
        rows.append(
            {
                "group": group,
                "n": int(len(values)),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
            }
        )
    return pd.DataFrame(rows).sort_values("mae", ascending=False)


def worst_predictions(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 10,
) -> pd.DataFrame:
    residuals = y_pred - y_true
    order = np.argsort(np.abs(residuals))[::-1][:n]
    columns = [
        column for column in ("Id", "Neighborhood", "GrLivArea", "OverallQual") if column in frame
    ]
    result = frame.iloc[order][columns].copy()
    result["actual"] = y_true[order]
    result["predicted"] = y_pred[order]
    result["error"] = residuals[order].round(2)
    result["error_pct"] = (residuals[order] / y_true[order] * 100).round(2)
    return result


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> Path:
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(y_true, residuals, s=12, alpha=0.55, color="#2563eb")
    axes[0].axhline(0, color="#111827", linewidth=1)
    axes[0].set_xlabel("Actual price")
    axes[0].set_ylabel("Prediction error")
    axes[0].set_title("Residuals vs actual price")

    axes[1].hist(residuals, bins=40, color="#16a34a", alpha=0.85)
    axes[1].axvline(0, color="#111827", linewidth=1)
    axes[1].set_xlabel("Prediction error")
    axes[1].set_title("Error distribution")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(y_true, y_pred, s=14, alpha=0.55, color="#7c3aed")
    low, high = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([low, high], [low, high], color="#111827", linewidth=1)
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.set_title("Actual vs predicted")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(importances: pd.Series, path: Path, top_n: int = 20) -> Path:
    top = importances.sort_values(ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.to_numpy(), color="#0ea5e9")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} features")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
