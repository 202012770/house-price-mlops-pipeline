"""Exploratory analysis figures and a compact numeric summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from house_price.config import data_config  # noqa: E402
from house_price.data.ingest import load_raw_data  # noqa: E402
from house_price.data.schema import TARGET  # noqa: E402
from house_price.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def _plot_target_distribution(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(frame[TARGET], bins=50, color="#2563eb", alpha=0.85)
    axes[0].set_title("SalePrice distribution (raw)")
    axes[0].set_xlabel("USD")
    axes[1].hist(np.log1p(frame[TARGET]), bins=50, color="#16a34a", alpha=0.85)
    axes[1].set_title("log1p(SalePrice)")
    axes[1].set_xlabel("log(1 + USD)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_correlations(frame: pd.DataFrame, path: Path) -> None:
    numeric = frame.select_dtypes(include="number")
    correlations = numeric.corr(numeric_only=True)[TARGET].drop(TARGET).abs()
    top = correlations.sort_values(ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.to_numpy(), color="#0ea5e9")
    ax.set_title("Top correlations with SalePrice")
    ax.set_xlabel("|Pearson r|")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_neighborhood(frame: pd.DataFrame, path: Path) -> None:
    order = frame.groupby("Neighborhood")[TARGET].median().sort_values().index
    fig, ax = plt.subplots(figsize=(11, 4.5))
    data = [frame.loc[frame["Neighborhood"] == name, TARGET].to_numpy() for name in order]
    ax.boxplot(data, tick_labels=list(order))
    ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel("SalePrice")
    ax.set_title("SalePrice by neighborhood")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_area_vs_price(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(frame["GrLivArea"], frame[TARGET], s=12, alpha=0.55, color="#7c3aed")
    ax.set_xlabel("Above-grade living area (sqft)")
    ax.set_ylabel("SalePrice")
    ax.set_title("Living area vs price")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_missingness(frame: pd.DataFrame, path: Path) -> None:
    missing = frame.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(missing.index, missing.to_numpy() * 100, color="#f59e0b")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Columns with missing values")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_eda() -> dict:
    cfg = data_config()
    frame = load_raw_data()
    figures = cfg.path("paths.reports_dir") / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    _plot_target_distribution(frame, figures / "eda_target_distribution.png")
    _plot_correlations(frame, figures / "eda_correlations.png")
    _plot_neighborhood(frame, figures / "eda_neighborhood.png")
    _plot_area_vs_price(frame, figures / "eda_area_vs_price.png")
    _plot_missingness(frame, figures / "eda_missingness.png")

    numeric = frame.select_dtypes(include="number")
    correlations = numeric.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(ascending=False)
    summary = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "target": {
            "mean": float(frame[TARGET].mean()),
            "median": float(frame[TARGET].median()),
            "skew": float(frame[TARGET].skew()),
        },
        "target_skew_after_log": float(np.log1p(frame[TARGET]).skew()),
        "top_correlations": {k: round(float(v), 4) for k, v in correlations.head(10).items()},
        "missing_columns_gt_10pct": {
            k: round(float(v), 4) for k, v in frame.isna().mean().items() if v > 0.1
        },
        "neighborhood_median_price": {
            k: round(float(v), 0)
            for k, v in frame.groupby("Neighborhood")[TARGET].median().sort_values().items()
        },
    }
    path = cfg.path("paths.reports_dir") / "eda_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("EDA figures written to %s", figures)
    print(json.dumps(summary["target"], indent=2))
    print(f"Saved EDA summary to {path}")
    return summary


if __name__ == "__main__":
    run_eda()
