"""Persist and load the trained pipeline together with its metadata."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from house_price.config import data_config


def library_versions() -> dict[str, str]:
    import catboost
    import lightgbm
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__,
        "catboost": catboost.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def compute_clip_bounds(y_raw: np.ndarray) -> tuple[float, float]:
    quantiles = data_config().features.prediction_clip_quantiles
    low, high = np.quantile(y_raw, quantiles)
    return float(low), float(high)


def save_bundle(
    pipeline: Any,
    metadata: dict[str, Any],
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[Path, Path]:
    cfg = data_config()
    model_path = cfg.path("paths.model_file") if model_path is None else model_path
    metadata_path = cfg.path("paths.metadata_file") if metadata_path is None else metadata_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(metadata)
    payload.setdefault("trained_at", datetime.now(timezone.utc).isoformat())
    payload.setdefault("library_versions", library_versions())

    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return model_path, metadata_path


def load_bundle(
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    cfg = data_config()
    model_path = cfg.path("paths.model_file") if model_path is None else model_path
    metadata_path = cfg.path("paths.metadata_file") if metadata_path is None else metadata_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run `hpp train` first.")
    pipeline = joblib.load(model_path)
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    return pipeline, metadata
