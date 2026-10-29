"""Model loading, input assembly, prediction and request logging."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from house_price.data.schema import ALLOWED_CATEGORIES, ID_COLUMN, RAW_COLUMNS, TARGET
from house_price.logging_utils import get_logger
from house_price.models.registry import load_bundle
from house_price.models.train import predict_dollars
from house_price.monitoring.prediction_log import PredictionLog

logger = get_logger(__name__)

FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column for column in RAW_COLUMNS if column not in {ID_COLUMN, TARGET}
)


class InputValidationError(ValueError):
    """Raised when a request contains values the model cannot handle."""


class Predictor:
    def __init__(
        self,
        model_path: Path | None = None,
        metadata_path: Path | None = None,
        *,
        log_predictions: bool = True,
    ) -> None:
        self.pipeline, self.metadata = load_bundle(model_path, metadata_path)
        self.model_version = str(
            self.metadata.get("trained_at") or self.metadata.get("model_name", "unknown")
        )
        clip = self.metadata.get("clip")
        self.clip: tuple[float, float] | None = (float(clip[0]), float(clip[1])) if clip else None
        self.prediction_log = PredictionLog() if log_predictions else None
        logger.info(
            "Loaded model %s version %s",
            self.metadata.get("model_name", "unknown"),
            self.model_version,
        )

    def predict_records(self, records: list[dict[str, Any]]) -> np.ndarray:
        if not records:
            raise InputValidationError("No records provided")
        frame = pd.DataFrame(records)
        unknown = [column for column in frame.columns if column not in FEATURE_COLUMNS]
        if unknown:
            raise InputValidationError(f"Unknown fields: {unknown}")

        for column in FEATURE_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame.loc[:, list(FEATURE_COLUMNS)]

        self._check_categories(frame)
        predictions = predict_dollars(self.pipeline, frame, self.clip)
        return np.round(predictions, 2)

    def _check_categories(self, frame: pd.DataFrame) -> None:
        problems: dict[str, list[str]] = {}
        for column, allowed in ALLOWED_CATEGORIES.items():
            if column not in frame.columns:
                continue
            values = frame[column].dropna().astype(str)
            invalid = sorted(set(values) - allowed)
            if invalid:
                problems[column] = invalid
        if problems:
            allowed_values = {column: sorted(ALLOWED_CATEGORIES[column]) for column in problems}
            raise InputValidationError(
                f"Unknown categorical values {problems}. Allowed values: {allowed_values}"
            )

    def log_request(
        self,
        records: list[dict[str, Any]],
        predictions: list[float],
        *,
        latency_ms: float,
        endpoint: str,
    ) -> None:
        if self.prediction_log is None:
            return
        try:
            self.prediction_log.log(
                records,
                predictions,
                latency_ms=latency_ms,
                model_version=self.model_version,
                endpoint=endpoint,
            )
        except Exception as exc:  # logging must never break serving
            logger.warning("Failed to persist prediction log: %s", exc)

    def info(self) -> dict[str, Any]:
        metadata = self.metadata
        return {
            "model_name": str(metadata.get("model_name", "unknown")),
            "model_version": self.model_version,
            "training_rows": int(metadata.get("n_train", 0)),
            "feature_count": int(metadata.get("feature_count", 0)),
            "cv_rmse_log": metadata.get("cv_rmse_log"),
            "test_metrics": metadata.get("test_metrics", {}),
            "trained_at": metadata.get("trained_at"),
            "library_versions": metadata.get("library_versions", {}),
        }


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()
