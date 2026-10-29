from __future__ import annotations

import numpy as np
import pandas as pd

from house_price.monitoring.drift import (
    build_reference_stats,
    compute_drift,
    population_stability_index,
    write_html_report,
)
from house_price.monitoring.prediction_log import PredictionLog


def test_psi_is_zero_for_identical_distributions() -> None:
    distribution = np.array([0.1, 0.2, 0.3, 0.4])
    assert abs(population_stability_index(distribution, distribution)) < 1e-9


def test_psi_grows_with_shift() -> None:
    expected = np.array([0.25, 0.25, 0.25, 0.25])
    mild = population_stability_index(expected, np.array([0.4, 0.3, 0.2, 0.1]))
    severe = population_stability_index(expected, np.array([0.9, 0.05, 0.03, 0.02]))
    assert severe > mild > 0


def test_drift_stable_on_same_distribution(synthetic_frame: pd.DataFrame) -> None:
    reference = build_reference_stats(synthetic_frame)
    result = compute_drift(reference, synthetic_frame.sample(140, random_state=1), min_rows=50)
    assert result["status"] == "stable"
    assert result["n_checked"] > 20


def test_drift_detects_large_shift(synthetic_frame: pd.DataFrame) -> None:
    reference = build_reference_stats(synthetic_frame)
    shifted = synthetic_frame.sample(140, random_state=2).copy()
    shifted["GrLivArea"] = shifted["GrLivArea"] * 5
    shifted["Neighborhood"] = "NewTown"
    result = compute_drift(reference, shifted, min_rows=50)
    assert result["status"] == "drift"
    assert "GrLivArea" in result["drifted_features"]
    assert "Neighborhood" in result["drifted_features"]


def test_drift_requires_enough_rows(synthetic_frame: pd.DataFrame) -> None:
    reference = build_reference_stats(synthetic_frame)
    result = compute_drift(reference, synthetic_frame.head(10), min_rows=50)
    assert result["status"] in {"insufficient_data", "stable"}


def test_html_report_written(synthetic_frame: pd.DataFrame, tmp_path) -> None:
    reference = build_reference_stats(synthetic_frame)
    result = compute_drift(reference, synthetic_frame.head(50), min_rows=10)
    path = write_html_report(result, tmp_path / "drift.html")
    content = path.read_text(encoding="utf-8")
    assert "Data Drift Report" in content
    assert result["status"] in content


def test_prediction_log_roundtrip(tmp_path) -> None:
    log = PredictionLog(tmp_path / "predictions.db")
    records = [{"GrLivArea": 1500.0, "Neighborhood": "NAmes"}]
    log.log(
        records,
        [250_000.0],
        latency_ms=12.5,
        model_version="v1",
        endpoint="/predict",
    )
    assert log.count() == 1
    frame = log.recent_features(limit=10)
    assert frame.iloc[0]["Neighborhood"] == "NAmes"
