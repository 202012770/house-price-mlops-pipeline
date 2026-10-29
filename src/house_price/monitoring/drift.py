"""Population Stability Index based data drift detection."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from house_price.data.schema import (
    ENGINEERED_CONTINUOUS,
    HIGH_CARDINALITY_COLUMNS,
    NOMINAL_COLUMNS,
    NUMERIC_CONTINUOUS,
    NUMERIC_DISCRETE,
    ORDINAL_COLUMNS,
)

NUMERIC_REFERENCE_COLUMNS = list(NUMERIC_CONTINUOUS) + list(NUMERIC_DISCRETE)
CATEGORICAL_REFERENCE_COLUMNS = (
    list(ORDINAL_COLUMNS) + list(NOMINAL_COLUMNS) + list(HIGH_CARDINALITY_COLUMNS)
)
ENGINEERED_REFERENCE_COLUMNS = list(ENGINEERED_CONTINUOUS)

EPSILON = 1e-6
MAX_BINS = 10
MIN_DRIFT_SAMPLES = 20
RARE_CATEGORY_THRESHOLD = 0.02
MIN_ROWS_FOR_STATUS = 200


def _bin_edges(values: pd.Series) -> list[float]:
    quantiles = np.linspace(0, 1, MAX_BINS + 1)
    edges = np.unique(np.quantile(values.to_numpy(dtype="float64"), quantiles))
    if len(edges) < 2:
        edges = np.array([values.min(), values.max()])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return [float(edge) for edge in edges]


def _bin_proportions(values: pd.Series, edges: list[float]) -> np.ndarray:
    binned = pd.cut(values, bins=edges, include_lowest=True)
    counts = binned.value_counts(sort=False).to_numpy(dtype="float64")
    total = counts.sum()
    return counts / total if total else np.full(len(counts), 1.0 / len(counts))


def build_reference_stats(frame: pd.DataFrame) -> dict[str, Any]:
    """Snapshot the training distribution for later drift comparisons."""
    numeric: dict[str, Any] = {}
    for column in NUMERIC_REFERENCE_COLUMNS + ENGINEERED_REFERENCE_COLUMNS:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        edges = _bin_edges(values)
        numeric[column] = {
            "edges": edges,
            "proportions": _bin_proportions(values, edges).round(6).tolist(),
            "quantiles": np.quantile(values.to_numpy(dtype="float64"), np.linspace(0, 1, 101))
            .round(4)
            .tolist(),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "missing_rate": round(float(frame[column].isna().mean()), 4),
        }

    categorical: dict[str, Any] = {}
    for column in CATEGORICAL_REFERENCE_COLUMNS:
        if column not in frame.columns:
            continue
        raw = frame[column]
        provided = raw.dropna().astype(str)
        frequencies = provided.value_counts(normalize=True).head(50)
        categorical[column] = {
            "frequencies": {str(k): round(float(v), 6) for k, v in frequencies.items()},
            "n_categories": int(provided.nunique()),
            "missing_rate": round(float(raw.isna().mean()), 4),
        }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "numeric": numeric,
        "categorical": categorical,
    }


def population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(np.asarray(expected, dtype="float64"), EPSILON, None)
    actual = np.clip(np.asarray(actual, dtype="float64"), EPSILON, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _numeric_drift(
    column: str, reference: dict[str, Any], current: pd.Series
) -> dict[str, Any] | None:
    values = pd.to_numeric(current, errors="coerce").dropna()
    if len(values) < MIN_DRIFT_SAMPLES:
        return None
    actual = _bin_proportions(values, reference["edges"])
    psi = population_stability_index(reference["proportions"], actual)
    sample = values.to_numpy(dtype="float64")
    # The stored quantile grid approximates the reference CDF closely enough for KS.
    reference_sample = np.asarray(reference["quantiles"], dtype="float64")
    ks_statistic, ks_pvalue = stats.ks_2samp(reference_sample, sample)
    return {
        "type": "numeric",
        "psi": round(psi, 4),
        "ks_statistic": round(float(ks_statistic), 4),
        "ks_pvalue": round(float(ks_pvalue), 4),
        "current_mean": round(float(values.mean()), 4),
        "reference_mean": round(reference["mean"], 4),
        "provided_ratio": round(float(len(values) / max(len(current), 1)), 4),
    }


def _categorical_drift(
    column: str, reference: dict[str, Any], current: pd.Series
) -> dict[str, Any] | None:
    # Only compare values the caller actually supplied; omitted optional
    # features would otherwise look like a sudden "Missing" shift.
    provided = current.dropna()
    if len(provided) < MIN_DRIFT_SAMPLES:
        return None
    # Match the category normalisation used at training time (20.0 -> "20").
    values = provided.astype(str).str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    values = values.replace({"nan": "Missing"})
    actual_freq = values.value_counts(normalize=True)
    reference_freq = reference["frequencies"]
    if not reference_freq:
        return None
    unseen = sorted(set(actual_freq.index) - set(reference_freq))

    # Collapse rare reference categories so PSI is not dominated by sampling noise.
    rare = {
        category
        for category in set(reference_freq) | set(actual_freq.index)
        if reference_freq.get(category, 0.0) < RARE_CATEGORY_THRESHOLD
    }
    expected_bucket: dict[str, float] = {}
    actual_bucket: dict[str, float] = {}
    for category, value in reference_freq.items():
        key = "other" if category in rare else category
        expected_bucket[key] = expected_bucket.get(key, 0.0) + value
    for category, value in actual_freq.items():
        key = "other" if category in rare else category
        actual_bucket[key] = actual_bucket.get(key, 0.0) + value

    categories = sorted(set(expected_bucket) | set(actual_bucket))
    expected = np.array([expected_bucket.get(category, 0.0) for category in categories])
    actual = np.array([actual_bucket.get(category, 0.0) for category in categories])
    psi = population_stability_index(expected, actual)

    return {
        "type": "categorical",
        "psi": round(psi, 4),
        "unseen_categories": unseen[:10],
        "n_unseen": len(unseen),
        "provided_ratio": round(float(len(provided) / max(len(current), 1)), 4),
    }


def compute_drift(
    reference: dict[str, Any],
    current: pd.DataFrame,
    *,
    psi_stable: float = 0.1,
    psi_moderate: float = 0.25,
    min_rows: int = MIN_ROWS_FOR_STATUS,
) -> dict[str, Any]:
    features: dict[str, Any] = {}

    for column, stats_ref in reference.get("numeric", {}).items():
        if column in current.columns:
            result = _numeric_drift(column, stats_ref, current[column])
        elif column in ENGINEERED_CONTINUOUS:
            continue
        else:
            result = None
        if result:
            features[column] = result

    for column, stats_ref in reference.get("categorical", {}).items():
        if column in current.columns:
            result = _categorical_drift(column, stats_ref, current[column])
            if result:
                features[column] = result

    if not features:
        return {
            "status": "insufficient_data",
            "max_psi": 0.0,
            "n_checked": 0,
            "n_drifted": 0,
            "features": {},
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "rows": int(len(current)),
        }

    drifted = {name: details for name, details in features.items() if details["psi"] > psi_moderate}
    moderate = {
        name: details
        for name, details in features.items()
        if psi_stable < details["psi"] <= psi_moderate
    }
    if len(current) < min_rows:
        status = "insufficient_data"
    elif drifted:
        status = "drift"
    elif moderate:
        status = "moderate"
    else:
        status = "stable"

    return {
        "status": status,
        "max_psi": max(details["psi"] for details in features.values()),
        "n_checked": len(features),
        "n_drifted": len(drifted),
        "n_moderate": len(moderate),
        "drifted_features": list(drifted),
        "features": dict(sorted(features.items(), key=lambda item: item[1]["psi"], reverse=True)),
        "thresholds": {"stable": psi_stable, "moderate": psi_moderate},
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(current)),
    }


def save_reference_stats(stats: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_reference_stats(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_html_report(result: dict[str, Any], path: Path) -> Path:
    rows = []
    for name, details in result.get("features", {}).items():
        flags = ""
        if details["psi"] > result.get("thresholds", {}).get("moderate", 0.25):
            flags = "drift"
        elif details["psi"] > result.get("thresholds", {}).get("stable", 0.1):
            flags = "moderate"
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{details['type']}</td>"
            f"<td>{details['psi']:.4f}</td>"
            f"<td>{flags}</td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Data Drift Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 2rem; color: #1b1f24; }}
table {{ border-collapse: collapse; width: 100%; max-width: 720px; }}
th, td {{ border-bottom: 1px solid #d8dee4; padding: 6px 10px; text-align: left; }}
th {{ background: #f6f8fa; }}
.badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; color: #fff; }}
.stable {{ background: #1a7f37; }} .moderate {{ background: #9a6700; }}
.drift {{ background: #cf222e; }} .insufficient_data {{ background: #57606a; }}
</style></head><body>
<h1>Data Drift Report</h1>
<p>Status: <span class="badge {result["status"]}">{result["status"]}</span></p>
<p>Checked {result.get("n_checked", 0)} features over {result.get("rows", 0)} rows.
Max PSI: {result.get("max_psi", 0):.4f}. Drifted: {result.get("n_drifted", 0)}.</p>
<table><thead><tr><th>Feature</th><th>Type</th><th>PSI</th><th>Flag</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p>PSI thresholds: &lt;0.1 stable, 0.1-0.25 moderate, &gt;0.25 drift.
Generated at {html.escape(result.get("checked_at", ""))}.</p>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
