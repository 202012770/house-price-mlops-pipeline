"""Data quality checks for training and inference."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from house_price.config import data_config
from house_price.data.schema import (
    ALLOWED_CATEGORIES,
    ID_COLUMN,
    NUMERIC_RANGES,
    RAW_COLUMNS,
    TARGET,
)
from house_price.logging_utils import get_logger

logger = get_logger(__name__)


class DataValidationError(ValueError):
    """Raised when data violates a critical schema or business rule."""


def _range_violations(frame: pd.DataFrame) -> dict[str, int]:
    violations: dict[str, int] = {}
    for column, (low, high) in NUMERIC_RANGES.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        count = int(((values < low) | (values > high)).sum())
        if count:
            violations[column] = count
    return violations


def _unknown_categories(frame: pd.DataFrame) -> dict[str, list[str]]:
    unknown: dict[str, list[str]] = {}
    for column, allowed in ALLOWED_CATEGORIES.items():
        if column not in frame.columns:
            continue
        seen = set(frame[column].dropna().astype(str).unique())
        unexpected = sorted(seen - allowed)
        if unexpected:
            unknown[column] = unexpected
    return unknown


def validate_frame(
    frame: pd.DataFrame,
    *,
    require_target: bool = True,
    critical: bool = True,
) -> dict[str, Any]:
    """Validate a raw housing frame and return a machine-readable report.

    ``critical=True`` raises on structural problems (missing columns, duplicate ids,
    invalid target) but still only warns about business-rule outliers.
    """
    issues: list[str] = []
    required = list(RAW_COLUMNS)
    if not require_target:
        required = [column for column in required if column != TARGET]

    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        message = f"Missing required columns: {missing_columns}"
        if critical:
            raise DataValidationError(message)
        issues.append(message)

    if frame.empty:
        message = "Frame is empty"
        if critical:
            raise DataValidationError(message)
        issues.append(message)

    duplicates = 0
    if ID_COLUMN in frame.columns:
        duplicates = int(frame[ID_COLUMN].duplicated().sum())
        if duplicates and critical:
            raise DataValidationError(f"Duplicate {ID_COLUMN} values: {duplicates}")

    target_summary: dict[str, float] = {}
    if require_target and TARGET in frame.columns:
        target = pd.to_numeric(frame[TARGET], errors="coerce")
        nulls = int(target.isna().sum())
        if nulls and critical:
            raise DataValidationError(f"{TARGET} has {nulls} missing/invalid values")
        target_summary = {
            "min": float(target.min()),
            "mean": float(target.mean()),
            "max": float(target.max()),
            "median": float(target.median()),
        }

    violations = _range_violations(frame)
    unknown = _unknown_categories(frame)
    if violations:
        issues.append(f"Range violations (rows per column): {violations}")
    if unknown:
        issues.append(f"Unknown categories: {unknown}")

    missing_ratio = (
        frame.isna().mean().sort_values(ascending=False).head(15).round(4).to_dict()
        if len(frame)
        else {}
    )

    status = "ok"
    if violations or unknown or duplicates:
        status = "warning"
    if missing_columns or frame.empty or (require_target and target_summary.get("min", 1) is None):
        status = "failed"

    return {
        "status": status,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_ids": duplicates,
        "range_violations": violations,
        "unknown_categories": unknown,
        "missing_ratio_top15": missing_ratio,
        "target": target_summary,
        "issues": issues,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_training_data(frame: pd.DataFrame) -> dict[str, Any]:
    report = validate_frame(frame, require_target=True, critical=True)
    _write_report(report, data_config().path("paths.reports_dir") / "data_quality.json")
    logger.info("Data validation status=%s issues=%d", report["status"], len(report["issues"]))
    return report


def _write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
