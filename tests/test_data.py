from __future__ import annotations

import pandas as pd
import pytest

from house_price.data.validate import DataValidationError, validate_frame


def test_validate_frame_reports_ok(synthetic_frame: pd.DataFrame) -> None:
    report = validate_frame(synthetic_frame)
    assert report["status"] in {"ok", "warning"}
    assert report["rows"] == len(synthetic_frame)
    assert report["duplicate_ids"] == 0
    assert report["target"]["min"] > 0


def test_missing_column_raises(synthetic_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        validate_frame(synthetic_frame.drop(columns=["GrLivArea"]))


def test_duplicate_ids_raise(synthetic_frame: pd.DataFrame) -> None:
    duplicated = pd.concat([synthetic_frame, synthetic_frame.head(2)], ignore_index=True)
    with pytest.raises(DataValidationError):
        validate_frame(duplicated)


def test_null_target_raises(synthetic_frame: pd.DataFrame) -> None:
    broken = synthetic_frame.copy()
    broken.loc[0, "SalePrice"] = None
    with pytest.raises(DataValidationError):
        validate_frame(broken)


def test_unknown_category_is_reported(synthetic_frame: pd.DataFrame) -> None:
    frame = synthetic_frame.copy()
    frame.loc[0, "Neighborhood"] = "Atlantis"
    report = validate_frame(frame)
    assert "Atlantis" in report["unknown_categories"].get("Neighborhood", [])
    assert report["status"] == "warning"
