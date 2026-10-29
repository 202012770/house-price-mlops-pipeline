from __future__ import annotations

import subprocess
import sys

import pytest

from house_price.serving.predictor import InputValidationError
from house_price.serving.schemas import HouseFeatures


def test_schema_accepts_nan_as_missing(sample_payload) -> None:
    payload = {**sample_payload, "LotFrontage": float("nan")}
    features = HouseFeatures(**payload)
    assert features.model_dump(exclude_none=True).get("LotFrontage") is None


def test_schema_accepts_integer_ms_subclass(sample_payload) -> None:
    features = HouseFeatures(**sample_payload, MSSubClass=50)
    assert features.MSSubClass == 50

    as_float = HouseFeatures(**sample_payload, MSSubClass=50.0)
    assert as_float.MSSubClass == 50


def test_predictor_single_and_batch(predictor, sample_payload) -> None:
    prediction = predictor.predict_records([sample_payload])
    assert prediction.shape == (1,)
    batch = predictor.predict_records([sample_payload, sample_payload])
    assert batch.shape == (2,)
    assert batch[0] == batch[1]


def test_predictor_rejects_unknown_field(predictor, sample_payload) -> None:
    with pytest.raises(InputValidationError):
        predictor.predict_records([{**sample_payload, "MadeUp": 1}])


def test_predictor_rejects_unknown_category(predictor, sample_payload) -> None:
    with pytest.raises(InputValidationError):
        predictor.predict_records([{**sample_payload, "Neighborhood": "Atlantis"}])


def test_predictor_rejects_empty(predictor) -> None:
    with pytest.raises(InputValidationError):
        predictor.predict_records([])


def test_predictor_info(predictor) -> None:
    info = predictor.info()
    assert info["model_name"] == "lightgbm"
    assert info["model_version"]
    assert info["test_metrics"]["r2"] > 0


def test_predictor_logging_disabled_is_safe(predictor, sample_payload) -> None:
    predictor.log_request([sample_payload], [123.0], latency_ms=1.0, endpoint="/test")


def test_api_does_not_import_training_only_dependencies() -> None:
    """The serving image installs core deps only; importing the app must not need them."""
    code = (
        "import sys; import house_price.serving.api; "
        "print(any(m in sys.modules for m in ['optuna', 'shap', 'mlflow']))"
    )
    output = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert output.strip() == "False"
