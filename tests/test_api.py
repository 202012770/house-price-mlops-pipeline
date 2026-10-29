from __future__ import annotations


def test_health(api_client) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_version"] == "2026-01-01T00:00:00+00:00"


def test_model_info(api_client) -> None:
    response = api_client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "lightgbm"
    assert body["training_rows"] > 0


def test_features_endpoint(api_client) -> None:
    response = api_client.get("/features")
    assert response.status_code == 200
    body = response.json()
    assert "OverallQual" in body["required"]
    assert "Neighborhood" in body["allowed_categories"]


def test_predict_returns_price(api_client, sample_payload) -> None:
    response = api_client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["currency"] == "USD"
    assert 35_000 <= body["predicted_price"] <= 800_000
    assert response.headers["x-request-id"]


def test_predict_accepts_core_fields_only(api_client, sample_payload) -> None:
    response = api_client.post("/predict", json=sample_payload)
    assert response.status_code == 200


def test_predict_rejects_out_of_range(api_client, sample_payload) -> None:
    payload = {**sample_payload, "OverallQual": 99}
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_unknown_field(api_client, sample_payload) -> None:
    response = api_client.post("/predict", json={**sample_payload, "MadeUp": 1})
    assert response.status_code == 422


def test_predict_rejects_unknown_category(api_client, sample_payload) -> None:
    payload = {**sample_payload, "Neighborhood": "Atlantis"}
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Atlantis" in response.json()["detail"]


def test_batch_predict(api_client, sample_payload) -> None:
    response = api_client.post(
        "/predict/batch", json={"instances": [sample_payload, sample_payload]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_batch_rejects_empty(api_client) -> None:
    response = api_client.post("/predict/batch", json={"instances": []})
    assert response.status_code == 422


def test_metrics_endpoint(api_client) -> None:
    response = api_client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
