"""FastAPI application exposing health, metadata and prediction endpoints."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from house_price.config import data_config
from house_price.data.schema import (
    ALLOWED_CATEGORIES,
    CORE_API_FEATURES,
    OPTIONAL_API_FEATURES,
)
from house_price.logging_utils import configure_logging, get_logger
from house_price.serving.predictor import (
    InputValidationError,
    Predictor,
    get_predictor,
)
from house_price.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    HouseFeatures,
    ModelInfoResponse,
    PredictionResponse,
)

configure_logging()
logger = get_logger("house_price.api")

REQUEST_COUNT = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["path"])
PREDICTIONS = Counter("house_price_predictions_total", "Predictions served")
PREDICTED_VALUE = Histogram(
    "house_price_predicted_value",
    "Distribution of predicted prices",
    buckets=[50_000, 100_000, 150_000, 200_000, 300_000, 400_000, 500_000, 750_000],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = get_predictor()
    app.state.started_at = time.time()
    app.state.predictor = predictor
    yield


app = FastAPI(
    title="House Price Prediction API",
    description=(
        "Production-style regression service for Ames house sale prices. "
        "Core features are required; all remaining fields are optional and imputed "
        "from the training distribution."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next: Any) -> Response:
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.perf_counter()
    response: Response
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - started) * 1000
        REQUEST_COUNT.labels(request.method, request.url.path, "500").inc()
        REQUEST_LATENCY.labels(request.url.path).observe(latency_ms / 1000)
        logger.exception(
            "Unhandled error",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": 500,
                "latency_ms": round(latency_ms, 2),
            },
        )
        raise
    latency_ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = request_id
    REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(latency_ms / 1000)
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
        },
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="validation_error",
            detail=str(exc.errors()),
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(),
    )


@app.exception_handler(InputValidationError)
async def input_exception_handler(request: Request, exc: InputValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="input_error",
            detail=str(exc),
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(),
    )


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "service": "house-price-prediction-api",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info",
        "predict": "/predict",
        "batch_predict": "/predict/batch",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    predictor: Predictor = request.app.state.predictor
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.pipeline is not None,
        model_version=predictor.model_version,
        uptime_seconds=round(time.time() - request.app.state.started_at, 2),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
def model_info(request: Request) -> ModelInfoResponse:
    predictor: Predictor = request.app.state.predictor
    return ModelInfoResponse(**predictor.info())


@app.get("/features", tags=["meta"])
def features() -> dict[str, Any]:
    return {
        "required": list(CORE_API_FEATURES),
        "optional": list(OPTIONAL_API_FEATURES),
        "allowed_categories": {
            column: sorted(values) for column, values in ALLOWED_CATEGORIES.items()
        },
        "note": "Optional fields default to the training median (numeric) or Missing (categorical).",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(payload: HouseFeatures, request: Request) -> PredictionResponse:
    predictor: Predictor = request.app.state.predictor
    records = [payload.model_dump(exclude_none=True)]
    started = time.perf_counter()
    predictions = predictor.predict_records(records)
    latency_ms = (time.perf_counter() - started) * 1000

    price = float(predictions[0])
    PREDICTIONS.inc()
    PREDICTED_VALUE.observe(price)
    predictor.log_request(records, [price], latency_ms=latency_ms, endpoint="/predict")
    return PredictionResponse(
        predicted_price=price,
        model_version=predictor.model_version,
        request_id=getattr(request.state, "request_id", ""),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(payload: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    predictor: Predictor = request.app.state.predictor
    records = [instance.model_dump(exclude_none=True) for instance in payload.instances]
    started = time.perf_counter()
    predictions = predictor.predict_records(records)
    latency_ms = (time.perf_counter() - started) * 1000

    values = [float(value) for value in predictions]
    PREDICTIONS.inc(len(values))
    for value in values:
        PREDICTED_VALUE.observe(value)
    predictor.log_request(records, values, latency_ms=latency_ms, endpoint="/predict/batch")
    return BatchPredictionResponse(
        predictions=values,
        count=len(values),
        model_version=predictor.model_version,
        request_id=getattr(request.state, "request_id", ""),
    )


@app.get("/metrics", tags=["meta"])
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/config", tags=["meta"], include_in_schema=False)
def config_summary() -> dict[str, Any]:
    cfg = data_config()
    return {
        "target": cfg.dataset.target,
        "test_size": cfg.split.test_size,
        "target_transform": cfg.features.target_transform,
    }
