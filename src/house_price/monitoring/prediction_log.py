"""Lightweight SQLite log of served requests, used as the drift input."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from house_price.config import resolve_path, training_config

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    n_rows INTEGER NOT NULL,
    prediction_mean REAL NOT NULL,
    prediction_min REAL NOT NULL,
    prediction_max REAL NOT NULL,
    input_digest TEXT NOT NULL,
    features_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
"""


class PredictionLog:
    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            db_path = resolve_path(training_config().serving.prediction_log_db)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def log(
        self,
        features: list[dict[str, Any]],
        predictions: list[float],
        *,
        latency_ms: float,
        model_version: str,
        endpoint: str = "/predict",
    ) -> None:
        if not features:
            return
        payload = json.dumps(features, ensure_ascii=False, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    created_at, model_version, endpoint, latency_ms, n_rows,
                    prediction_mean, prediction_min, prediction_max,
                    input_digest, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    model_version,
                    endpoint,
                    float(latency_ms),
                    len(features),
                    float(sum(predictions) / len(predictions)),
                    float(min(predictions)),
                    float(max(predictions)),
                    digest,
                    payload,
                ),
            )

    def recent_features(self, limit: int = 1000) -> pd.DataFrame:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT features_json FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        records: list[dict[str, Any]] = []
        for (payload,) in rows:
            records.extend(json.loads(payload))
        return pd.DataFrame(records)

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
