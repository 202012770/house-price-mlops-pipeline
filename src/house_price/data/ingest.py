"""Dataset ingestion with a local cache."""

from __future__ import annotations

import gzip
import time
from pathlib import Path

import pandas as pd
from scipy.io import arff
from sklearn.datasets import fetch_openml

from house_price.config import data_config
from house_price.data.schema import RAW_COLUMNS
from house_price.logging_utils import get_logger

logger = get_logger(__name__)


def ingest_data(force: bool = False) -> pd.DataFrame:
    """Load the Ames housing dataset, downloading it once into ``data/cache``.

    The cached raw CSV lives at ``data/raw/house_prices.csv`` so the rest of the
    pipeline never needs network access.
    """
    cfg = data_config()
    raw_path = cfg.path("paths.raw")
    if raw_path.exists() and not force:
        logger.info("Loading cached raw dataset from %s", raw_path)
        return _normalise_categories(pd.read_csv(raw_path))

    cache_dir = cfg.path("paths.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    frame = _normalise_categories(
        _download_frame(cache_dir, cfg.dataset.openml_name, cfg.dataset.version)
    )

    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    frame = frame.loc[:, list(RAW_COLUMNS)].copy()
    frame[cfg.dataset.id_column] = frame[cfg.dataset.id_column].astype("int64")
    frame[cfg.dataset.target] = pd.to_numeric(frame[cfg.dataset.target], errors="coerce")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_path, index=False)
    logger.info("Saved raw dataset to %s (%s rows)", raw_path, len(frame))
    return frame


def _download_frame(cache_dir: Path, name: str, version: int) -> pd.DataFrame:
    """Fetch from OpenML with retries, falling back to a cached ARFF archive."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            logger.info("Downloading %s v%s from OpenML (attempt %d)", name, version, attempt + 1)
            bunch = fetch_openml(
                name=name,
                version=version,
                as_frame=True,
                parser="auto",
                data_home=str(cache_dir),
                n_retries=3,
                delay=3,
            )
            return bunch.frame
        except Exception as exc:  # network resets are common on flaky links
            last_error = exc
            logger.warning("OpenML download failed: %s", exc)
            time.sleep(5)

    cached = sorted(cache_dir.glob("openml/**/download/*.gz"))
    if cached:
        logger.warning("Falling back to cached ARFF archive %s", cached[0])
        return _read_arff(cached[0])

    raise RuntimeError(f"Could not obtain dataset after retries: {last_error}")


def _normalise_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip stray quotes that OpenML's ARFF export adds around some categories."""
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].str.replace(r"^'(.*)'$", r"\1", regex=True)
    return frame


def _read_arff(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data, _ = arff.loadarff(handle)
    frame = pd.DataFrame(data)
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(
                lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
            )
    return frame


def load_raw_data() -> pd.DataFrame:
    """Load the raw dataset, ingesting it first if needed."""
    cfg = data_config()
    raw_path = cfg.path("paths.raw")
    if not raw_path.exists():
        return ingest_data()
    return _normalise_categories(pd.read_csv(raw_path))


def split_data(
    frame: pd.DataFrame,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic train/test split, stratified on price quintiles."""
    from sklearn.model_selection import train_test_split

    cfg = data_config()
    target = cfg.dataset.target
    test_size = cfg.split.test_size if test_size is None else test_size
    random_state = cfg.split.random_state if random_state is None else random_state

    strat = pd.qcut(frame[target], q=5, labels=False, duplicates="drop")
    train_df, test_df = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[Path, Path]:
    cfg = data_config()
    processed_dir = cfg.path("paths.processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


if __name__ == "__main__":
    data = ingest_data()
    print(f"rows={len(data)} columns={data.shape[1]}")
