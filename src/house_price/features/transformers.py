"""Custom sklearn transformers used by the model pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

from house_price.data.schema import ID_COLUMN, TARGET


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _stringify(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def cast_and_fill_missing(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast category columns to strings and turn NaN into a literal category.

    Numeric-looking codes such as ``MSSubClass`` arrive as ``20.0`` from pandas
    but were trained as ``"20"``; normalising keeps the category vocabulary stable.
    """
    return pd.DataFrame(
        {
            column: frame[column].map(
                lambda value: "Missing" if pd.isna(value) else _stringify(value)
            )
            for column in frame.columns
        },
        index=frame.index,
    )


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain features derived from the Ames housing schema.

    These are the interactions that top public solutions rely on: total living
    area, bathroom count, house age and quality-area interactions.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureEngineer:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for column in (ID_COLUMN, TARGET):
            df = df.drop(columns=[column], errors="ignore")

        total_bsmt = _num(df, "TotalBsmtSF")
        first_floor = _num(df, "1stFlrSF")
        second_floor = _num(df, "2ndFlrSF")
        gr_liv = _num(df, "GrLivArea")
        rooms = _num(df, "TotRmsAbvGrd")
        year_built = _num(df, "YearBuilt")
        year_sold = _num(df, "YrSold")
        overall_qual = _num(df, "OverallQual")

        df["TotalSF"] = total_bsmt + first_floor + second_floor
        df["TotalBath"] = (
            _num(df, "FullBath")
            + 0.5 * _num(df, "HalfBath")
            + _num(df, "BsmtFullBath")
            + 0.5 * _num(df, "BsmtHalfBath")
        )
        df["TotalPorchSF"] = (
            _num(df, "WoodDeckSF")
            + _num(df, "OpenPorchSF")
            + _num(df, "EnclosedPorch")
            + _num(df, "3SsnPorch")
            + _num(df, "ScreenPorch")
        )
        df["TotalFinSF"] = _num(df, "BsmtFinSF1") + _num(df, "BsmtFinSF2")
        df["HouseAge"] = year_sold - year_built
        df["RemodAge"] = year_sold - _num(df, "YearRemodAdd")
        df["GarageAge"] = year_sold - _num(df, "GarageYrBlt")
        df["LivAreaPerRoom"] = gr_liv / (rooms + 1.0)
        df["QualSF"] = overall_qual * gr_liv
        df["BhmsRatio"] = total_bsmt / (gr_liv + 1.0)
        df["GarageRatio"] = _num(df, "GarageArea") / (gr_liv + 1.0)
        df["PorchRatio"] = df["TotalPorchSF"] / (gr_liv + 1.0)
        df["QualCondScore"] = overall_qual * _num(df, "OverallCond")
        df["LotAreaPerRoom"] = _num(df, "LotArea") / (rooms + 1.0)

        return df

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self._columns_after_transform(), dtype=object)

    def _columns_after_transform(self) -> list[str]:
        base = [c for c in self.feature_names_in_ if c not in {ID_COLUMN, TARGET}]
        engineered = [
            "TotalSF",
            "TotalBath",
            "TotalPorchSF",
            "TotalFinSF",
            "HouseAge",
            "RemodAge",
            "GarageAge",
            "LivAreaPerRoom",
            "QualSF",
            "BhmsRatio",
            "GarageRatio",
            "PorchRatio",
            "QualCondScore",
            "LotAreaPerRoom",
        ]
        return base + engineered


class MappedOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Maps ordinal quality labels to integers; missing means the feature is absent."""

    def __init__(
        self,
        mapping: dict[str, dict[str, float]] | None = None,
        defaults: dict[str, float] | None = None,
    ) -> None:
        self.mapping = mapping
        self.defaults = defaults

    def fit(self, X: pd.DataFrame, y: Any = None) -> MappedOrdinalEncoder:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.columns_ = [column for column in (self.mapping or {}) if column in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "columns_")
        mapping = self.mapping or {}
        defaults = self.defaults or {}
        result: dict[str, pd.Series] = {}
        for column in self.columns_:
            mapped = X[column].map(mapping[column])
            mapped = pd.to_numeric(mapped, errors="coerce")
            result[column] = mapped.fillna(float(defaults.get(column, 0.0))).astype("float64")
        return pd.DataFrame(result, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "columns_")
        return np.asarray(self.columns_, dtype=object)


class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target encoding with out-of-fold values during training.

    ``fit_transform`` computes encodings on held-out folds so the training rows
    never see their own target; ``fit`` then rebuilds the mapping on all data for
    inference. Unknown categories fall back to the global prior mean.
    """

    def __init__(
        self,
        smoothing: float = 10.0,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any) -> OutOfFoldTargetEncoder:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.columns_ = list(X.columns)
        target = np.asarray(y, dtype="float64")
        self.prior_ = float(np.mean(target))
        self.mappings_ = self._compute_mappings(X, target)
        return self

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **fit_params: Any) -> pd.DataFrame:
        if y is None:
            raise ValueError("OutOfFoldTargetEncoder requires y during fit_transform")
        target = np.asarray(y, dtype="float64")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.columns_ = list(X.columns)

        encoded = np.full((len(X), len(self.columns_)), np.nan)
        folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, valid_idx in folds.split(X):
            mappings, prior = self._compute_mappings(
                X.iloc[train_idx], target[train_idx], return_prior=True
            )
            for position, column in enumerate(self.columns_):
                series = X[column].iloc[valid_idx].astype(str)
                encoded[valid_idx, position] = series.map(mappings[column]).fillna(prior)

        self.prior_ = float(np.mean(target))
        self.mappings_ = self._compute_mappings(X, target)
        return pd.DataFrame(
            encoded,
            index=X.index,
            columns=[f"{column}_target_enc" for column in self.columns_],
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "mappings_")
        result = {}
        for column in self.columns_:
            series = X[column].astype(str)
            result[f"{column}_target_enc"] = (
                series.map(self.mappings_[column]).fillna(self.prior_).astype("float64")
            )
        return pd.DataFrame(result, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "columns_")
        return np.asarray([f"{column}_target_enc" for column in self.columns_], dtype=object)

    def _compute_mappings(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        return_prior: bool = False,
    ) -> Any:
        prior = float(np.mean(y))
        mappings: dict[str, pd.Series] = {}
        for column in self.columns_:
            stats = (
                pd.DataFrame({"category": X[column].astype(str).to_numpy(), "target": y})
                .groupby("category")["target"]
                .agg(["sum", "count"])
            )
            smoothed = (stats["sum"] + prior * self.smoothing) / (stats["count"] + self.smoothing)
            mappings[column] = smoothed
        if return_prior:
            return mappings, prior
        return mappings
