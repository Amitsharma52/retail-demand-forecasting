"""
src/feature_selection.py — Advanced Feature Selection & Macro Trend Engineering.

Implements automated feature importance ranking and collinearity pruning:
1. Low-Variance Filter (drops near-constant features)
2. Collinear Redundancy Pruning (drops pairwise Pearson |r| > threshold)
3. LightGBM Gain Importance Feature Selection (selects Top-K informative features)
4. Macro Trend Features (oil momentum, holiday proximity, biweekly salary paydays)
5. Strict Leakage Prevention: All rolling windows and lags enforce .shift(1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionReport:
    """Diagnostic report on feature pruning and importance ranking."""
    initial_feature_count: int
    selected_feature_count: int
    dropped_low_variance: List[str]
    dropped_collinear: List[str]
    selected_features: List[str]
    feature_importances: dict[str, float]


def prune_low_variance_features(
    X: pd.DataFrame,
    threshold: float = 1e-4,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove numerical features with variance below threshold."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    variances = X[numeric_cols].var()
    low_var_cols = list(variances[variances <= threshold].index)
    X_pruned = X.drop(columns=low_var_cols)
    return X_pruned, low_var_cols


def prune_collinear_features(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove collinear numerical features with pairwise |correlation| > threshold.
    Retains the feature with higher variance.
    """
    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    if len(numeric_cols) < 2:
        return X.copy(), []

    corr_matrix = X[numeric_cols].corr().abs()
    variances = X[numeric_cols].var()
    dropped_cols: set[str] = set()

    for i in range(len(numeric_cols)):
        col_i = numeric_cols[i]
        if col_i in dropped_cols:
            continue
        for j in range(i + 1, len(numeric_cols)):
            col_j = numeric_cols[j]
            if col_j in dropped_cols:
                continue

            if corr_matrix.loc[col_i, col_j] >= threshold:
                # Drop the one with lower variance
                drop_target = col_j if variances[col_i] >= variances[col_j] else col_i
                dropped_cols.add(drop_target)
                if drop_target == col_i:
                    break

    X_pruned = X.drop(columns=list(dropped_cols))
    return X_pruned, list(dropped_cols)


def select_top_features_by_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    top_k: int = 15,
    random_state: int = 42,
) -> Tuple[List[str], dict[str, float]]:
    """
    Train a fast LightGBM regressor to rank features by total split gain,
    returning the top_k most informative feature names.
    """
    X_clean = X.copy()
    for col in X_clean.select_dtypes(include=["category", "object"]).columns:
        X_clean[col] = X_clean[col].astype("category")

    model = LGBMRegressor(
        n_estimators=30,
        max_depth=4,
        learning_rate=0.1,
        random_state=random_state,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(X_clean, y)

    importances = model.booster_.feature_importance(importance_type="gain")
    feat_imp = {feat: float(imp) for feat, imp in zip(X_clean.columns, importances)}
    sorted_feats = sorted(feat_imp.items(), key=lambda kv: kv[1], reverse=True)

    k = min(top_k, len(sorted_feats))
    top_features = [feat for feat, _ in sorted_feats[:k]]
    return top_features, feat_imp


def run_automated_feature_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    variance_threshold: float = 1e-4,
    correlation_threshold: float = 0.95,
    top_k: int = 15,
) -> Tuple[pd.DataFrame, FeatureSelectionReport]:
    """
    Execute full feature selection pipeline:
    1. Prune zero/low variance
    2. Prune redundant collinear features
    3. Select top_k most predictive features via LightGBM gain
    """
    initial_count = X.shape[1]

    # 1. Variance
    X_v, dropped_var = prune_low_variance_features(X, threshold=variance_threshold)

    # 2. Correlation
    X_c, dropped_corr = prune_collinear_features(X_v, threshold=correlation_threshold)

    # 3. Tree-gain importance
    top_feats, importances = select_top_features_by_importance(X_c, y, top_k=top_k)

    X_selected = X_c[top_feats].copy()

    report = FeatureSelectionReport(
        initial_feature_count=initial_count,
        selected_feature_count=len(top_feats),
        dropped_low_variance=dropped_var,
        dropped_collinear=dropped_corr,
        selected_features=top_feats,
        feature_importances=importances,
    )

    return X_selected, report


def add_macro_and_calendar_features(
    df: pd.DataFrame,
    oil_series: Optional[pd.Series] = None,
    holiday_dates: Optional[List[pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Add domain-specific retail features:
    - Biweekly Ecuadorian paydays (15th and end-of-month)
    - Proximity to nearest public holiday
    - Oil price 7-day relative momentum: (oil_t - oil_roll7) / oil_roll7 (shifted by 1)
    """
    df_out = df.copy()
    dates = pd.to_datetime(df_out["date"])

    # 1. Biweekly Payday flag (15th and last day of month)
    is_mid_month_payday = dates.dt.day.isin([14, 15, 16]).astype(int)
    is_end_month_payday = dates.dt.is_month_end.astype(int)
    df_out["is_payday"] = np.maximum(is_mid_month_payday, is_end_month_payday)

    # 2. Holiday Proximity
    if holiday_dates:
        h_dates = pd.DatetimeIndex(sorted(pd.to_datetime(holiday_dates)))
        # Minimum absolute day distance to nearest holiday
        distances = []
        for d in dates:
            diffs = np.abs((h_dates - d).days)
            distances.append(int(np.min(diffs)) if len(diffs) > 0 else 99)
        df_out["days_to_holiday"] = distances
        df_out["is_holiday_week"] = (df_out["days_to_holiday"] <= 3).astype(int)
    else:
        df_out["days_to_holiday"] = 99
        df_out["is_holiday_week"] = 0

    # 3. Oil Momentum (shifted by 1 to strictly prevent leakage)
    if oil_series is not None and not oil_series.empty:
        oil_s = oil_series.sort_index()
        oil_roll7 = oil_s.shift(1).rolling(7, min_periods=1).mean()
        oil_mom = (oil_s.shift(1) - oil_roll7) / np.maximum(1.0, oil_roll7)
        df_out["oil_momentum_7d"] = dates.map(oil_mom).fillna(0.0)
    else:
        df_out["oil_momentum_7d"] = 0.0

    return df_out
