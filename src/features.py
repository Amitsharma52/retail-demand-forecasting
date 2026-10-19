"""
src/features.py — Leakage-free feature engineering for daily retail demand.

DESIGN PRINCIPLE
----------------
All lag and rolling features are computed AFTER grouping by (store_nbr, family)
and shifting by at least 1 day.  This ensures that the current day's target
sales value never appears in any feature used to predict it.

store_nbr and family are encoded as integers using consistent label-encoding
stored in model metadata so that training and inference codes always match.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# Global — populated by build_features(); also stored in model metadata
FEATURE_COLS: list[str] = []
STORE_MAP:    dict = {}    # original store_nbr → integer code
FAMILY_MAP:   dict = {}    # original family string → integer code


# ── Calendar helpers ────────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add day-of-week, month, and cyclic sine/cosine calendar features."""
    df = df.copy()
    df["day_of_week"]  = df[date_col].dt.dayofweek          # 0=Mon … 6=Sun
    df["month"]        = df[date_col].dt.month
    df["day_of_month"] = df[date_col].dt.day
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["quarter"]      = df[date_col].dt.quarter

    # Cyclic encoding so that day 6→0 wraps smoothly for the model
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    return df


# ── Lag and rolling features ───────────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    lag_days: list[int] = config.LAG_DAYS,
    group_cols: list[str] = ("store_nbr", "family"),
    target_col: str = "sales",
) -> pd.DataFrame:
    """
    Add lagged sales features grouped by (store_nbr, family).

    Lags are computed with .shift(n) which moves the series forward by n
    positions — so ``lag_1`` for row t is the sales value at t-1.
    """
    df = df.copy()
    for lag in lag_days:
        df[f"lag_{lag}"] = df.groupby(list(group_cols))[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = config.ROLLING_WINDOWS,
    group_cols: list[str] = ("store_nbr", "family"),
    target_col: str = "sales",
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Add rolling mean, rolling std, and EWMA features.

    The rolling window is applied to the SHIFTED series (shift(1)) so that the
    current day's sales do not enter any rolling statistic.
    """
    df = df.copy()

    for w in windows:
        grp = df.groupby(list(group_cols))[target_col]
        df[f"roll_mean_{w}"] = grp.transform(
            lambda s: s.shift(1).rolling(w, min_periods=min_periods).mean()
        )
        df[f"roll_std_{w}"] = grp.transform(
            lambda s: s.shift(1).rolling(w, min_periods=min_periods).std()
        )

    df["ewma_7"] = df.groupby(list(group_cols))[target_col].transform(
        lambda s: s.shift(1).ewm(span=config.EWMA_SPAN, min_periods=1).mean()
    )

    return df


def add_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add promotion-related features."""
    df = df.copy()
    df["is_on_promotion"] = (df["onpromotion"] > 0).astype(int)
    df["promo_roll_7"] = df.groupby(["store_nbr", "family"])["is_on_promotion"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).sum()
    )
    return df


# ── Label-encoding helpers ─────────────────────────────────────────────────────

def _build_store_map(df: pd.DataFrame) -> dict:
    """Build a mapping from original store_nbr values to integer codes."""
    stores = sorted(df["store_nbr"].unique().tolist())
    return {s: i for i, s in enumerate(stores)}


def _build_family_map(df: pd.DataFrame) -> dict:
    """Build a mapping from original family strings to integer codes."""
    families = sorted(df["family"].unique().tolist())
    return {f: i for i, f in enumerate(families)}


def apply_label_encoding(
    df: pd.DataFrame,
    store_map: dict,
    family_map: dict,
) -> pd.DataFrame:
    """
    Apply consistent integer label encoding to store_nbr and family columns.

    Parameters
    ----------
    df         : DataFrame whose store_nbr / family columns need encoding.
    store_map  : Mapping built from the full training dataset.
    family_map : Mapping built from the full training dataset.
    """
    df = df.copy()
    df["store_nbr"] = df["store_nbr"].map(store_map)
    df["family"]    = df["family"].map(family_map)

    # Fill any unmapped values with -1 (unknown category)
    df["store_nbr"] = df["store_nbr"].fillna(-1).astype(int)
    df["family"]    = df["family"].fillna(-1).astype(int)

    return df


# ── Master feature builder ─────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    is_training: bool = True,
) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to a sorted DataFrame.

    Also populates module-level FEATURE_COLS, STORE_MAP, FAMILY_MAP globals
    which are stored in model metadata for consistent inference.
    """
    global FEATURE_COLS, STORE_MAP, FAMILY_MAP

    # Build encoding maps BEFORE any transformation (use original values)
    STORE_MAP  = _build_store_map(df)
    FAMILY_MAP = _build_family_map(df)

    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_promotion_features(df)

    # Apply consistent label encoding
    df = apply_label_encoding(df, STORE_MAP, FAMILY_MAP)

    # Define the exact feature set
    feature_candidates = [
        # Calendar
        "day_of_week", "month", "day_of_month", "week_of_year", "quarter",
        "dow_sin", "dow_cos", "month_sin", "month_cos",
        # Lags
        *[f"lag_{d}" for d in config.LAG_DAYS],
        # Rolling
        *[f"roll_mean_{w}" for w in config.ROLLING_WINDOWS],
        *[f"roll_std_{w}"  for w in config.ROLLING_WINDOWS],
        "ewma_7",
        # Promotion
        "is_on_promotion", "promo_roll_7",
        # Encoded identifiers (integers)
        "store_nbr", "family",
    ]

    FEATURE_COLS = [c for c in feature_candidates if c in df.columns]

    if is_training:
        before = len(df)
        df = df.dropna(subset=["sales"])
        dropped = before - len(df)
        if dropped:
            logger.info("Dropped %d rows with NaN sales for training.", dropped)

        # Drop rows where lag_1 is NaN (insufficient history to train on)
        lag_cols = [f"lag_{d}" for d in config.LAG_DAYS if f"lag_{d}" in df.columns]
        if lag_cols:
            df = df.dropna(subset=[lag_cols[0]])

    return df


# ── Future feature builder ─────────────────────────────────────────────────────

def build_future_features(
    history_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    store_nbr: int,
    family: str,
    future_promotions: Optional[pd.Series] = None,
    store_map: Optional[dict] = None,
    family_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Construct feature rows for future dates using only historical information.

    Parameters
    ----------
    history_df        : Historical data for the single (store, family) series.
    future_dates      : pd.DatetimeIndex of dates to forecast.
    store_nbr         : Store identifier (original value, e.g. 1).
    family            : Product family string (e.g. 'GROCERY I').
    future_promotions : Optional Series indexed by date with promotion values.
                        If None, 0 is assumed (documented assumption).
    store_map         : Encoding map from model metadata. Falls back to
                        module-level STORE_MAP if None.
    family_map        : Encoding map from model metadata.

    Returns
    -------
    pd.DataFrame with one row per future date and all feature columns.
    """
    # Use provided maps or fall back to module globals
    _smap = store_map  or STORE_MAP
    _fmap = family_map or FAMILY_MAP

    # Filter to the target series using original (un-encoded) values
    mask = (history_df["store_nbr"] == store_nbr) & (history_df["family"] == family)
    series = history_df[mask].copy()

    if series.empty:
        raise ValueError(
            f"No history found for store={store_nbr}, family={family}."
        )

    series = series.sort_values("date").reset_index(drop=True)

    # Build future skeleton rows (with original store_nbr / family values)
    future_rows = []
    for d in future_dates:
        promo_val = 0
        if future_promotions is not None and d in future_promotions.index:
            promo_val = int(future_promotions[d])

        future_rows.append({
            "date"            : d,
            "store_nbr"       : store_nbr,   # original value — encoded below
            "family"          : family,       # original value — encoded below
            "sales"           : np.nan,
            "onpromotion"     : promo_val,
            "is_holiday"      : 0,
            "is_predicted"    : False,
            "sales_is_missing": True,
        })

    future_df = pd.DataFrame(future_rows)
    combined  = pd.concat([series, future_df], ignore_index=True)

    # Compute features while store_nbr/family are still original values
    # (add_lag_features groups by them so the grouping must be correct)
    combined = add_calendar_features(combined)
    combined = add_lag_features(combined)
    combined = add_rolling_features(combined)
    combined = add_promotion_features(combined)

    # Now apply consistent label encoding
    if _smap:
        combined = apply_label_encoding(combined, _smap, _fmap)
    else:
        # Fallback: use cat.codes (may differ from training if fewer categories)
        combined["store_nbr"] = combined["store_nbr"].astype("category").cat.codes
        combined["family"]    = combined["family"].astype("category").cat.codes
        combined["store_nbr"] = combined["store_nbr"].astype(int)
        combined["family"]    = combined["family"].astype(int)

    # Extract only the future rows
    future_features = combined[combined["date"].isin(future_dates)].copy()
    future_features["is_predicted"] = True

    return future_features
