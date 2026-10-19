"""
src/forecasting.py — Forecast generation for future dates.

Produces 7-day ahead forecasts (configurable) for a selected store and
product family using the saved LightGBM quantile models.

Crossing correction is applied at prediction time: displayed lower bound
never exceeds the median, and median never exceeds the upper bound.

Documented assumption: when future promotion data is not provided,
onpromotion = 0 is assumed for all future days.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.features import build_future_features, FEATURE_COLS

logger = logging.getLogger(__name__)


def generate_forecast(
    history_df: pd.DataFrame,
    q10_model: LGBMRegressor,
    q50_model: LGBMRegressor,
    q90_model: LGBMRegressor,
    meta: dict,
    store_nbr: int,
    family: str,
    horizon: int = config.FORECAST_HORIZON,
    future_promotions: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Generate horizon-day forecasts for a single (store, family) pair.

    Strategy (recursive multi-step)
    --------------------------------
    1. Build feature rows for future_date[0] using real history.
    2. Predict Q10, Q50, Q90 for that day.
    3. Append Q50 as a synthetic 'sales' observation for next day's features.
    4. Repeat until all horizon days are covered.

    Assumption
    ----------
    When future_promotions is None, onpromotion = 0 is assumed for all
    future days.  This limitation is displayed in the dashboard.

    Returns
    -------
    pd.DataFrame with columns:
        date, pred_q10, pred_q50, pred_q90, is_predicted
    """
    feature_cols = meta.get("feature_cols", FEATURE_COLS)
    if not feature_cols:
        raise ValueError("feature_cols is empty — model metadata may be corrupted.")

    last_date     = history_df[
        (history_df["store_nbr"] == store_nbr) &
        (history_df["family"]    == family)
    ]["date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    # We'll build features iteratively, appending predictions to history
    working_history = history_df.copy()
    results: list[dict] = []

    for target_date in future_dates:
        # Build features for this single future date
        feat_row = build_future_features(
            history_df=working_history,
            future_dates=pd.DatetimeIndex([target_date]),
            store_nbr=store_nbr,
            family=family,
            future_promotions=future_promotions,
            store_map=meta.get("store_map"),
            family_map=meta.get("family_map"),
        )

        # Only keep columns the model expects; fill missing with NaN
        X = feat_row.reindex(columns=feature_cols)

        # Predict
        q10 = float(np.clip(q10_model.predict(X)[0], 0, None))
        q50 = float(np.clip(q50_model.predict(X)[0], 0, None))
        q90 = float(np.clip(q90_model.predict(X)[0], 0, None))

        # Crossing correction
        q10, q50, q90 = _fix_crossing_single(q10, q50, q90)

        results.append({
            "date"        : target_date,
            "pred_q10"    : q10,
            "pred_q50"    : q50,
            "pred_q90"    : q90,
            "is_predicted": True,
        })

        # Append Q50 prediction as synthetic history for the next step
        synthetic_row = pd.DataFrame([{
            "date"            : target_date,
            "store_nbr"       : store_nbr,
            "family"          : family,
            "sales"           : q50,           # ← predicted, not observed
            "onpromotion"     : 0,
            "is_holiday"      : 0,
            "sales_is_missing": False,
            "is_predicted"    : True,
        }])
        working_history = pd.concat([working_history, synthetic_row], ignore_index=True)

    forecast_df = pd.DataFrame(results)
    logger.info(
        "Generated %d-day forecast for store=%s, family=%s.",
        horizon, store_nbr, family,
    )
    return forecast_df


def _fix_crossing_single(q10: float, q50: float, q90: float) -> tuple[float, float, float]:
    """Ensure q10 ≤ q50 ≤ q90 for a single prediction."""
    vals = sorted([q10, q50, q90])
    return vals[0], vals[1], vals[2]


def seasonal_naive_forecast(
    history_df: pd.DataFrame,
    store_nbr: int,
    family: str,
    horizon: int = config.FORECAST_HORIZON,
    season_period: int = 7,
) -> pd.DataFrame:
    """
    Seasonal-naive baseline: forecast = sales from exactly season_period days ago.

    Used as a benchmark — not a machine learning model.
    """
    series = history_df[
        (history_df["store_nbr"] == store_nbr) &
        (history_df["family"]    == family)
    ].sort_values("date")[["date", "sales"]].set_index("date")

    last_date = series.index.max()
    results = []
    for h in range(1, horizon + 1):
        target_date = last_date + pd.Timedelta(days=h)
        ref_date    = target_date - pd.Timedelta(days=season_period)
        baseline_val = float(series.loc[ref_date, "sales"]) if ref_date in series.index else np.nan
        results.append({"date": target_date, "baseline": baseline_val})

    return pd.DataFrame(results)
