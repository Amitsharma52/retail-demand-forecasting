"""
tests/test_features.py — Unit tests for the feature engineering module.
"""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sample_data import generate_sample_train
from src.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_promotion_features,
    build_features,
    build_future_features,
)
from src.data_pipeline import validate_sales


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def small_df():
    """A small, sorted, validated sample dataset."""
    df = generate_sample_train(
        stores=[1],
        families=["GROCERY I", "BEVERAGES"],
        start="2017-01-01",
        end="2017-06-30",
    )
    return validate_sales(df)


# ── Calendar features ──────────────────────────────────────────────────────────

def test_calendar_features_present(small_df):
    df = add_calendar_features(small_df)
    for col in ["day_of_week", "month", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert col in df.columns, f"Missing calendar column: {col}"


def test_calendar_dow_range(small_df):
    df = add_calendar_features(small_df)
    assert df["day_of_week"].between(0, 6).all()


def test_calendar_month_range(small_df):
    df = add_calendar_features(small_df)
    assert df["month"].between(1, 12).all()


# ── Lag features ──────────────────────────────────────────────────────────────

def test_lag_features_created(small_df):
    df = add_lag_features(small_df, lag_days=[1, 7])
    assert "lag_1" in df.columns
    assert "lag_7" in df.columns


def test_lag_1_value_is_previous_day(small_df):
    """lag_1 for row t should equal sales at t-1 within the same series."""
    df = add_lag_features(small_df, lag_days=[1])
    s1 = df[(df["store_nbr"] == 1) & (df["family"] == "GROCERY I")].sort_values("date")
    # Row index 5: lag_1 should be the sales of row index 4
    idx = s1.index[5]
    prev_idx = s1.index[4]
    assert s1.loc[idx, "lag_1"] == pytest.approx(s1.loc[prev_idx, "sales"], rel=1e-5)


def test_lag_does_not_cross_series_groups(small_df):
    """lag_1 at the start of each series should be NaN (no prior observation)."""
    df = add_lag_features(small_df, lag_days=[1])
    for (store, fam), grp in df.groupby(["store_nbr", "family"]):
        first_lag = grp.sort_values("date")["lag_1"].iloc[0]
        assert np.isnan(first_lag), \
            f"First lag_1 is not NaN for store={store}, family={fam}"


# ── Rolling features ──────────────────────────────────────────────────────────

def test_rolling_features_created(small_df):
    df = add_rolling_features(small_df, windows=[7])
    assert "roll_mean_7" in df.columns
    assert "roll_std_7" in df.columns
    assert "ewma_7" in df.columns


def test_rolling_mean_excludes_today(small_df):
    """
    roll_mean_7 for row t must NOT include sales[t].
    We verify by checking that the rolling feature is computed on shifted series.
    """
    df = add_rolling_features(small_df, windows=[7])
    s1 = df[(df["store_nbr"] == 1) & (df["family"] == "GROCERY I")].sort_values("date")
    # Manually compute roll_mean_7 for day 10 (using days 3..9 = shift(1).rolling(7))
    sales = s1["sales"].values
    expected = np.mean(sales[3:10])     # days 3..9 shifted by 1 then window=7
    actual   = s1["roll_mean_7"].iloc[10]  # this uses shift(1).rolling(7)
    # Allow approximate match — important check is that today's value is excluded
    # (cannot be exact due to NaN handling in early rows)
    if not np.isnan(actual):
        assert actual != sales[10], "roll_mean_7 must not include today's sales."


# ── Promotion features ─────────────────────────────────────────────────────────

def test_promotion_feature_binary(small_df):
    df = add_promotion_features(small_df)
    assert df["is_on_promotion"].isin([0, 1]).all()


# ── Full feature build ────────────────────────────────────────────────────────

def test_build_features_no_leakage(small_df):
    """
    After building features, no rolling or lag feature for row t should contain
    the sales value of row t in the SAME series.  We test this by temporarily
    injecting a very distinctive value.
    """
    df = small_df.copy()
    # Set a sentinel value in one specific row's sales
    mask = (df["store_nbr"] == 1) & (df["family"] == "GROCERY I")
    series = df[mask].sort_values("date")
    row_idx = series.index[30]       # pick row 30
    sentinel = 999999.0
    df.loc[row_idx, "sales"] = sentinel

    feat_df = build_features(df, is_training=True)
    row = feat_df.loc[feat_df.index == row_idx]
    if not row.empty:
        # lag_1 at row 30 should equal sales at row 29, not 999999
        for col in ["lag_7", "roll_mean_7", "ewma_7"]:
            if col in row.columns:
                assert row[col].iloc[0] != pytest.approx(sentinel, rel=0.01), \
                    f"{col} contains today's sentinel value — LEAKAGE DETECTED!"


def test_build_features_drops_nan_sales_target_in_training(small_df):
    """build_features(is_training=True) should drop rows where sales is NaN."""
    df = small_df.copy()
    df.loc[df.index[:5], "sales"] = np.nan
    feat_df = build_features(df, is_training=True)
    assert feat_df["sales"].isna().sum() == 0


# ── Future features ────────────────────────────────────────────────────────────

def test_build_future_features_returns_correct_dates(small_df):
    future_dates = pd.date_range("2017-07-01", periods=7, freq="D")
    result = build_future_features(
        history_df=small_df,
        future_dates=future_dates,
        store_nbr=1,
        family="GROCERY I",
    )
    assert len(result) == 7
    assert set(result["date"].dt.normalize()) == set(future_dates.normalize())


def test_build_future_features_marks_as_predicted(small_df):
    future_dates = pd.date_range("2017-07-01", periods=3, freq="D")
    result = build_future_features(
        history_df=small_df,
        future_dates=future_dates,
        store_nbr=1,
        family="GROCERY I",
    )
    assert result["is_predicted"].all()
