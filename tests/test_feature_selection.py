"""
tests/test_feature_selection.py — Unit tests for automated feature pruning and macro engineering.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_selection import (
    add_macro_and_calendar_features,
    prune_collinear_features,
    prune_low_variance_features,
    run_automated_feature_selection,
    select_top_features_by_importance,
)


def test_prune_low_variance():
    df = pd.DataFrame({
        "var_high": np.random.randn(50),
        "var_zero": np.ones(50),
        "var_low": np.zeros(50),
    })
    pruned, dropped = prune_low_variance_features(df, threshold=1e-4)

    assert "var_high" in pruned.columns
    assert "var_zero" not in pruned.columns
    assert "var_low" not in pruned.columns
    assert "var_zero" in dropped
    assert "var_low" in dropped


def test_prune_collinear():
    x1 = np.random.randn(60)
    x2 = x1 * 2.0  # Perfect correlation r = 1.0
    x3 = np.random.randn(60)  # Independent

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    pruned, dropped = prune_collinear_features(df, threshold=0.95)

    assert len(pruned.columns) == 2
    assert "x3" in pruned.columns
    assert len(dropped) == 1


def test_automated_feature_selection():
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "signal1": np.random.randn(n),
        "signal2": np.random.uniform(0, 5, n),
        "const_feat": np.ones(n),
        "noise1": np.random.randn(n) * 0.01,
    })
    y = X["signal1"] * 10 + X["signal2"] * 5 + np.random.randn(n)

    X_sel, report = run_automated_feature_selection(X, y, top_k=2)

    assert report.selected_feature_count <= 2
    assert "const_feat" in report.dropped_low_variance
    assert "signal1" in X_sel.columns or "signal2" in X_sel.columns


def test_macro_and_calendar_features():
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    df = pd.DataFrame({"date": dates, "sales": np.arange(20)})
    oil = pd.Series(np.linspace(70, 75, 20), index=dates)
    holidays = [pd.Timestamp("2026-01-05")]

    df_enhanced = add_macro_and_calendar_features(df, oil_series=oil, holiday_dates=holidays)

    assert "is_payday" in df_enhanced.columns
    assert "days_to_holiday" in df_enhanced.columns
    assert "oil_momentum_7d" in df_enhanced.columns

    # Payday test: Day 15 should be marked as payday
    day15_mask = df_enhanced["date"] == pd.Timestamp("2026-01-15")
    assert df_enhanced.loc[day15_mask, "is_payday"].iloc[0] == 1
