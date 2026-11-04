"""
tests/test_benchmarks.py — Unit tests for multi-model benchmarks and cross validation.
"""

import numpy as np
import pandas as pd
import pytest

from src.benchmarks import (
    BaselineRollingMean,
    BaselineSeasonalNaive,
    QuantileLGBMEnsemble,
    calculate_coverage,
    calculate_pinball_loss,
    calculate_wape,
    run_walk_forward_benchmark,
)


def test_metric_calculations():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    # WAPE = (|10-12| + |20-18| + |30-33|) / (10+20+30) = (2 + 2 + 3) / 60 = 7/60
    assert calculate_wape(y_true, y_pred) == pytest.approx(7.0 / 60.0)

    # Pinball loss
    loss = calculate_pinball_loss(y_true, y_pred, alpha=0.5)
    assert loss > 0.0

    # Coverage
    q10 = np.array([9.0, 15.0, 25.0])
    q90 = np.array([15.0, 25.0, 35.0])
    assert calculate_coverage(y_true, q10, q90) == 1.0


def test_baseline_models():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "store_nbr": 1,
        "family": "GROCERY",
        "sales": np.arange(30, dtype=float),
    })

    # Seasonal naive
    smodel = BaselineSeasonalNaive(season_length=7)
    smodel.fit(df)
    test_df = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-20")],
        "store_nbr": [1],
        "family": ["GROCERY"],
    })
    q10, q50, q90 = smodel.predict_quantiles(test_df)
    assert len(q50) == 1
    assert q10[0] <= q50[0] <= q90[0]

    # Rolling mean
    rmodel = BaselineRollingMean(window_size=7)
    rmodel.fit(df)
    q10_r, q50_r, q90_r = rmodel.predict_quantiles(test_df)
    assert len(q50_r) == 1
    assert q10_r[0] <= q50_r[0] <= q90_r[0]


def test_quantile_lgbm_ensemble():
    np.random.seed(42)
    X = pd.DataFrame({
        "feat1": np.random.randn(60),
        "feat2": np.random.randn(60),
    })
    y = np.maximum(0.0, X["feat1"] * 5 + 20 + np.random.randn(60))

    ensemble = QuantileLGBMEnsemble(n_estimators=10, max_depth=3)
    ensemble.fit(X, y)
    q10, q50, q90 = ensemble.predict_quantiles(X.head(5))

    assert len(q10) == 5
    assert np.all(q10 <= q50)
    assert np.all(q50 <= q90)


def test_walk_forward_benchmark():
    dates = pd.date_range("2026-01-01", periods=55, freq="D")
    np.random.seed(42)
    df = pd.DataFrame({
        "date": dates,
        "store_nbr": 1,
        "family": "GROCERY",
        "feat1": np.random.randn(55),
        "sales": np.random.poisson(20, 55).astype(float),
    })

    report = run_walk_forward_benchmark(
        feature_df=df,
        feature_cols=["feat1"],
        target_col="sales",
        date_col="date",
        n_splits=2,
        test_horizon_days=7,
        include_xgboost=False,  # Keep test fast
    )

    assert report.n_folds == 2
    assert len(report.results) >= 2
    assert isinstance(report.summary_df, pd.DataFrame)
    assert len(report.summary_df) >= 2
