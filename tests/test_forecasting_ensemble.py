"""
tests/test_forecasting_ensemble.py — Unit tests for CatBoost and Hybrid Validation-Weighted Ensemble.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasting_ensemble import (
    CandidateModelMetrics,
    EnsembleFitReport,
    HybridQuantileEnsemble,
    QuantileCatBoostEnsemble,
)


@pytest.fixture
def sample_regression_data():
    np.random.seed(42)
    n = 80
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.uniform(0, 10, n),
        "cat1": ["A" if i % 2 == 0 else "B" for i in range(n)],
    })
    y = np.maximum(0.0, 20.0 + 3.0 * X["feat1"] + 0.5 * X["feat2"] + np.random.randn(n))
    return X, y


def test_quantile_catboost_ensemble(sample_regression_data):
    X, y = sample_regression_data
    cat_ensemble = QuantileCatBoostEnsemble(iterations=15, depth=3)
    cat_ensemble.fit(X, y)

    q10, q50, q90 = cat_ensemble.predict_quantiles(X.head(10))

    assert len(q10) == 10
    assert len(q50) == 10
    assert len(q90) == 10
    assert np.all(q10 >= 0.0)
    assert np.all(q10 <= q50)
    assert np.all(q50 <= q90)


def test_hybrid_quantile_ensemble_fit_and_predict(sample_regression_data):
    X, y = sample_regression_data
    train_idx = 50
    X_train, y_train = X.iloc[:train_idx], y[:train_idx]
    X_val, y_val = X.iloc[train_idx:], y[train_idx:]

    # Drop categorical col for LightGBM/XGB test compatibility or leave clean
    X_train_num = X_train[["feat1", "feat2"]].copy()
    X_val_num = X_val[["feat1", "feat2"]].copy()

    ensemble = HybridQuantileEnsemble(mode="weighted_blend", include_catboost=True, include_xgboost=True)
    report = ensemble.fit(X_train_num, y_train, X_val_num, y_val)

    assert isinstance(report, EnsembleFitReport)
    assert report.best_single_model in report.candidate_metrics
    assert sum(report.weights.values()) == pytest.approx(1.0, abs=1e-3)
    assert report.ensemble_wape >= 0.0

    # Test Prediction (weighted blend)
    q10, q50, q90 = ensemble.predict_quantiles(X_val_num)
    assert len(q10) == len(X_val_num)
    assert np.all(q10 <= q50)
    assert np.all(q50 <= q90)

    # Test Best Single Mode
    ensemble.mode = "best_single"
    q10_b, q50_b, q90_b = ensemble.predict_quantiles(X_val_num)
    assert len(q50_b) == len(X_val_num)
    assert np.all(q10_b <= q50_b)
    assert np.all(q50_b <= q90_b)
