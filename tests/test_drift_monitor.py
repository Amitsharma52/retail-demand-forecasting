"""
tests/test_drift_monitor.py — Unit tests for model drift and stability monitoring.
"""

import numpy as np
import pandas as pd
import pytest

from src.drift_monitor import (
    SystemMonitoringReport,
    audit_dataset_drift,
    calculate_psi,
    calculate_tracking_signal,
)


def test_calculate_psi_identical_distributions():
    np.random.seed(42)
    s1 = np.random.normal(50, 10, 500)
    s2 = np.random.normal(50, 10, 500)

    # Self comparison must be 0
    assert calculate_psi(s1, s1) == pytest.approx(0.0, abs=1e-4)

    # Identical population draws should have very small PSI (< 0.10)
    psi = calculate_psi(s1, s2)
    assert psi < 0.10


def test_calculate_psi_shifted_distribution():
    np.random.seed(42)
    s1 = np.random.normal(50, 5, 200)
    s2 = np.random.normal(90, 5, 200)  # Massive distribution shift

    psi = calculate_psi(s1, s2)
    assert psi >= 0.25


def test_calculate_tracking_signal():
    y_true = np.array([100.0, 120.0, 110.0, 130.0, 140.0])
    # Systematic under-forecasting: model predicts 50 every day
    y_pred_low = np.full(5, 50.0)

    ts_val, status = calculate_tracking_signal(y_true, y_pred_low)
    assert ts_val > 4.0
    assert "BIASED_LOW" in status


def test_audit_dataset_drift():
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "sales": np.random.normal(50, 10, 100),
        "feat1": np.random.normal(0, 1, 100),
    })

    # Curated with significant drift in feat1
    cur_df = pd.DataFrame({
        "sales": np.random.normal(52, 10, 100),
        "feat1": np.random.normal(5, 1, 100),  # shifted
    })

    report = audit_dataset_drift(ref_df, cur_df, feature_cols=["feat1"], target_col="sales")

    assert isinstance(report, SystemMonitoringReport)
    assert "feat1" in report.feature_details
    assert report.feature_details["feat1"].psi_score > 0.10
