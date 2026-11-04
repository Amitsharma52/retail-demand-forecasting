"""
tests/test_conformal.py — Unit tests for Conformalized Quantile Regression (CQR).
"""

import numpy as np
import pandas as pd
import pytest

from src.conformal import (
    apply_conformal_calibration,
    calibrate_forecast_dataframe,
    compute_nonconformity_scores,
    evaluate_conformal_calibration,
    fit_conformal_calibrator,
)


def test_nonconformity_scores_inside_and_outside():
    y_true = np.array([10.0, 25.0, 5.0])
    q_lower = np.array([8.0, 10.0, 8.0])
    q_upper = np.array([20.0, 20.0, 20.0])

    scores = compute_nonconformity_scores(y_true, q_lower, q_upper)

    # Point 0: y=10 is inside [8, 20]. max(8-10, 10-20) = max(-2, -10) = -2.0
    assert scores[0] == pytest.approx(-2.0)
    # Point 1: y=25 is above upper [10, 20]. max(10-25, 25-20) = max(-15, 5) = 5.0
    assert scores[1] == pytest.approx(5.0)
    # Point 2: y=5 is below lower [8, 20]. max(8-5, 5-20) = max(3, -15) = 3.0
    assert scores[2] == pytest.approx(3.0)


def test_fit_conformal_calibrator():
    np.random.seed(42)
    n = 100
    y = np.random.normal(50, 10, n)
    # Under-covered raw intervals
    q_lower = np.full(n, 45.0)
    q_upper = np.full(n, 55.0)

    delta = fit_conformal_calibrator(y, q_lower, q_upper, target_coverage=0.80)
    assert isinstance(delta, float)
    assert delta > 0.0  # Because raw intervals were too narrow


def test_apply_conformal_calibration_monotonicity():
    q10 = np.array([10.0, 20.0, 0.0])
    q50 = np.array([15.0, 25.0, 10.0])
    q90 = np.array([22.0, 30.0, 15.0])

    delta = 5.0
    cal_q10, cal_q50, cal_q90 = apply_conformal_calibration(q10, q50, q90, delta)

    # Check non-negativity
    assert np.all(cal_q10 >= 0.0)
    assert np.all(cal_q50 >= 0.0)
    assert np.all(cal_q90 >= 0.0)

    # Check monotonicity q10 <= q50 <= q90
    assert np.all(cal_q10 <= cal_q50)
    assert np.all(cal_q50 <= cal_q90)

    # Check wider interval
    assert np.all((cal_q90 - cal_q10) >= (q90 - q10))


def test_calibrate_forecast_dataframe():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=5),
        "pred_q10": [10.0, 12.0, 14.0, 8.0, 15.0],
        "pred_q50": [15.0, 18.0, 20.0, 14.0, 22.0],
        "pred_q90": [22.0, 25.0, 28.0, 20.0, 30.0],
    })

    calibrated_df = calibrate_forecast_dataframe(df, conformal_delta=3.0)

    assert "pred_q10_calib" in calibrated_df.columns
    assert "pred_q90_calib" in calibrated_df.columns
    assert np.all(calibrated_df["pred_q10_calib"] <= calibrated_df["pred_q50"])
    assert np.all(calibrated_df["pred_q50"] <= calibrated_df["pred_q90_calib"])


def test_evaluate_conformal_calibration():
    np.random.seed(123)
    n = 200
    y_true = np.random.uniform(20, 80, n)
    q10 = y_true - 15.0
    q50 = y_true.copy()
    q90 = y_true + 15.0

    res = evaluate_conformal_calibration(
        y_test=y_true,
        q10_test=q10,
        q50_test=q50,
        q90_test=q90,
        conformal_delta=2.0,
        target_coverage=0.90,
    )
    assert 0.0 <= res.calibrated_coverage <= 1.0
    assert res.n_calibration_samples == n
