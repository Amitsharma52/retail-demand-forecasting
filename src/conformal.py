"""
src/conformal.py — Conformalized Quantile Regression (CQR) for Calibrated Prediction Intervals.

In Phase 1, empirical validation showed ~27% to 60% coverage for the nominal 80% interval
due to non-normal error distributions and volatile promotional demand.

Conformalized Quantile Regression (Romano et al., 2019) calibrates the raw quantiles
on held-out calibration residuals to guarantee finite-sample coverage at exact target levels
(e.g., exactly 80% or 90% coverage).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConformalCalibrationResult:
    """Artifacts and metrics from conformal quantile calibration."""
    nominal_coverage: float
    raw_coverage: float
    calibrated_coverage: float
    conformal_adjustment: float
    raw_mean_interval_width: float
    calibrated_mean_interval_width: float
    n_calibration_samples: int


def compute_nonconformity_scores(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> np.ndarray:
    """
    Compute signed non-conformity scores for interval forecasting:
    s_i = max(q_lower_i - y_i, y_i - q_upper_i)

    - Positive s_i indicates y_i fell OUTSIDE the interval by s_i units.
    - Negative s_i indicates y_i fell INSIDE the interval with margin |s_i|.
    """
    y = np.asarray(y_true).ravel()
    ql = np.asarray(q_lower).ravel()
    qu = np.asarray(q_upper).ravel()
    return np.maximum(ql - y, y - qu)


def fit_conformal_calibrator(
    y_calib: np.ndarray,
    q_lower_calib: np.ndarray,
    q_upper_calib: np.ndarray,
    target_coverage: float = 0.80,
) -> float:
    """
    Compute the conformal adjustment quantile delta from calibration data.

    For nominal coverage 1 - alpha (e.g., 0.80 -> alpha = 0.20):
    k = ceil((n + 1) * (1 - alpha)) / n
    adjustment = quantile(scores, k)
    """
    scores = compute_nonconformity_scores(y_calib, q_lower_calib, q_upper_calib)
    n = len(scores)
    if n == 0:
        return 0.0

    # Finite-sample correction level
    alpha = 1.0 - target_coverage
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    conformal_delta = float(np.quantile(scores, q_level, method="higher"))
    return conformal_delta


def apply_conformal_calibration(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    conformal_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply conformal adjustment delta to raw quantiles with order-preserving safeguards:
    q10_calib = max(0, q10 - delta)
    q90_calib = max(q50, q90 + delta)
    """
    q10_arr = np.asarray(q10).copy()
    q50_arr = np.asarray(q50).copy()
    q90_arr = np.asarray(q90).copy()

    calib_q10 = np.maximum(0.0, q10_arr - conformal_delta)
    calib_q90 = np.maximum(0.0, q90_arr + conformal_delta)

    # Monotonicity check
    calib_q10 = np.minimum(calib_q10, q50_arr)
    calib_q90 = np.maximum(calib_q90, q50_arr)

    return calib_q10, q50_arr, calib_q90


def calibrate_forecast_dataframe(
    forecast_df: pd.DataFrame,
    conformal_delta: float,
) -> pd.DataFrame:
    """
    Return a copy of forecast_df with added columns:
    pred_q10_calib, pred_q90_calib, and calibrated interval width.
    """
    df_out = forecast_df.copy()
    c_q10, c_q50, c_q90 = apply_conformal_calibration(
        df_out["pred_q10"].values,
        df_out["pred_q50"].values,
        df_out["pred_q90"].values,
        conformal_delta=conformal_delta,
    )
    df_out["pred_q10_calib"] = c_q10
    df_out["pred_q90_calib"] = c_q90
    df_out["interval_width_raw"] = df_out["pred_q90"] - df_out["pred_q10"]
    df_out["interval_width_calib"] = df_out["pred_q90_calib"] - df_out["pred_q10_calib"]
    return df_out


def evaluate_conformal_calibration(
    y_test: np.ndarray,
    q10_test: np.ndarray,
    q50_test: np.ndarray,
    q90_test: np.ndarray,
    conformal_delta: float,
    target_coverage: float = 0.80,
) -> ConformalCalibrationResult:
    """Evaluate empirical coverage and interval widths before and after calibration."""
    y = np.asarray(y_test).ravel()
    q10 = np.asarray(q10_test).ravel()
    q50 = np.asarray(q50_test).ravel()
    q90 = np.asarray(q90_test).ravel()

    # Raw metrics
    raw_inside = (y >= q10) & (y <= q90)
    raw_cov = float(np.mean(raw_inside))
    raw_width = float(np.mean(np.maximum(0.0, q90 - q10)))

    # Calibrated metrics
    c_q10, c_q50, c_q90 = apply_conformal_calibration(q10, q50, q90, conformal_delta)
    calib_inside = (y >= c_q10) & (y <= c_q90)
    calib_cov = float(np.mean(calib_inside))
    calib_width = float(np.mean(np.maximum(0.0, c_q90 - c_q10)))

    return ConformalCalibrationResult(
        nominal_coverage=target_coverage,
        raw_coverage=raw_cov,
        calibrated_coverage=calib_cov,
        conformal_adjustment=conformal_delta,
        raw_mean_interval_width=raw_width,
        calibrated_mean_interval_width=calib_width,
        n_calibration_samples=len(y),
    )
