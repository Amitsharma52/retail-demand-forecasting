"""
tests/test_evaluation.py — Unit tests for evaluation metrics.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import mae, rmse, wape, pinball_loss, empirical_coverage, compute_all_metrics


# ── MAE ────────────────────────────────────────────────────────────────────────

def test_mae_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == pytest.approx(0.0)


def test_mae_basic():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    assert mae(y_true, y_pred) == pytest.approx(1.0)


def test_mae_ignores_nan():
    y_true = np.array([10.0, np.nan, 30.0])
    y_pred = np.array([11.0, 99.0, 31.0])
    assert mae(y_true, y_pred) == pytest.approx(1.0)


# ── RMSE ───────────────────────────────────────────────────────────────────────

def test_rmse_perfect():
    y = np.array([5.0, 10.0, 15.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_basic():
    y_true = np.array([3.0, 4.0])
    y_pred = np.array([0.0, 0.0])
    # sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5), rel=1e-4)


# ── WAPE ───────────────────────────────────────────────────────────────────────

def test_wape_basic():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    # |10-12|+|20-18| = 4; |10|+|20| = 30; 4/30 ≈ 0.1333
    assert wape(y_true, y_pred) == pytest.approx(4.0 / 30.0, rel=1e-4)


def test_wape_returns_undefined_when_total_zero():
    """WAPE should return the string 'undefined' when total actual demand is 0."""
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    result = wape(y_true, y_pred)
    assert result == "undefined"


# ── Pinball loss ───────────────────────────────────────────────────────────────

def test_pinball_loss_q50_is_mae_halved():
    """Pinball at alpha=0.5 is 0.5 * MAE."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    pb = pinball_loss(y_true, y_pred, alpha=0.5)
    assert pb == pytest.approx(0.5 * mae(y_true, y_pred), rel=1e-5)


def test_pinball_loss_perfect_q10():
    """Perfect Q10 prediction should have zero pinball loss."""
    y_true = np.array([10.0, 20.0, 30.0])
    assert pinball_loss(y_true, y_true, alpha=0.10) == pytest.approx(0.0)


# ── Coverage ──────────────────────────────────────────────────────────────────

def test_coverage_all_inside():
    y_true  = np.array([5.0, 10.0, 15.0])
    q_lower = np.array([0.0,  5.0, 10.0])
    q_upper = np.array([10.0, 15.0, 20.0])
    assert empirical_coverage(y_true, q_lower, q_upper) == pytest.approx(1.0)


def test_coverage_none_inside():
    y_true  = np.array([100.0, 200.0])
    q_lower = np.array([0.0,   0.0])
    q_upper = np.array([10.0, 10.0])
    assert empirical_coverage(y_true, q_lower, q_upper) == pytest.approx(0.0)


def test_coverage_partial():
    y_true  = np.array([5.0,  50.0])
    q_lower = np.array([0.0,  0.0])
    q_upper = np.array([10.0, 10.0])  # 50 is outside
    assert empirical_coverage(y_true, q_lower, q_upper) == pytest.approx(0.5)


# ── Full metrics suite ────────────────────────────────────────────────────────

def test_compute_all_metrics_returns_required_keys():
    val_df = pd.DataFrame({
        "sales"    : [10.0, 20.0, 30.0],
        "pred_q10" : [8.0,  16.0, 24.0],
        "pred_q50" : [10.0, 20.0, 30.0],
        "pred_q90" : [12.0, 24.0, 36.0],
        "baseline" : [9.0,  18.0, 27.0],
    })
    metrics = compute_all_metrics(val_df)
    for key in ["lgbm_mae", "lgbm_rmse", "lgbm_wape", "pinball_q50", "coverage_80"]:
        assert key in metrics, f"Missing metric: {key}"


def test_compute_all_metrics_handles_missing_baseline():
    val_df = pd.DataFrame({
        "sales"    : [10.0, 20.0],
        "pred_q10" : [8.0,  16.0],
        "pred_q50" : [10.0, 20.0],
        "pred_q90" : [12.0, 24.0],
    })
    metrics = compute_all_metrics(val_df)
    assert "lgbm_mae" in metrics
    assert "baseline_mae" not in metrics  # no baseline column provided
