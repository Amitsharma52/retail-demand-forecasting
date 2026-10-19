"""
tests/test_inventory.py — Unit tests for the inventory calculator.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inventory import InventoryInputs, compute_inventory, _NORMAL_80_FACTOR


# ── Helper to build a fake forecast DataFrame ──────────────────────────────────

def make_forecast_df(
    n_days: int = 14,
    q50_val: float = 100.0,
    q10_val: float = 70.0,
    q90_val: float = 130.0,
) -> pd.DataFrame:
    """Create a uniform forecast for testing."""
    dates = pd.date_range("2017-08-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "date"    : dates,
        "pred_q10": [q10_val] * n_days,
        "pred_q50": [q50_val] * n_days,
        "pred_q90": [q90_val] * n_days,
    })


# ── Net inventory position ─────────────────────────────────────────────────────

def test_net_inventory_position():
    forecast_df = make_forecast_df()
    inputs = InventoryInputs(
        on_hand=200, on_order=50, backorders=30,
        review_period=7, lead_time=3, service_level=0.95,
    )
    result = compute_inventory(inputs, forecast_df)
    assert result.net_inventory_position == pytest.approx(200 + 50 - 30, rel=1e-5)


# ── Protection horizon ─────────────────────────────────────────────────────────

def test_protection_horizon():
    forecast_df = make_forecast_df()
    inputs = InventoryInputs(
        on_hand=0, on_order=0, backorders=0,
        review_period=7, lead_time=5, service_level=0.95,
    )
    result = compute_inventory(inputs, forecast_df)
    assert result.protection_horizon == 12   # R + L = 7 + 5


# ── Safety stock formula ──────────────────────────────────────────────────────

def test_safety_stock_formula():
    """SS = ceil(z_p * sigma_H); verify the formula matches manual calculation."""
    from scipy.stats import norm

    R, L    = 7, 3
    H       = R + L
    q10, q90 = 70.0, 130.0
    sigma_t = (q90 - q10) / _NORMAL_80_FACTOR
    sigma_H = np.sqrt(H * sigma_t ** 2)
    z_p     = norm.ppf(0.95)
    expected_ss = int(np.ceil(z_p * sigma_H))

    forecast_df = make_forecast_df(q10_val=q10, q90_val=q90)
    inputs = InventoryInputs(
        on_hand=0, on_order=0, backorders=0,
        review_period=R, lead_time=L, service_level=0.95,
    )
    result = compute_inventory(inputs, forecast_df)
    assert result.safety_stock == expected_ss


# ── Order quantity ────────────────────────────────────────────────────────────

def test_order_quantity_non_negative():
    """Order quantity must never be negative."""
    forecast_df = make_forecast_df()
    for on_hand in [0, 100, 5000]:
        inputs = InventoryInputs(
            on_hand=on_hand, on_order=0, backorders=0,
            review_period=7, lead_time=3, service_level=0.95,
        )
        result = compute_inventory(inputs, forecast_df)
        assert result.order_quantity >= 0, \
            f"Negative order for on_hand={on_hand}"


def test_zero_order_when_inventory_covers_demand():
    """If on_hand is very large, no order should be recommended."""
    forecast_df = make_forecast_df(q50_val=10.0, q10_val=5.0, q90_val=15.0)
    inputs = InventoryInputs(
        on_hand=100_000, on_order=0, backorders=0,
        review_period=7, lead_time=3, service_level=0.95,
    )
    result = compute_inventory(inputs, forecast_df)
    assert result.order_quantity == 0


def test_order_quantity_correct_formula():
    """Manually verify Q = ceil(S - I_pos) = ceil(D_H + SS - I_pos)."""
    from scipy.stats import norm
    R, L    = 7, 3
    H       = R + L
    q50     = 100.0
    q10, q90 = 70.0, 130.0
    on_hand = 200

    sigma_t = (q90 - q10) / _NORMAL_80_FACTOR
    sigma_H = np.sqrt(H * sigma_t ** 2)
    D_H     = q50 * H
    z_p     = norm.ppf(0.95)
    SS      = max(0, int(np.ceil(z_p * sigma_H)))
    S       = D_H + SS
    I_pos   = on_hand
    expected_Q = max(0, int(np.ceil(S - I_pos)))

    forecast_df = make_forecast_df(q50_val=q50, q10_val=q10, q90_val=q90)
    inputs = InventoryInputs(
        on_hand=on_hand, on_order=0, backorders=0,
        review_period=R, lead_time=L, service_level=0.95,
    )
    result = compute_inventory(inputs, forecast_df)
    assert result.order_quantity == expected_Q


# ── Service level consistency ─────────────────────────────────────────────────

def test_higher_service_level_gives_higher_safety_stock():
    """Higher service level should require equal or higher safety stock."""
    forecast_df = make_forecast_df()
    inputs_95 = InventoryInputs(
        on_hand=0, on_order=0, backorders=0,
        review_period=7, lead_time=3, service_level=0.95,
    )
    inputs_99 = InventoryInputs(
        on_hand=0, on_order=0, backorders=0,
        review_period=7, lead_time=3, service_level=0.99,
    )
    result_95 = compute_inventory(inputs_95, forecast_df)
    result_99 = compute_inventory(inputs_99, forecast_df)
    assert result_99.safety_stock >= result_95.safety_stock


def test_higher_service_level_gives_higher_or_equal_order():
    """Keeping other inputs constant, 99% service level should order >= 95%."""
    forecast_df = make_forecast_df(q50_val=50.0, q10_val=30.0, q90_val=70.0)
    inputs_90 = InventoryInputs(
        on_hand=100, on_order=0, backorders=0,
        review_period=7, lead_time=3, service_level=0.90,
    )
    inputs_99 = InventoryInputs(
        on_hand=100, on_order=0, backorders=0,
        review_period=7, lead_time=3, service_level=0.99,
    )
    r90 = compute_inventory(inputs_90, forecast_df)
    r99 = compute_inventory(inputs_99, forecast_df)
    assert r99.order_quantity >= r90.order_quantity


# ── Forecast extension ────────────────────────────────────────────────────────

def test_short_forecast_is_extended():
    """If forecast has fewer days than H, the calculator should extend it."""
    forecast_df = make_forecast_df(n_days=3)    # only 3 days
    inputs = InventoryInputs(
        on_hand=0, on_order=0, backorders=0,
        review_period=7, lead_time=5, service_level=0.95,  # H=12
    )
    # Should not raise; should extend forecast by repeating last row
    result = compute_inventory(inputs, forecast_df)
    assert result.protection_horizon == 12
    assert result.median_demand_H > 0
