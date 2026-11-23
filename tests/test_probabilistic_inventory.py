"""
tests/test_probabilistic_inventory.py — Unit tests for probabilistic inventory replenishment.
"""

import numpy as np
import pandas as pd
import pytest

from src.probabilistic_inventory import (
    ProbabilisticInventoryInputs,
    ProbabilisticInventoryResult,
    compute_probabilistic_inventory,
)


@pytest.fixture
def sample_forecast_df():
    dates = pd.date_range("2026-01-01", periods=15, freq="D")
    return pd.DataFrame({
        "date": dates,
        "pred_q10": np.full(15, 15.0),
        "pred_q50": np.full(15, 25.0),
        "pred_q90": np.full(15, 35.0),
    })


def test_probabilistic_inventory_healthy_stock(sample_forecast_df):
    inputs = ProbabilisticInventoryInputs(
        on_hand=300.0,
        on_order=50.0,
        backorders=0.0,
        review_period=7,
        lead_time=3,
        target_service_level=0.95,
    )
    res = compute_probabilistic_inventory(inputs, sample_forecast_df)

    assert isinstance(res, ProbabilisticInventoryResult)
    assert res.protection_horizon_H == 10
    assert res.net_inventory_position == 350.0
    # Plentiful stock -> low stockout risk
    assert res.stockout_probability_pct <= 5.0
    assert res.expected_fill_rate_pct >= 95.0
    assert res.recommended_order_quantity == 0
    assert res.reorder_triggered is False


def test_probabilistic_inventory_low_stock_triggers_reorder(sample_forecast_df):
    inputs = ProbabilisticInventoryInputs(
        on_hand=30.0,
        on_order=0.0,
        backorders=10.0,
        review_period=7,
        lead_time=3,
        target_service_level=0.95,
    )
    res = compute_probabilistic_inventory(inputs, sample_forecast_df)

    assert res.net_inventory_position == 20.0
    # Net position of 20 is far below lead time demand (3 * 25 = 75)
    assert res.reorder_triggered is True
    assert res.recommended_order_quantity > 0
    assert res.stockout_probability_pct > 50.0
    assert "RISK" in res.risk_urgency_tier
