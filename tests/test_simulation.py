"""
tests/test_simulation.py — Unit tests for Inventory Policy Simulation & Backtesting.
"""

import numpy as np
import pandas as pd
import pytest

from src.simulation import (
    CostParameters,
    PolicyType,
    format_simulation_comparison_table,
    run_comparative_simulation,
    simulate_policy,
)


@pytest.fixture
def sample_demand_series():
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    np.random.seed(42)
    demand = np.random.poisson(lam=25, size=60)
    return pd.Series(demand, index=dates)


def test_cost_parameters_holding_cost():
    costs = CostParameters(unit_cost=10.0, holding_cost_rate_annual=0.365)
    # 10.0 * 0.365 / 365 = 0.01 per day
    assert costs.holding_cost_per_unit_day == pytest.approx(0.01)


def test_simulate_policy_static_min_max(sample_demand_series):
    costs = CostParameters()
    result = simulate_policy(
        daily_demand_series=sample_demand_series,
        policy_type=PolicyType.STATIC_MIN_MAX,
        review_period=7,
        lead_time=3,
        service_level=0.95,
        initial_stock=100.0,
        cost_params=costs,
    )

    summary = result.summary
    assert summary.policy_name == PolicyType.STATIC_MIN_MAX.value
    assert summary.total_days == 60
    assert summary.total_sales_units > 0
    assert 0.0 <= summary.fill_rate_pct <= 100.0
    assert summary.orders_placed_count >= 1
    assert len(result.daily_log) == 60


def test_simulate_policy_moving_average(sample_demand_series):
    result = simulate_policy(
        daily_demand_series=sample_demand_series,
        policy_type=PolicyType.MOVING_AVERAGE,
        review_period=5,
        lead_time=2,
    )
    assert result.summary.fill_rate_pct >= 0.0
    assert result.summary.gross_revenue > 0


def test_run_comparative_simulation(sample_demand_series):
    results = run_comparative_simulation(
        daily_demand_series=sample_demand_series,
        review_period=7,
        lead_time=3,
        service_level=0.95,
    )

    assert len(results) == 3
    assert PolicyType.STOCKWISE_DYNAMIC in results
    assert PolicyType.STATIC_MIN_MAX in results
    assert PolicyType.MOVING_AVERAGE in results

    table_df = format_simulation_comparison_table(results)
    assert isinstance(table_df, pd.DataFrame)
    assert len(table_df) == 3
    assert "Fill Rate (%)" in table_df.columns
    assert "Net Profit ($)" in table_df.columns
