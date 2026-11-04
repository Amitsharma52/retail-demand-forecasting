"""
src/simulation.py — Historical Inventory Policy Simulation & Financial Backtesting.

Allows store owners and managers to simulate day-by-day inventory operations
under different replenishment policies over real historical timelines.

Compares:
1. StockWise Dynamic Quantile Policy (Adaptive Q10/Q50/Q90 + Protection Horizon)
2. Classical Static Min-Max (s, S) Policy
3. Moving Average Heuristic Policy (7-day historical mean)

Computes service levels, stockouts, holding costs, lost margins, and net ROI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


class PolicyType(str, Enum):
    STOCKWISE_DYNAMIC = "StockWise Dynamic Quantile"
    STATIC_MIN_MAX = "Static Min-Max (s, S)"
    MOVING_AVERAGE = "7-Day Moving Average Heuristic"


@dataclass
class CostParameters:
    """Financial parameters for realistic inventory simulation."""
    unit_cost: float = 10.0                # Purchase price per unit from distributor ($)
    selling_price: float = 16.0            # Retail price per unit to consumers ($)
    holding_cost_rate_annual: float = 0.20 # 20% annualized holding/capital cost
    stockout_penalty_per_unit: float = 8.0 # Lost profit margin ($6) + customer goodwill penalty ($2)
    order_fixed_cost: float = 25.0         # Administrative/freight cost per replenishment order placed ($)

    @property
    def holding_cost_per_unit_day(self) -> float:
        return (self.unit_cost * self.holding_cost_rate_annual) / 365.0


@dataclass
class PolicySummaryMetrics:
    """Consolidated KPI report for a simulated policy."""
    policy_name: str
    total_days: int
    total_demand: float
    total_sales_units: float
    total_lost_sales_units: float
    fill_rate_pct: float
    stockout_days_count: int
    stockout_frequency_pct: float
    orders_placed_count: int
    total_units_ordered: float
    average_on_hand_units: float
    total_holding_cost: float
    total_stockout_cost: float
    total_ordering_cost: float
    total_inventory_cost: float
    gross_revenue: float
    gross_profit: float
    net_operating_profit: float


@dataclass
class SimulationRunResult:
    """Detailed output including daily timeline log and summary metrics."""
    policy_name: str
    summary: PolicySummaryMetrics
    daily_log: pd.DataFrame


def simulate_policy(
    daily_demand_series: pd.Series,
    policy_type: PolicyType,
    review_period: int = 7,
    lead_time: int = 3,
    service_level: float = 0.95,
    initial_stock: Optional[float] = None,
    cost_params: Optional[CostParameters] = None,
    q10_predictions: Optional[pd.Series] = None,
    q50_predictions: Optional[pd.Series] = None,
    q90_predictions: Optional[pd.Series] = None,
) -> SimulationRunResult:
    """
    Run an exact historical day-by-day discrete-event inventory simulation.

    Parameters
    ----------
    daily_demand_series : pd.Series indexed chronologically with daily actual demand
    policy_type         : PolicyType enum
    review_period       : Review cycle R in days
    lead_time           : Supplier lead time L in days
    service_level       : Target cycle service level (e.g. 0.95)
    initial_stock       : Starting on-hand stock (default: 3 days of avg demand)
    cost_params         : CostParameters dataclass
    q10/q50/q90_predictions : Quantile forecast series aligned with timeline
    """
    costs = cost_params or CostParameters()
    demands = daily_demand_series.fillna(0.0).clip(lower=0.0).values
    dates = daily_demand_series.index
    n_days = len(demands)

    if n_days == 0:
        raise ValueError("Cannot simulate on empty demand series.")

    mean_demand = float(np.mean(demands))
    std_demand = float(np.std(demands)) if len(demands) > 1 else max(1.0, mean_demand * 0.3)
    z_p = float(norm.ppf(service_level))

    if initial_stock is None:
        initial_stock = mean_demand * (lead_time + 1)

    # Pre-compute static (s, S) parameters if applicable
    H = review_period + lead_time
    static_ss = z_p * std_demand * np.sqrt(H)
    static_s = (mean_demand * lead_time) + static_ss
    static_S = (mean_demand * H) + static_ss

    # Pending orders in transit: queue of (arrival_day_index, quantity)
    in_transit_orders: list[tuple[int, float]] = []

    # Daily tracking state
    current_stock = float(initial_stock)
    records = []

    for t in range(n_days):
        date_t = dates[t]
        demand_t = float(demands[t])

        # 1. Process inbound order arrivals due on or before day t
        arrived_qty = 0.0
        remaining_transit = []
        for arr_day, qty in in_transit_orders:
            if arr_day <= t:
                arrived_qty += qty
            else:
                remaining_transit.append((arr_day, qty))
        in_transit_orders = remaining_transit

        # Starting inventory after delivery
        stock_available = current_stock + arrived_qty

        # 2. Satisfy customer demand
        met_sales = min(stock_available, demand_t)
        lost_sales = max(0.0, demand_t - stock_available)
        ending_stock = max(0.0, stock_available - demand_t)
        is_stockout = stock_available < demand_t

        # 3. Replenishment decision (evaluated on review days: t % review_period == 0)
        order_qty = 0.0
        if t % review_period == 0:
            total_in_transit = sum(qty for _, qty in in_transit_orders)
            net_position = ending_stock + total_in_transit

            if policy_type == PolicyType.STOCKWISE_DYNAMIC:
                # Use dynamic quantile forecasts if available
                if q50_predictions is not None and t < len(q50_predictions):
                    # Sum next H days of available predictions
                    future_slice = q50_predictions.iloc[t : t + H]
                    d_h = float(future_slice.sum()) if len(future_slice) > 0 else mean_demand * H
                    
                    if q90_predictions is not None and q10_predictions is not None:
                        q90_slice = q90_predictions.iloc[t : t + H].values
                        q10_slice = q10_predictions.iloc[t : t + H].values
                        daily_sig = np.maximum(0.0, (q90_slice - q10_slice) / 2.5632)
                        sigma_h = float(np.sqrt(np.sum(daily_sig ** 2)))
                    else:
                        sigma_h = std_demand * np.sqrt(H)
                else:
                    d_h = mean_demand * H
                    sigma_h = std_demand * np.sqrt(H)

                dynamic_ss = max(0.0, np.ceil(z_p * sigma_h))
                target_S = d_h + dynamic_ss
                order_qty = max(0.0, float(np.ceil(target_S - net_position)))

            elif policy_type == PolicyType.STATIC_MIN_MAX:
                if net_position <= static_s:
                    order_qty = max(0.0, float(np.ceil(static_S - net_position)))

            elif policy_type == PolicyType.MOVING_AVERAGE:
                # Naive: order expected demand over lead time based on last 7 days mean
                past_window = demands[max(0, t - 7) : t] if t > 0 else [mean_demand]
                roll_mean = float(np.mean(past_window)) if len(past_window) > 0 else mean_demand
                heuristic_target = roll_mean * (review_period + lead_time)
                order_qty = max(0.0, float(np.ceil(heuristic_target - net_position)))

            if order_qty > 0.0:
                arrival_day = t + lead_time
                in_transit_orders.append((arrival_day, order_qty))

        # 4. Daily accounting costs
        h_cost = ending_stock * costs.holding_cost_per_unit_day
        s_cost = lost_sales * costs.stockout_penalty_per_unit
        o_cost = costs.order_fixed_cost if order_qty > 0 else 0.0
        revenue = met_sales * costs.selling_price
        cogs = met_sales * costs.unit_cost
        gross_profit = revenue - cogs
        net_operating = gross_profit - (h_cost + s_cost + o_cost)

        records.append({
            "date": date_t,
            "starting_stock": current_stock,
            "deliveries_arrived": arrived_qty,
            "available_stock": stock_available,
            "customer_demand": demand_t,
            "met_sales": met_sales,
            "lost_sales": lost_sales,
            "ending_stock": ending_stock,
            "is_stockout": is_stockout,
            "order_placed": order_qty,
            "holding_cost": h_cost,
            "stockout_cost": s_cost,
            "ordering_cost": o_cost,
            "total_inventory_cost": h_cost + s_cost + o_cost,
            "gross_revenue": revenue,
            "gross_profit": gross_profit,
            "net_operating_profit": net_operating,
        })

        # Update on-hand stock for next day
        current_stock = ending_stock

    log_df = pd.DataFrame(records)

    # 5. Aggregate summary calculations
    tot_demand = float(log_df["customer_demand"].sum())
    tot_sales = float(log_df["met_sales"].sum())
    tot_lost = float(log_df["lost_sales"].sum())
    fill_rate = (tot_sales / tot_demand * 100.0) if tot_demand > 0 else 100.0
    stockout_count = int(log_df["is_stockout"].sum())
    stockout_freq = (stockout_count / n_days * 100.0) if n_days > 0 else 0.0
    orders_count = int((log_df["order_placed"] > 0).sum())
    tot_ordered = float(log_df["order_placed"].sum())
    avg_stock = float(log_df["ending_stock"].mean())

    tot_h_cost = float(log_df["holding_cost"].sum())
    tot_s_cost = float(log_df["stockout_cost"].sum())
    tot_o_cost = float(log_df["ordering_cost"].sum())
    tot_inv_cost = tot_h_cost + tot_s_cost + tot_o_cost
    tot_rev = float(log_df["gross_revenue"].sum())
    tot_gp = float(log_df["gross_profit"].sum())
    tot_net = float(log_df["net_operating_profit"].sum())

    summary = PolicySummaryMetrics(
        policy_name=policy_type.value,
        total_days=n_days,
        total_demand=tot_demand,
        total_sales_units=tot_sales,
        total_lost_sales_units=tot_lost,
        fill_rate_pct=round(fill_rate, 2),
        stockout_days_count=stockout_count,
        stockout_frequency_pct=round(stockout_freq, 2),
        orders_placed_count=orders_count,
        total_units_ordered=tot_ordered,
        average_on_hand_units=round(avg_stock, 1),
        total_holding_cost=round(tot_h_cost, 2),
        total_stockout_cost=round(tot_s_cost, 2),
        total_ordering_cost=round(tot_o_cost, 2),
        total_inventory_cost=round(tot_inv_cost, 2),
        gross_revenue=round(tot_rev, 2),
        gross_profit=round(tot_gp, 2),
        net_operating_profit=round(tot_net, 2),
    )

    return SimulationRunResult(
        policy_name=policy_type.value,
        summary=summary,
        daily_log=log_df,
    )


def run_comparative_simulation(
    daily_demand_series: pd.Series,
    review_period: int = 7,
    lead_time: int = 3,
    service_level: float = 0.95,
    initial_stock: Optional[float] = None,
    cost_params: Optional[CostParameters] = None,
    q10_predictions: Optional[pd.Series] = None,
    q50_predictions: Optional[pd.Series] = None,
    q90_predictions: Optional[pd.Series] = None,
) -> dict[str, SimulationRunResult]:
    """
    Run all three inventory policies on identical demand series and compare KPIs.
    """
    results = {}
    for pol in [PolicyType.STOCKWISE_DYNAMIC, PolicyType.STATIC_MIN_MAX, PolicyType.MOVING_AVERAGE]:
        results[pol.value] = simulate_policy(
            daily_demand_series=daily_demand_series,
            policy_type=pol,
            review_period=review_period,
            lead_time=lead_time,
            service_level=service_level,
            initial_stock=initial_stock,
            cost_params=cost_params,
            q10_predictions=q10_predictions,
            q50_predictions=q50_predictions,
            q90_predictions=q90_predictions,
        )
    return results


def format_simulation_comparison_table(
    sim_results: dict[str, SimulationRunResult]
) -> pd.DataFrame:
    """Format comparison metrics into a clean DataFrame for Streamlit and reports."""
    rows = []
    for pol_name, res in sim_results.items():
        s = res.summary
        rows.append({
            "Policy Strategy": s.policy_name,
            "Fill Rate (%)": f"{s.fill_rate_pct:.1f}%",
            "Lost Sales (Units)": f"{s.total_lost_sales_units:,.0f}",
            "Stockout Days": f"{s.stockout_days_count} d ({s.stockout_frequency_pct:.1f}%)",
            "Orders Placed": s.orders_placed_count,
            "Avg On-Hand": f"{s.average_on_hand_units:,.0f}",
            "Holding Cost ($)": f"${s.total_holding_cost:,.2f}",
            "Stockout Cost ($)": f"${s.total_stockout_cost:,.2f}",
            "Total Inv Cost ($)": f"${s.total_inventory_cost:,.2f}",
            "Net Profit ($)": f"${s.net_operating_profit:,.2f}",
        })
    return pd.DataFrame(rows)
