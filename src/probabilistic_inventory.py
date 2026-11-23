"""
src/probabilistic_inventory.py — Advanced Probabilistic Inventory Optimization & Tail-Risk Engine.

Extends the deterministic periodic-review (R, S) framework with probabilistic risk quantification:
1. Dynamic Safety Stock sizing: SS = ceil(z_p * sigma_H)
2. Dynamic Reorder Point: s = D_L + SS_L (triggers order when I_pos <= s)
3. Target Order-Up-To Level: S = D_H + SS_H
4. Optimal Order Quantity: Q = max(0, ceil(S - I_pos))
5. Probabilistic Tail Stockout Risk: P(Demand_H > I_pos) via exact tail integrals
6. Expected Excess / Overstock Volume: E[max(0, I_pos - Demand_H)]
7. Expected Unit Fill Rate & Cycle Service Level
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

_NORMAL_80_FACTOR = 2 * 1.2816  # 2.5632


@dataclass
class ProbabilisticInventoryInputs:
    """Planning inputs for probabilistic inventory optimization."""
    on_hand: float
    on_order: float = 0.0
    backorders: float = 0.0
    review_period: int = 7
    lead_time: int = 3
    target_service_level: float = 0.95
    unit_cost: float = 10.0
    selling_price: float = 16.0
    holding_cost_rate_annual: float = 0.20


@dataclass
class ProbabilisticInventoryResult:
    """Full optimization metrics including tail risk and expected excess."""
    net_inventory_position: float
    review_period_R: int
    lead_time_L: int
    protection_horizon_H: int

    # Demand projections
    expected_demand_lead_time_DL: float
    expected_demand_horizon_DH: float
    uncertainty_lead_time_sigmaL: float
    uncertainty_horizon_sigmaH: float

    # Thresholds & Recommendations
    safety_stock: int
    reorder_point_s: float
    target_level_S: float
    recommended_order_quantity: int
    reorder_triggered: bool

    # Risk & Service Level Metrics
    stockout_probability_pct: float
    cycle_service_level_pct: float
    expected_shortage_units: float
    expected_fill_rate_pct: float
    expected_overstock_units: float
    expected_holding_cost_period: float
    expected_stockout_cost_period: float

    # Qualitative explanation
    risk_urgency_tier: str
    detailed_explanation: str


def compute_probabilistic_inventory(
    inputs: ProbabilisticInventoryInputs,
    forecast_df: pd.DataFrame,
) -> ProbabilisticInventoryResult:
    """
    Compute probabilistic inventory replenishment recommendations and tail risks.

    Parameters
    ----------
    inputs : ProbabilisticInventoryInputs
    forecast_df : pd.DataFrame containing pred_q10, pred_q50, pred_q90 for at least R+L days.
    """
    R = max(1, int(inputs.review_period))
    L = max(1, int(inputs.lead_time))
    H = R + L

    # Extend forecast if needed
    if len(forecast_df) < H:
        last_row = forecast_df.iloc[-1:]
        extra_rows = [last_row] * (H - len(forecast_df))
        forecast_df = pd.concat([forecast_df] + extra_rows, ignore_index=True)

    fc_H = forecast_df.iloc[:H].copy()
    fc_L = forecast_df.iloc[:L].copy()

    # Median demand projections
    D_H = float(fc_H["pred_q50"].sum())
    D_L = float(fc_L["pred_q50"].sum())

    # Daily standard deviation approximations
    # sigma_t = (Q90 - Q10) / 2.564
    sigma_t_H = np.maximum(0.1, (fc_H["pred_q90"].values - fc_H["pred_q10"].values) / _NORMAL_80_FACTOR)
    sigma_t_L = np.maximum(0.1, (fc_L["pred_q90"].values - fc_L["pred_q10"].values) / _NORMAL_80_FACTOR)

    sigma_H = float(np.sqrt(np.sum(sigma_t_H ** 2)))
    sigma_L = float(np.sqrt(np.sum(sigma_t_L ** 2)))

    # Net inventory position: I_pos = OnHand + OnOrder - Backorders
    I_pos = float(inputs.on_hand + inputs.on_order - inputs.backorders)

    # Normal quantile z_p for target service level
    p = np.clip(inputs.target_service_level, 0.50, 0.999)
    z_p = float(norm.ppf(p))

    # Safety stock for horizon and lead time
    SS_H = max(0, int(math.ceil(z_p * sigma_H)))
    SS_L = max(0, int(math.ceil(z_p * sigma_L)))

    # Dynamic Reorder Point s and Target Level S
    s = D_L + SS_L
    S = D_H + SS_H

    # Trigger logic: Reorder if inventory position is at or below reorder point s
    reorder_triggered = bool(I_pos <= s)
    order_qty = max(0, int(math.ceil(S - I_pos)))

    # Tail Stockout Probability: P(Demand_H > I_pos)
    # Using normal approximation N(D_H, sigma_H^2)
    if sigma_H > 0:
        z_current = (I_pos - D_H) / sigma_H
        prob_stockout = float(1.0 - norm.cdf(z_current))
        prob_stockout = float(np.clip(prob_stockout, 0.0, 1.0))

        # Expected Shortage Units (Standard normal loss function L(z))
        # E[max(0, Demand - I_pos)] = sigma_H * [phi(z) - z * (1 - Phi(z))]
        pdf_z = float(norm.pdf(z_current))
        cdf_z = float(norm.cdf(z_current))
        expected_shortage = max(0.0, float(sigma_H * (pdf_z - z_current * (1.0 - cdf_z))))

        # Expected Overstock / Excess Units: E[max(0, I_pos - Demand)]
        # E[max(0, I_pos - Demand)] = (I_pos - D_H) * Phi(z) + sigma_H * phi(z)
        expected_overstock = max(0.0, float((I_pos - D_H) * cdf_z + sigma_H * pdf_z))
    else:
        prob_stockout = 1.0 if I_pos < D_H else 0.0
        expected_shortage = max(0.0, D_H - I_pos)
        expected_overstock = max(0.0, I_pos - D_H)

    # Unit Fill Rate = 1 - (Expected Shortage / Total Demand)
    if D_H > 0:
        expected_fill_rate = max(0.0, float(1.0 - (expected_shortage / D_H)))
    else:
        expected_fill_rate = 1.0

    cycle_service_level = max(0.0, float(1.0 - prob_stockout))

    # Financial Cost Expectations
    holding_cost_unit_period = (inputs.unit_cost * inputs.holding_cost_rate_annual) * (H / 365.0)
    expected_holding_cost = expected_overstock * holding_cost_unit_period

    lost_margin_per_unit = max(1.0, inputs.selling_price - inputs.unit_cost)
    expected_stockout_cost = expected_shortage * lost_margin_per_unit

    # Urgency Classification
    if prob_stockout >= 0.35 or I_pos < D_L:
        urgency = "HIGH RISK — Stockout Highly Likely Before Next Delivery"
    elif prob_stockout >= 0.15 or I_pos <= s:
        urgency = "ELEVATED RISK — Reorder Point Breached, Place Order Now"
    elif expected_overstock > (2.5 * D_H):
        urgency = "OVERSTOCK WARNING — Substantial Excess Working Capital Tied Up"
    else:
        urgency = "BALANCED — Healthy Buffer and Minimal Stockout Exposure"

    explanation = (
        f"Net position is {I_pos:.0f} units against a dynamic reorder point s = {s:.0f}. "
        f"Over the {H}-day protection horizon, expected demand is {D_H:.0f} units with standard deviation {sigma_H:.1f}. "
        f"To guarantee an availability target of {int(p*100)}%, target stock level is S = {S:.0f} (including {SS_H} units safety buffer). "
        f"Stockout probability without reordering is {prob_stockout*100:.1f}%. Recommended purchase order is {order_qty} units."
    )

    return ProbabilisticInventoryResult(
        net_inventory_position=I_pos,
        review_period_R=R,
        lead_time_L=L,
        protection_horizon_H=H,
        expected_demand_lead_time_DL=round(D_L, 2),
        expected_demand_horizon_DH=round(D_H, 2),
        uncertainty_lead_time_sigmaL=round(sigma_L, 2),
        uncertainty_horizon_sigmaH=round(sigma_H, 2),
        safety_stock=SS_H,
        reorder_point_s=round(s, 1),
        target_level_S=round(S, 1),
        recommended_order_quantity=order_qty,
        reorder_triggered=reorder_triggered,
        stockout_probability_pct=round(prob_stockout * 100.0, 1),
        cycle_service_level_pct=round(cycle_service_level * 100.0, 1),
        expected_shortage_units=round(expected_shortage, 1),
        expected_fill_rate_pct=round(expected_fill_rate * 100.0, 1),
        expected_overstock_units=round(expected_overstock, 1),
        expected_holding_cost_period=round(expected_holding_cost, 2),
        expected_stockout_cost_period=round(expected_stockout_cost, 2),
        risk_urgency_tier=urgency,
        detailed_explanation=explanation,
    )
