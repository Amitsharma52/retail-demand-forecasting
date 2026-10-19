"""
src/inventory.py — Deterministic inventory replenishment calculator.

This module is fully independent of Streamlit and is directly testable.

Equations (all documented here and in reports/methodology.md)
--------------------------------------------------------------

Net inventory position:
    I_pos = on_hand + on_order - backorders

Protection horizon:
    H = R + L     (review period + lead time)

Median cumulative demand over H days:
    D_H = Σ q50_t  for t in [1 … H]

Approximate daily standard deviation from quantile interval:
    σ_t = (q90_t - q10_t) / (2 × 1.2816)
    (assumes the interval is approximately a normal 80% PI; this is a documented
    approximation — not a guarantee)

Protection-horizon standard deviation (independence assumption):
    σ_H = sqrt(Σ σ_t²)

Safety stock for cycle service level p:
    z_p = scipy.stats.norm.ppf(p)   (standard normal quantile)
    SS  = max(0, ceil(z_p × σ_H))

Target order-up-to level:
    S = D_H + SS

Recommended order quantity:
    Q = max(0, ceil(S - I_pos))

⚠ These equations approximate uncertainty from MARGINAL quantiles under
  normality and independence assumptions.  They do not guarantee that the
  stated service level will be achieved.  A calibrated multi-horizon model
  is noted as a Phase 2 enhancement.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Factor for converting an 80 % normal PI to a standard deviation
# P(−1.2816 ≤ Z ≤ 1.2816) ≈ 0.80 → σ ≈ (q90 − q10) / (2 × 1.2816)
_NORMAL_80_FACTOR = 2 * 1.2816


@dataclass
class InventoryInputs:
    """All user-facing inputs to the inventory calculator."""
    on_hand       : float      # Current stock on hand (units)
    on_order      : float      # Units already ordered but not yet received
    backorders    : float      # Outstanding backorders (demand not yet fulfilled)
    review_period : int        # R — days between inventory reviews
    lead_time     : int        # L — supplier lead time in days
    service_level : float      # Target cycle service level, e.g. 0.95


@dataclass
class InventoryResult:
    """All computed outputs from the inventory calculator."""
    net_inventory_position : float
    protection_horizon     : int
    median_demand_H        : float
    sigma_H                : float
    z_p                    : float
    safety_stock           : int
    target_level_S         : float
    order_quantity         : int
    explanation            : str
    assumptions            : list[str]


def compute_inventory(
    inputs: InventoryInputs,
    forecast_df: pd.DataFrame,
) -> InventoryResult:
    """
    Compute the inventory replenishment recommendation.

    Parameters
    ----------
    inputs      : InventoryInputs dataclass with user-supplied parameters.
    forecast_df : DataFrame with columns pred_q10, pred_q50, pred_q90.
                  Must contain at least R + L rows.

    Returns
    -------
    InventoryResult with all intermediate values for full traceability.
    """
    # ── 1. Net inventory position ──────────────────────────────────────────────
    I_pos = inputs.on_hand + inputs.on_order - inputs.backorders

    # ── 2. Protection horizon ─────────────────────────────────────────────────
    H = inputs.review_period + inputs.lead_time

    # ── 3. Extract H days of forecasts ────────────────────────────────────────
    n_available = len(forecast_df)
    if n_available < H:
        logger.warning(
            "Forecast has %d days but protection horizon H=%d. "
            "Repeating last day's forecast to fill the gap.",
            n_available, H,
        )
        # Extend by repeating last row
        last_row = forecast_df.iloc[[-1]]
        extras = pd.concat([last_row] * (H - n_available), ignore_index=True)
        forecast_df = pd.concat([forecast_df, extras], ignore_index=True)

    horizon_df = forecast_df.iloc[:H]

    q50_vals = horizon_df["pred_q50"].values
    q10_vals = horizon_df["pred_q10"].values
    q90_vals = horizon_df["pred_q90"].values

    # ── 4. Cumulative median demand ───────────────────────────────────────────
    D_H = float(np.sum(q50_vals))

    # ── 5. Daily σ from quantile interval ────────────────────────────────────
    sigma_t = (q90_vals - q10_vals) / _NORMAL_80_FACTOR
    sigma_t = np.maximum(sigma_t, 0)   # guard against crossing artefacts

    # ── 6. Protection-horizon σ (independence assumption) ─────────────────────
    sigma_H = float(np.sqrt(np.sum(sigma_t ** 2)))

    # ── 7. Safety stock ───────────────────────────────────────────────────────
    z_p = float(norm.ppf(inputs.service_level))
    SS  = max(0, math.ceil(z_p * sigma_H))

    # ── 8. Target and order quantity ──────────────────────────────────────────
    S = D_H + SS
    Q = max(0, math.ceil(S - I_pos))

    # ── 9. Plain-language explanation ─────────────────────────────────────────
    explanation = _build_explanation(inputs, I_pos, H, D_H, sigma_H, z_p, SS, S, Q)

    assumptions = [
        "Daily demand forecast errors are treated as independent across days.",
        "The Q10–Q90 interval is approximated as an 80% normal prediction interval.",
        "σ_t ≈ (Q90 − Q10) / 2.564 per day.",
        "Future promotion activity is assumed zero unless known promotion data is provided.",
        "This is a model-supported recommendation, not an automatically executed purchase order.",
        "Stockout correction is NOT applied — the Favorita dataset lacks on-hand inventory logs.",
    ]

    return InventoryResult(
        net_inventory_position=round(I_pos, 2),
        protection_horizon=H,
        median_demand_H=round(D_H, 2),
        sigma_H=round(sigma_H, 2),
        z_p=round(z_p, 4),
        safety_stock=SS,
        target_level_S=round(S, 2),
        order_quantity=Q,
        explanation=explanation,
        assumptions=assumptions,
    )


def _build_explanation(
    inputs: InventoryInputs,
    I_pos: float,
    H: int,
    D_H: float,
    sigma_H: float,
    z_p: float,
    SS: int,
    S: float,
    Q: int,
) -> str:
    """Generate a plain-language walkthrough of every step."""
    lines = [
        f"**Step 1 — Net Inventory Position**",
        f"  On-hand ({inputs.on_hand:.0f}) + On-order ({inputs.on_order:.0f})"
        f" − Backorders ({inputs.backorders:.0f}) = **{I_pos:.0f} units**",
        "",
        f"**Step 2 — Protection Horizon**",
        f"  Review period R = {inputs.review_period} days"
        f" + Lead time L = {inputs.lead_time} days = **H = {H} days**",
        "",
        f"**Step 3 — Expected Demand over H days**",
        f"  Sum of daily median (Q50) forecasts = **{D_H:.1f} units**",
        "",
        f"**Step 4 — Demand Uncertainty (σ_H)**",
        f"  σ_t = (Q90 − Q10) / 2.564 per day; σ_H = √Σσ_t² = **{sigma_H:.2f} units**",
        "(Assumes daily errors are independent and interval ≈ 80% normal PI)",
        "",
        f"**Step 5 — Safety Stock**",
        f"  Service level {inputs.service_level:.0%} → z = {z_p:.3f}",
        f"  SS = ⌈z × σ_H⌉ = ⌈{z_p:.3f} × {sigma_H:.2f}⌉ = **{SS} units**",
        "",
        f"**Step 6 — Target Order-up-to Level**",
        f"  S = D_H + SS = {D_H:.1f} + {SS} = **{S:.1f} units**",
        "",
        f"**Step 7 — Recommended Order Quantity**",
        f"  Q = max(0, ⌈S − I_pos⌉) = max(0, ⌈{S:.1f} − {I_pos:.0f}⌉) = **{Q} units**",
    ]
    if Q == 0:
        lines.append(
            "\n✅ Current inventory position already meets or exceeds the target level."
            " No order is required at this time."
        )
    return "\n".join(lines)
