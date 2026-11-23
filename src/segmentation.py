"""
src/segmentation.py — ABC-XYZ Product Portfolio Segmentation & Strategic Inventory Profiler.

Implements the 9-box ABC-XYZ classification matrix:
- ABC Classification (Pareto Revenue/Volume Velocity):
    - Class A: Top 80% cumulative volume / revenue (Critical high-velocity core)
    - Class B: Next 15% cumulative volume / revenue (Moderate velocity)
    - Class C: Bottom 5% cumulative volume / revenue (Long-tail low volume)

- XYZ Classification (Demand Variability & Predictability via Coefficient of Variation):
    - Class X: Stable, predictable demand (CV = sigma / mu <= 0.50)
    - Class Y: Moderately variable, seasonal demand (0.50 < CV <= 1.00)
    - Class Z: Highly erratic, intermittent demand (CV > 1.00)

Provides tailored inventory & forecasting policy recommendations for each of the 9 cells
(AX, AY, AZ, BX, BY, BZ, CX, CY, CZ).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ABCClass(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class XYZClass(str, Enum):
    X = "X"
    Y = "Y"
    Z = "Z"


@dataclass
class SegmentPolicy:
    """Tailored replenishment and forecasting parameters for a segment."""
    segment: str
    target_service_level: float
    recommended_review_period: int
    forecasting_strategy: str
    inventory_strategy: str
    priority_level: str


SEGMENT_POLICIES: Dict[str, SegmentPolicy] = {
    "AX": SegmentPolicy(
        segment="AX",
        target_service_level=0.98,
        recommended_review_period=3,
        forecasting_strategy="Advanced Multi-Model Quantile Ensemble (LightGBM/XGB/CatBoost).",
        inventory_strategy="High-priority Just-In-Time (JIT) or tight review cycles. Low safety stock buffer needed due to high stability.",
        priority_level="CRITICAL / HIGH PRIORITY",
    ),
    "AY": SegmentPolicy(
        segment="AY",
        target_service_level=0.95,
        recommended_review_period=5,
        forecasting_strategy="Quantile Ensemble with Calendar/Promotion Seasonal Tracking.",
        inventory_strategy="Frequent reviews with dynamic safety buffers to protect against seasonal swing.",
        priority_level="HIGH PRIORITY",
    ),
    "AZ": SegmentPolicy(
        segment="AZ",
        target_service_level=0.92,
        recommended_review_period=7,
        forecasting_strategy="Conformalized Quantile Regression with Probabilistic Scenario Planning.",
        inventory_strategy="High-value but erratic. Strict order approval with buffered safety stocks or supplier consignment.",
        priority_level="HIGH RISK / CLOSE MONITORING",
    ),
    "BX": SegmentPolicy(
        segment="BX",
        target_service_level=0.95,
        recommended_review_period=7,
        forecasting_strategy="Standard LightGBM / XGBoost Quantile Regression.",
        inventory_strategy="Standard periodic review (R, S). Automated reorders with moderate safety buffer.",
        priority_level="MEDIUM PRIORITY",
    ),
    "BY": SegmentPolicy(
        segment="BY",
        target_service_level=0.90,
        recommended_review_period=7,
        forecasting_strategy="LightGBM with Promotion and Lag-7 Seasonality.",
        inventory_strategy="Periodic review with buffer sizing proportional to observed demand volatility.",
        priority_level="MEDIUM PRIORITY",
    ),
    "BZ": SegmentPolicy(
        segment="BZ",
        target_service_level=0.88,
        recommended_review_period=14,
        forecasting_strategy="Seasonal Naive + Upper Quantile Bounds.",
        inventory_strategy="Longer review cycle with modest safety stock to avoid obsolescence and holding costs.",
        priority_level="MODERATE RISK",
    ),
    "CX": SegmentPolicy(
        segment="CX",
        target_service_level=0.92,
        recommended_review_period=14,
        forecasting_strategy="Simple Rolling Average or Seasonal Naive Baseline.",
        inventory_strategy="Bulk reordering (large case packs) to exploit distributor discounts. Two-bin min-max system.",
        priority_level="LOW EFFORT / AUTOMATED",
    ),
    "CY": SegmentPolicy(
        segment="CY",
        target_service_level=0.88,
        recommended_review_period=14,
        forecasting_strategy="Rolling Mean / Trend Baseline.",
        inventory_strategy="Standard Min-Max reorder. Low management overhead; order only when shelf reaches minimum.",
        priority_level="LOW PRIORITY",
    ),
    "CZ": SegmentPolicy(
        segment="CZ",
        target_service_level=0.85,
        recommended_review_period=21,
        forecasting_strategy="Intermittent Croston / Quantile Zero-Inflated Model.",
        inventory_strategy="Order-on-demand or make-to-order where feasible. Minimize stock on hand to avoid dead stock.",
        priority_level="LONG TAIL / ON DEMAND",
    ),
}


@dataclass
class ProductSegmentationItem:
    """Individual item segmentation result and metrics."""
    store_nbr: int
    family: str
    total_sales_units: float
    total_valuation: float
    volume_share_pct: float
    cumulative_share_pct: float
    abc_class: str
    mean_daily_sales: float
    std_daily_sales: float
    coefficient_of_variation: float
    xyz_class: str
    segment: str
    policy: SegmentPolicy


def classify_abc_xyz(
    df: pd.DataFrame,
    unit_price_map: Optional[Dict[str, float]] = None,
    abc_thresholds: Tuple[float, float] = (0.80, 0.95),
    xyz_thresholds: Tuple[float, float] = (0.50, 1.00),
    store_nbr: Optional[int] = None,
) -> pd.DataFrame:
    """
    Perform ABC-XYZ product portfolio classification on a sales DataFrame.

    Parameters
    ----------
    df : pd.DataFrame with columns ['store_nbr', 'family', 'sales', 'date']
    unit_price_map : Optional dictionary mapping family -> price in dollars
    abc_thresholds : Cumulative share cutoffs for (A, B) — remainder is C
    xyz_thresholds : CV cutoffs for (X, Y) — remainder is Z
    store_nbr : Optional store filter

    Returns
    -------
    pd.DataFrame containing full ABC-XYZ classification for each (store, family)
    """
    df_work = df[df["store_nbr"] == store_nbr].copy() if store_nbr is not None else df.copy()

    # Calculate Volume & Valuation per group
    stats = (
        df_work.groupby(["store_nbr", "family"])["sales"]
        .agg(
            total_units="sum",
            mean_units="mean",
            std_units="std",
            days_count="count",
        )
        .reset_index()
    )

    stats["std_units"] = stats["std_units"].fillna(0.0)

    # Unit price / valuation weighting
    if unit_price_map:
        stats["unit_price"] = stats["family"].map(lambda f: unit_price_map.get(f, 10.0))
        stats["valuation"] = stats["total_units"] * stats["unit_price"]
    else:
        stats["valuation"] = stats["total_units"]  # Unit-volume Pareto

    # 1. ABC Classification (Pareto on Valuation/Volume)
    stats = stats.sort_values(by="valuation", ascending=False).reset_index(drop=True)
    total_val = stats["valuation"].sum()

    if total_val > 0:
        stats["share"] = stats["valuation"] / total_val
        stats["cum_share"] = stats["share"].cumsum()
    else:
        stats["share"] = 0.0
        stats["cum_share"] = 1.0

    a_cutoff, b_cutoff = abc_thresholds

    def assign_abc(cum_share: float) -> str:
        if cum_share <= a_cutoff:
            return "A"
        elif cum_share <= b_cutoff:
            return "B"
        return "C"

    # Edge case: Ensure at least one product in A if only few products exist
    stats["abc_class"] = stats["cum_share"].apply(assign_abc)
    if not (stats["abc_class"] == "A").any() and len(stats) > 0:
        stats.loc[0, "abc_class"] = "A"

    # 2. XYZ Classification (Coefficient of Variation: CV = std / mean)
    x_cutoff, y_cutoff = xyz_thresholds

    def compute_cv(row) -> float:
        mu = row["mean_units"]
        sig = row["std_units"]
        if mu <= 1e-4:
            return 2.0  # Zero/negligible mean -> high intermittency
        return float(sig / mu)

    stats["cv"] = stats.apply(compute_cv, axis=1)

    def assign_xyz(cv: float) -> str:
        if cv <= x_cutoff:
            return "X"
        elif cv <= y_cutoff:
            return "Y"
        return "Z"

    stats["xyz_class"] = stats["cv"].apply(assign_xyz)

    # Combine into 9-box segment
    stats["segment"] = stats["abc_class"] + stats["xyz_class"]

    # Append policy metadata
    stats["target_service_level"] = stats["segment"].map(lambda s: SEGMENT_POLICIES[s].target_service_level)
    stats["recommended_review_period"] = stats["segment"].map(lambda s: SEGMENT_POLICIES[s].recommended_review_period)
    stats["priority_level"] = stats["segment"].map(lambda s: SEGMENT_POLICIES[s].priority_level)
    stats["forecasting_strategy"] = stats["segment"].map(lambda s: SEGMENT_POLICIES[s].forecasting_strategy)
    stats["inventory_strategy"] = stats["segment"].map(lambda s: SEGMENT_POLICIES[s].inventory_strategy)

    return stats


def generate_segmentation_matrix_summary(segmented_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the segmented DataFrame into a standard 3x3 ABC-XYZ grid
    showing item counts per cell.
    """
    grid = (
        pd.crosstab(
            segmented_df["abc_class"],
            segmented_df["xyz_class"],
            margins=True,
            margins_name="Total",
        )
        .reindex(index=["A", "B", "C", "Total"], columns=["X", "Y", "Z", "Total"])
        .fillna(0)
        .astype(int)
    )
    return grid
