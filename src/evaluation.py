"""
src/evaluation.py — Time-series model evaluation metrics.

IMPORTANT: Never use a random train-test split for time series.
All evaluation functions expect predictions that were made ONLY using
information available before the prediction date.

Metrics
-------
MAE   — Mean Absolute Error
RMSE  — Root Mean Squared Error
WAPE  — Weighted Absolute Percentage Error (safe against zero demand)
Pinball loss (quantile loss) for Q10, Q50, Q90
Coverage — empirical fraction of actuals inside the [Q10, Q90] interval
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Point forecast metrics ─────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float | str:
    """
    Weighted Absolute Percentage Error.

    WAPE = sum(|actual - predicted|) / sum(|actual|)

    Returns "undefined" (as a string) when total actual demand is zero to avoid
    a misleading 0 % or division-by-zero error.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    total_actual = np.sum(np.abs(y_true[mask]))
    if total_actual == 0:
        logger.warning("WAPE is undefined: total actual demand is zero.")
        return "undefined"
    return float(np.sum(np.abs(y_true[mask] - y_pred[mask])) / total_actual)


# ── Quantile / interval metrics ────────────────────────────────────────────────

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Pinball (quantile) loss for a single quantile level.

    L_alpha(y, q) = alpha * max(y - q, 0) + (1 - alpha) * max(q - y, 0)
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    err  = y_true[mask] - y_pred[mask]
    return float(np.mean(np.where(err >= 0, alpha * err, (alpha - 1) * err)))


def empirical_coverage(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> float:
    """
    Fraction of actual values that fall within the [q_lower, q_upper] interval.

    NOTE: The models target a nominal 80% interval (Q10 to Q90).
    Whether this is actually achieved depends on calibration — do not
    assert 80% coverage without computing this metric on held-out data.
    """
    y_true  = np.asarray(y_true,  dtype=float)
    q_lower = np.asarray(q_lower, dtype=float)
    q_upper = np.asarray(q_upper, dtype=float)
    mask    = ~np.isnan(y_true) & ~np.isnan(q_lower) & ~np.isnan(q_upper)
    inside  = (y_true[mask] >= q_lower[mask]) & (y_true[mask] <= q_upper[mask])
    return float(np.mean(inside))


def interval_width(q_lower: np.ndarray, q_upper: np.ndarray) -> float:
    """Mean width of the prediction interval."""
    return float(np.nanmean(np.asarray(q_upper) - np.asarray(q_lower)))


# ── All-in-one evaluator ───────────────────────────────────────────────────────

def compute_all_metrics(val_df: pd.DataFrame) -> dict:
    """
    Compute the full metric suite from a validation DataFrame.

    Expected columns: sales, pred_q10, pred_q50, pred_q90, baseline (optional).

    Returns
    -------
    dict with keys for each metric.  All values are floats or the string
    "undefined" for WAPE when total actual demand is zero.
    """
    required = ["sales", "pred_q50"]
    for col in required:
        if col not in val_df.columns:
            raise ValueError(f"Missing required column in val_df: '{col}'")

    y_true   = val_df["sales"].values
    q50_pred = val_df["pred_q50"].values
    q10_pred = val_df["pred_q10"].values if "pred_q10" in val_df.columns else None
    q90_pred = val_df["pred_q90"].values if "pred_q90" in val_df.columns else None
    baseline = val_df["baseline"].values if "baseline" in val_df.columns else None

    metrics: dict = {
        "lgbm_mae"  : mae(y_true, q50_pred),
        "lgbm_rmse" : rmse(y_true, q50_pred),
        "lgbm_wape" : wape(y_true, q50_pred),
    }

    if q10_pred is not None and q90_pred is not None:
        metrics["pinball_q10"] = pinball_loss(y_true, q10_pred, alpha=0.10)
        metrics["pinball_q50"] = pinball_loss(y_true, q50_pred, alpha=0.50)
        metrics["pinball_q90"] = pinball_loss(y_true, q90_pred, alpha=0.90)
        metrics["coverage_80"] = empirical_coverage(y_true, q10_pred, q90_pred)
        metrics["interval_width"] = interval_width(q10_pred, q90_pred)

    if baseline is not None:
        baseline_mask = ~np.isnan(baseline)
        if baseline_mask.any():
            metrics["baseline_mae"]  = mae(y_true[baseline_mask], baseline[baseline_mask])
            metrics["baseline_rmse"] = rmse(y_true[baseline_mask], baseline[baseline_mask])
            metrics["baseline_wape"] = wape(y_true[baseline_mask], baseline[baseline_mask])

    return metrics


def format_metrics_table(metrics: dict) -> pd.DataFrame:
    """Return a tidy DataFrame suitable for Streamlit display."""
    rows = []
    label_map = {
        "lgbm_mae"     : ("LightGBM", "MAE"),
        "lgbm_rmse"    : ("LightGBM", "RMSE"),
        "lgbm_wape"    : ("LightGBM", "WAPE"),
        "baseline_mae" : ("Seasonal Naive", "MAE"),
        "baseline_rmse": ("Seasonal Naive", "RMSE"),
        "baseline_wape": ("Seasonal Naive", "WAPE"),
        "pinball_q10"  : ("LightGBM", "Pinball Q10"),
        "pinball_q50"  : ("LightGBM", "Pinball Q50"),
        "pinball_q90"  : ("LightGBM", "Pinball Q90"),
        "coverage_80"  : ("LightGBM", "80% PI Coverage"),
        "interval_width":("LightGBM", "Mean PI Width"),
    }
    for key, value in metrics.items():
        if key in label_map:
            model, metric = label_map[key]
            fmt_val = f"{value:.4f}" if isinstance(value, float) else str(value)
            rows.append({"Model": model, "Metric": metric, "Value": fmt_val})
    return pd.DataFrame(rows)
