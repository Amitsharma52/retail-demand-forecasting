"""
src/drift_monitor.py — Advanced Model Monitoring, Data Drift, and Concept Drift Detection.

Monitors machine learning reliability in production:
1. Population Stability Index (PSI): Quantifies feature and target distribution shifts.
   - PSI < 0.10: Stable / No drift
   - 0.10 <= PSI < 0.25: Moderate shift / Monitor closely
   - PSI >= 0.25: Significant drift / Retraining recommended
2. Kolmogorov-Smirnov (KS) Two-Sample Test for continuous feature distribution divergence.
3. Rolling Tracking Signal: Evaluates persistent directional forecasting bias (Tracking Signal > 4 or < -4).
4. Automated Retraining Advisor: Synthesizes drift metrics into actionable operational recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


@dataclass
class FeatureDriftResult:
    """Drift evaluation metrics for a single feature."""
    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    drift_status: str  # "STABLE", "MODERATE_DRIFT", "SIGNIFICANT_DRIFT"


@dataclass
class SystemMonitoringReport:
    """Consolidated model health, drift, and bias diagnostics."""
    overall_status: str  # "HEALTHY", "WARNING", "ACTION_REQUIRED"
    retrain_recommended: bool
    mean_psi_score: float
    features_with_drift: List[str]
    tracking_signal_bias: float
    tracking_signal_status: str
    feature_details: Dict[str, FeatureDriftResult]
    recommendation_summary: str


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Compute Population Stability Index (PSI) between reference and current samples:
        PSI = sum((P_b - Q_b) * ln(P_b / Q_b))
    """
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    if len(ref_clean) < 10 or len(cur_clean) < 10:
        return 0.0

    # Determine bin boundaries using reference quantiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    bin_edges = np.unique(bin_edges)  # Guard against duplicate edges

    if len(bin_edges) < 2:
        return 0.0

    # Bin counts and proportions
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

    p_ref = (ref_counts + epsilon) / (len(ref_clean) + epsilon * len(ref_counts))
    q_cur = (cur_counts + epsilon) / (len(cur_clean) + epsilon * len(cur_counts))

    psi_val = float(np.sum((q_cur - p_ref) * np.log(q_cur / p_ref)))
    return max(0.0, psi_val)


def calculate_tracking_signal(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, str]:
    """
    Compute cumulative forecast tracking signal:
        Tracking Signal = Cumulative Sum of Forecast Errors (CFE) / Mean Absolute Deviation (MAD)
    
    Rule: A tracking signal exceeding +4 (under-forecasting) or -4 (over-forecasting)
    signals non-random systematic model bias.
    """
    errors = y_true - y_pred
    cfe = float(np.sum(errors))
    mad = float(np.mean(np.abs(errors)))

    if mad < 1e-4:
        return 0.0, "UNBIASED"

    ts = cfe / mad
    if ts > 4.0:
        status = "BIASED_LOW (Persistent Under-forecasting)"
    elif ts < -4.0:
        status = "BIASED_HIGH (Persistent Over-forecasting)"
    else:
        status = "IN_CONTROL (Acceptable Random Variation)"

    return round(ts, 2), status


def audit_dataset_drift(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "sales",
) -> SystemMonitoringReport:
    """
    Run full statistical drift audit between baseline training data and production inference window.
    """
    feature_results: Dict[str, FeatureDriftResult] = {}
    psi_scores: List[float] = []
    drifting_features: List[str] = []

    # Check target and feature columns
    cols_to_check = [target_col] + [c for c in feature_cols if c in df_reference.columns and c in df_current.columns]

    for col in cols_to_check:
        ref_vals = pd.to_numeric(df_reference[col], errors="coerce").dropna().to_numpy()
        cur_vals = pd.to_numeric(df_current[col], errors="coerce").dropna().to_numpy()

        if len(ref_vals) < 15 or len(cur_vals) < 15:
            continue

        psi = calculate_psi(ref_vals, cur_vals, num_bins=8)
        ks_res = ks_2samp(ref_vals, cur_vals)

        # Classify drift level
        if psi >= 0.25 or ks_res.pvalue < 0.01:
            drift_status = "SIGNIFICANT_DRIFT"
            drifting_features.append(col)
        elif psi >= 0.10 or ks_res.pvalue < 0.05:
            drift_status = "MODERATE_DRIFT"
        else:
            drift_status = "STABLE"

        psi_scores.append(psi)
        feature_results[col] = FeatureDriftResult(
            feature_name=col,
            psi_score=round(psi, 4),
            ks_statistic=round(float(ks_res.statistic), 4),
            ks_p_value=round(float(ks_res.pvalue), 4),
            drift_status=drift_status,
        )

    mean_psi = float(np.mean(psi_scores)) if psi_scores else 0.0

    # Tracking Signal calculation if actuals exist in current window
    if target_col in df_current.columns and f"pred_{target_col}" in df_current.columns:
        ts_val, ts_status = calculate_tracking_signal(
            df_current[target_col].to_numpy(),
            df_current[f"pred_{target_col}"].to_numpy(),
        )
    else:
        ts_val, ts_status = 0.0, "NO_EVALUATION_RESIDUALS"

    # Overall Decision Logic
    retrain = False
    if mean_psi >= 0.20 or len(drifting_features) >= 3 or abs(ts_val) > 4.5:
        overall_status = "ACTION_REQUIRED"
        retrain = True
        summary = (
            f"Significant statistical drift detected across {len(drifting_features)} feature(s) "
            f"(Mean PSI: {mean_psi:.3f}). Model retraining is recommended to realign predictions."
        )
    elif mean_psi >= 0.08 or len(drifting_features) >= 1:
        overall_status = "WARNING"
        summary = (
            f"Moderate covariate shift detected in {drifting_features}. "
            f"Continue monitoring; retraining optional."
        )
    else:
        overall_status = "HEALTHY"
        summary = "Model distributions and error signals are stable and in control. No retraining needed."

    return SystemMonitoringReport(
        overall_status=overall_status,
        retrain_recommended=retrain,
        mean_psi_score=round(mean_psi, 4),
        features_with_drift=drifting_features,
        tracking_signal_bias=ts_val,
        tracking_signal_status=ts_status,
        feature_details=feature_results,
        recommendation_summary=summary,
    )
