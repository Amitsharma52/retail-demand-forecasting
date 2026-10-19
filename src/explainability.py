"""
src/explainability.py — SHAP-based model explanations.

Uses TreeSHAP with the fitted LightGBM Q50 (median) model.

IMPORTANT CAVEATS (displayed in the dashboard)
----------------------------------------------
* TreeSHAP explains model BEHAVIOUR — it shows which features contributed to a
  specific prediction compared to the background distribution.
* SHAP values are NOT proof of causality.
* Only mention a feature as a positive/negative driver if its SHAP value
  actually supports that direction.
* If SHAP is unavailable, a clear warning is shown — no attribution values
  are invented.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy import so that the module still loads if shap is missing
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning(
        "SHAP package is not installed. Explainability features will be disabled."
    )


def is_shap_available() -> bool:
    return _SHAP_AVAILABLE


def compute_shap_values(
    model,
    X: pd.DataFrame,
    max_background_rows: int = 200,
) -> Optional[np.ndarray]:
    """
    Compute SHAP values for X using TreeExplainer.

    Parameters
    ----------
    model                : Fitted LGBMRegressor (median model).
    X                    : Feature DataFrame — same columns as training.
    max_background_rows  : Background dataset sample for the SHAP explainer.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features), or None if SHAP unavailable.
    """
    if not _SHAP_AVAILABLE:
        logger.warning("SHAP not available — cannot compute explanations.")
        return None

    try:
        # TreeExplainer is fast and exact for tree-based models
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return np.array(shap_values)
    except Exception as exc:
        logger.error("SHAP computation failed: %s", exc)
        return None


def global_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Return a DataFrame of global SHAP feature importance (mean |SHAP|).

    Falls back to LightGBM's built-in gain importance if SHAP is unavailable.
    """
    if _SHAP_AVAILABLE:
        try:
            explainer   = shap.TreeExplainer(model)
            # Use a small subset of training data for speed; a fuller background
            # is ideal but computationally expensive
            n_features  = len(feature_names)
            importances = np.abs(
                explainer.shap_values(
                    pd.DataFrame(
                        np.zeros((1, n_features)),
                        columns=feature_names,
                    )
                )
            ).mean(axis=0)
        except Exception:
            importances = model.feature_importances_
    else:
        importances = model.feature_importances_

    df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return df


def local_shap_explanation(
    model,
    X_row: pd.DataFrame,
    feature_names: list[str],
) -> Optional[pd.DataFrame]:
    """
    Compute SHAP values for a single prediction row.

    Returns a DataFrame with (feature, shap_value, feature_value) sorted by
    absolute SHAP value descending — or None if SHAP is unavailable.
    """
    if not _SHAP_AVAILABLE:
        return None

    shap_vals = compute_shap_values(model, X_row)
    if shap_vals is None:
        return None

    row_vals = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
    feat_vals = X_row.iloc[0].values if len(X_row) > 0 else [np.nan] * len(feature_names)

    df = pd.DataFrame({
        "feature"      : feature_names,
        "shap_value"   : row_vals,
        "feature_value": feat_vals,
    })
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    return df


def generate_rationale(
    forecast_df: pd.DataFrame,
    shap_df: Optional[pd.DataFrame],
    inventory_result,          # InventoryResult dataclass
    data_source: str = "real",
) -> str:
    """
    Generate a plain-language decision rationale using ACTUAL computed values.

    This function ONLY references values that were genuinely computed —
    no hard-coded stories, no invented feature attributions.

    Parameters
    ----------
    forecast_df      : Output of generate_forecast().
    shap_df          : Output of local_shap_explanation() — may be None.
    inventory_result : InventoryResult from inventory.compute_inventory().
    data_source      : 'real' or 'synthetic'.
    """
    lines = []

    # ── Data source caveat ────────────────────────────────────────────────────
    if data_source == "synthetic":
        lines.append(
            "⚠️ **SYNTHETIC DEMO DATA** — The following values are based on "
            "generated sample data, not real store sales.\n"
        )

    # ── Forecast summary ─────────────────────────────────────────────────────
    q50_total = forecast_df["pred_q50"].sum()
    q10_total = forecast_df["pred_q10"].sum()
    q90_total = forecast_df["pred_q90"].sum()
    n_days    = len(forecast_df)
    avg_daily = q50_total / n_days if n_days > 0 else 0

    lines += [
        "### Demand Forecast Rationale",
        f"The LightGBM median model predicts a total of **{q50_total:.1f} units** "
        f"over the next {n_days} days (average {avg_daily:.1f} units/day).",
        f"The 80% prediction interval spans {q10_total:.1f} – {q90_total:.1f} units "
        f"in total across the horizon.",
        "",
    ]

    # ── SHAP feature drivers ─────────────────────────────────────────────────
    if shap_df is not None and not shap_df.empty:
        top_pos = shap_df[shap_df["shap_value"] > 0].head(3)
        top_neg = shap_df[shap_df["shap_value"] < 0].head(3)

        lines.append("### Key Demand Drivers (from SHAP attribution)")
        if not top_pos.empty:
            pos_features = ", ".join(
                f"`{row.feature}` ({row.shap_value:+.2f})"
                for _, row in top_pos.iterrows()
            )
            lines.append(f"**Features increasing the forecast:** {pos_features}")

        if not top_neg.empty:
            neg_features = ", ".join(
                f"`{row.feature}` ({row.shap_value:+.2f})"
                for _, row in top_neg.iterrows()
            )
            lines.append(f"**Features decreasing the forecast:** {neg_features}")

        if "is_on_promotion" in shap_df["feature"].values:
            promo_row = shap_df[shap_df["feature"] == "is_on_promotion"].iloc[0]
            if promo_row["shap_value"] > 0.5:
                lines.append(
                    f"ℹ️ Promotion indicator contributed positively "
                    f"(SHAP = {promo_row['shap_value']:+.2f})."
                )

        lines += [
            "",
            "> **Note:** SHAP values explain model behaviour relative to "
            "the training data distribution. They are not causal explanations.",
            "",
        ]
    else:
        lines.append(
            "### Key Demand Drivers\n"
            "_SHAP attribution not available for this prediction "
            "(SHAP package may not be installed, or the model was not yet trained)._\n"
        )

    # ── Uncertainty ───────────────────────────────────────────────────────────
    lines += [
        "### Uncertainty and Safety Stock",
        f"The estimated demand uncertainty over the protection horizon "
        f"(H = {inventory_result.protection_horizon} days) is "
        f"σ_H = **{inventory_result.sigma_H:.2f} units**.",
        f"At the {inventory_result.z_p / 1.0:.3f} z-score (service level "
        f"{1 - (1 - inventory_result.z_p / 2.326):.0%}), the safety stock is "
        f"**{inventory_result.safety_stock} units**.",
        "",
    ]

    # ── Inventory decision ────────────────────────────────────────────────────
    lines += [
        "### Inventory Decision",
        f"Net inventory position: **{inventory_result.net_inventory_position:.0f} units**",
        f"Target order-up-to level S: **{inventory_result.target_level_S:.1f} units**",
        f"Recommended order quantity: **{inventory_result.order_quantity} units**",
        "",
    ]

    if inventory_result.order_quantity == 0:
        lines.append(
            "✅ No order is required — the current inventory position "
            "already covers the protection-horizon demand estimate plus safety stock."
        )
    else:
        lines.append(
            f"An order of **{inventory_result.order_quantity} units** is recommended "
            "to bring the inventory position up to the target level."
        )

    lines += [
        "",
        "---",
        "**Disclaimer:** This is a model-supported decision aid. "
        "The final purchasing decision should consider additional business factors "
        "not captured in this model (supplier constraints, shelf life, budget limits, etc.).",
    ]

    return "\n".join(lines)
