"""
src/train_model.py — Model training orchestrator.

Trains:
  1. LightGBM quantile models at Q10, Q50, Q90.
  2. Records the seasonal-naive baseline (no ML training required).

Training uses a chronological split: the first TRAIN_FRACTION of the timeline
is used for training; the remainder is the holdout / validation set.  No random
split is ever used for time-series data.

Saved artefacts
---------------
  models/lgbm_q10.joblib
  models/lgbm_q50.joblib
  models/lgbm_q90.joblib
  models/model_meta.joblib   — stores metadata (feature columns, categories, split date)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.features import build_features

logger = logging.getLogger(__name__)


# ── Chronological split ────────────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = config.TRAIN_FRACTION,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Split a feature DataFrame into train and validation sets by date.

    Returns (train_df, val_df, split_date_str).
    The split is strictly chronological — no shuffling.
    """
    dates       = df[date_col].sort_values().unique()
    split_idx   = int(len(dates) * train_fraction)
    split_date  = dates[split_idx]

    train_df    = df[df[date_col] <  split_date].copy()
    val_df      = df[df[date_col] >= split_date].copy()

    logger.info(
        "Chronological split: train up to %s (%d rows), val from %s (%d rows).",
        pd.Timestamp(split_date).date(), len(train_df),
        pd.Timestamp(split_date).date(), len(val_df),
    )
    return train_df, val_df, str(pd.Timestamp(split_date).date())


# ── LightGBM quantile model ────────────────────────────────────────────────────

def _make_lgbm(alpha: float) -> LGBMRegressor:
    """Create a LightGBM quantile regressor for the given alpha."""
    params = dict(config.LGBM_PARAMS)
    params.update({
        "objective": "quantile",
        "alpha"    : alpha,
        "metric"   : "quantile",
    })
    return LGBMRegressor(**params)


def train_quantile_models(
    feature_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[LGBMRegressor, LGBMRegressor, LGBMRegressor, pd.DataFrame, dict]:
    """
    Train Q10, Q50, Q90 LightGBM models and return evaluation data.

    Parameters
    ----------
    feature_df   : Full feature DataFrame (output of build_features).
    feature_cols : Columns to use as X.  If None, derived from DataFrame columns.

    Returns
    -------
    q10_model, q50_model, q90_model : Fitted LGBMRegressor objects.
    val_df                           : Validation DataFrame with predictions.
    meta                             : Training metadata dictionary.
    """
    # ── Derive feature columns from the DataFrame if not provided ─────────────
    if feature_cols is None:
        # Import here so we get the post-build_features global if available
        import src.features as feat_module
        if feat_module.FEATURE_COLS:
            feature_cols = feat_module.FEATURE_COLS
        else:
            # Fallback: infer from column names that match known feature patterns
            exclude = {
                "date", "sales", "sales_is_missing", "is_holiday",
                "is_predicted", "oil_price", "store_count", "transactions",
                "city", "state", "type", "cluster",
            }
            feature_cols = [c for c in feature_df.columns if c not in exclude]

    if not feature_cols:
        raise ValueError(
            "feature_cols is empty — call build_features() before training."
        )

    # ── Split ─────────────────────────────────────────────────────────────────
    train_df, val_df, split_date = chronological_split(feature_df)

    X_train = train_df[feature_cols]
    y_train = train_df["sales"]
    X_val   = val_df[feature_cols]
    y_val   = val_df["sales"]

    logger.info("Training set: %d rows | Validation set: %d rows", len(X_train), len(X_val))

    # ── No categorical dtype conversion ──────────────────────────────────────
    # store_nbr and family are already integer-encoded by build_features.
    # day_of_week and month are also integers.
    # We pass all features as plain numeric values to avoid dtype mismatches
    # between training and prediction DataFrames.
    cat_cols_present = []  # no categorical dtype columns used

    models = {}
    for alpha, name in [(config.Q_LOWER, "q10"), (config.Q_MEDIAN, "q50"), (config.Q_UPPER, "q90")]:
        logger.info("Training LightGBM %s (alpha=%.2f)…", name, alpha)
        t0  = time.time()
        mdl = _make_lgbm(alpha)
        mdl.fit(X_train, y_train)
        elapsed = time.time() - t0
        logger.info("  → %s trained in %.1f s.", name, elapsed)
        models[name] = mdl

    # ── Predict on validation set ─────────────────────────────────────────────
    q10_pred = models["q10"].predict(X_val)
    q50_pred = models["q50"].predict(X_val)
    q90_pred = models["q90"].predict(X_val)

    q10_pred, q50_pred, q90_pred = _fix_crossing(q10_pred, q50_pred, q90_pred)

    val_df = val_df.copy()
    val_df["pred_q10"] = q10_pred
    val_df["pred_q50"] = q50_pred
    val_df["pred_q90"] = q90_pred
    val_df["baseline"] = val_df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(7)

    import src.features as _feat
    meta = {
        "feature_cols"  : feature_cols,
        "cat_cols"      : cat_cols_present,
        "split_date"    : split_date,
        "train_rows"    : len(train_df),
        "val_rows"      : len(val_df),
        "stores"        : sorted(feature_df["store_nbr"].unique().tolist()),
        "families"      : sorted(feature_df["family"].unique().tolist()),
        "lgbm_params"   : config.LGBM_PARAMS,
        "data_source"   : feature_df.attrs.get("data_source", "unknown"),
        "store_map"     : dict(_feat.STORE_MAP),
        "family_map"    : dict(_feat.FAMILY_MAP),
    }

    return models["q10"], models["q50"], models["q90"], val_df, meta


# ── Quantile crossing correction ──────────────────────────────────────────────

def _fix_crossing(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure that q10 ≤ q50 ≤ q90 for every prediction.

    Strategy: sort the three values for each observation.  This is an
    order-preserving safeguard — NOT a substitute for calibration evaluation.
    All values are also clipped to ≥ 0 (demand cannot be negative).
    """
    stacked = np.stack([q10, q50, q90], axis=1)
    stacked = np.sort(stacked, axis=1)          # monotone ordering
    stacked = np.clip(stacked, 0, None)         # non-negative

    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


# ── Save / load ───────────────────────────────────────────────────────────────

def save_models(
    q10: LGBMRegressor,
    q50: LGBMRegressor,
    q90: LGBMRegressor,
    meta: dict,
) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(q10,  config.MODEL_Q10_PATH)
    joblib.dump(q50,  config.MODEL_Q50_PATH)
    joblib.dump(q90,  config.MODEL_Q90_PATH)
    joblib.dump(meta, config.MODEL_META_PATH)
    logger.info("Models saved to %s", config.MODELS_DIR)


def load_models() -> tuple[LGBMRegressor, LGBMRegressor, LGBMRegressor, dict]:
    """
    Load saved quantile models and metadata.

    Raises FileNotFoundError if any model file is missing.
    """
    for p in [config.MODEL_Q10_PATH, config.MODEL_Q50_PATH,
              config.MODEL_Q90_PATH, config.MODEL_META_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Model file not found: {p}. "
                "Please run the training pipeline first (see README)."
            )

    q10  = joblib.load(config.MODEL_Q10_PATH)
    q50  = joblib.load(config.MODEL_Q50_PATH)
    q90  = joblib.load(config.MODEL_Q90_PATH)
    meta = joblib.load(config.MODEL_META_PATH)

    logger.info("Models loaded from %s", config.MODELS_DIR)
    return q10, q50, q90, meta


def models_exist() -> bool:
    """Return True if all model artefacts are present on disk."""
    return all(
        p.exists()
        for p in [config.MODEL_Q10_PATH, config.MODEL_Q50_PATH,
                  config.MODEL_Q90_PATH, config.MODEL_META_PATH]
    )


# ── Full training pipeline (callable from CLI or dashboard) ──────────────────

def run_training_pipeline(df_processed: pd.DataFrame) -> dict:
    """
    End-to-end training: features → models → save.

    Parameters
    ----------
    df_processed : Output of data_pipeline.load_and_preprocess().

    Returns
    -------
    dict with 'meta', 'val_df', and 'metrics' keys.
    """
    from src.evaluation import compute_all_metrics

    logger.info("Building features…")
    feat_df = build_features(df_processed, is_training=True)
    feat_df.attrs["data_source"] = df_processed.attrs.get("data_source", "unknown")

    logger.info("Training quantile models…")
    q10, q50, q90, val_df, meta = train_quantile_models(feat_df)

    logger.info("Saving models…")
    save_models(q10, q50, q90, meta)

    logger.info("Computing validation metrics…")
    metrics = compute_all_metrics(val_df)
    meta["metrics"] = metrics

    # Re-save meta with metrics included
    joblib.dump(meta, config.MODEL_META_PATH)

    logger.info("Training pipeline complete.")
    return {"meta": meta, "val_df": val_df, "metrics": metrics}
