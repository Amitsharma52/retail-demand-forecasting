"""
train_real_data.py — Complete training pipeline on the real Favorita dataset.

Trains:
  1. Seasonal Naive baseline
  2. LightGBM (Q10, Q50, Q90)
  3. XGBoost (Q10, Q50, Q90)
  4. CatBoost (Q10, Q50, Q90)
  5. Hybrid Validation-Weighted Ensemble

Data Scope:
  - 2.5 million item-level rows → aggregated to store-family daily totals
  - Covers all 54 stores, top 20 product families, most recent 24 months
  - Fully chronological train/validation split (80/20)

Usage:
  python train_real_data.py

Outputs:
  models/lgbm_q10.joblib, lgbm_q50.joblib, lgbm_q90.joblib
  models/xgb_q10.joblib,  xgb_q50.joblib,  xgb_q90.joblib
  models/catboost_q10.joblib, catboost_q50.joblib, catboost_q90.joblib
  models/model_meta.joblib   (unified metadata with all metrics)
  data/processed/train_processed.parquet  (real aggregated data)
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_log.txt", mode="w"),
    ],
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import config


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD & PREPROCESS REAL DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_real_data() -> pd.DataFrame:
    """
    Load the 4.65 GB Kaggle Favorita train.csv in memory-efficient chunks.

    Strategy
    --------
    - Read 500k rows per chunk.
    - Map item_nbr → family using items.csv.
    - Aggregate each chunk to (date, store_nbr, family) level immediately.
    - Cap total item-level rows read at 5M (covers ~3.5 years of data).
    - Merge stores, holidays, oil price, and transactions.
    - Keep ALL 54 stores and TOP 25 product families by total sales volume.
    """
    log.info("=" * 70)
    log.info("STEP 1: Loading real Favorita dataset")
    log.info("=" * 70)

    # Build item → family lookup
    items_df = pd.read_csv(config.RAW_DIR / "items.csv", usecols=["item_nbr", "family"])
    item_to_family = dict(zip(items_df["item_nbr"], items_df["family"]))
    log.info("Loaded items.csv — %d unique items", len(item_to_family))

    # Read train.csv in chunks and aggregate immediately to save RAM
    CHUNK_SIZE = 500_000
    ROW_LIMIT  = 5_000_000   # ~3.5 years of item-level data

    agg_chunks = []
    total_read = 0
    t0 = time.time()

    log.info("Reading train.csv in chunks (target: %d item-level rows)…", ROW_LIMIT)
    for chunk in pd.read_csv(config.RAW_DIR / "train.csv", chunksize=CHUNK_SIZE):
        chunk = chunk.rename(columns={"unit_sales": "sales"})
        chunk["sales"] = chunk["sales"].clip(lower=0)   # clip returns/negatives
        chunk["family"] = chunk["item_nbr"].map(item_to_family).fillna("OTHER")
        chunk["onpromotion"] = chunk["onpromotion"].fillna(0).astype(int)

        agg = chunk.groupby(["date", "store_nbr", "family"], as_index=False).agg(
            sales=("sales", "sum"),
            onpromotion=("onpromotion", "sum"),
        )
        agg_chunks.append(agg)
        total_read += len(chunk)

        if total_read % 1_000_000 == 0:
            log.info("  Read %dM rows so far (%.0fs elapsed)…",
                     total_read // 1_000_000, time.time() - t0)

        if total_read >= ROW_LIMIT:
            break

    log.info("Finished reading %d item-level rows in %.1fs", total_read, time.time() - t0)

    # Merge all chunks and re-aggregate (chunks may overlap on same date/store/family)
    df_raw = pd.concat(agg_chunks, ignore_index=True)
    df = df_raw.groupby(["date", "store_nbr", "family"], as_index=False).agg(
        sales=("sales", "sum"),
        onpromotion=("onpromotion", "sum"),
    )
    df["date"] = pd.to_datetime(df["date"])
    log.info("Aggregated to %d store-family-day rows | date range: %s → %s",
             len(df), df.date.min().date(), df.date.max().date())

    # ── Merge auxiliary datasets ────────────────────────────────────────────
    # Stores
    stores = pd.read_csv(config.RAW_DIR / "stores.csv")
    df = df.merge(stores, on="store_nbr", how="left")
    log.info("Merged stores.csv (city, state, type, cluster)")

    # Oil price (fill gaps via forward-fill + interpolation)
    oil = pd.read_csv(config.RAW_DIR / "oil.csv", parse_dates=["date"])
    oil = oil.rename(columns={"dcoilwtico": "oil_price"})
    oil["oil_price"] = oil["oil_price"].ffill().bfill()
    df = df.merge(oil, on="date", how="left")
    df["oil_price"] = df["oil_price"].ffill().bfill().fillna(90.0)
    log.info("Merged oil prices")

    # Holidays (national + regional binary flag)
    hol = pd.read_csv(config.RAW_DIR / "holidays_events.csv", parse_dates=["date"])
    hol = hol[~hol["transferred"]]
    national_holidays = set(
        hol[hol["locale"] == "National"]["date"].dt.normalize()
    )
    df["is_holiday"] = df["date"].dt.normalize().isin(national_holidays).astype(int)
    log.info("Added is_holiday flag (%d national holidays)", len(national_holidays))

    # Transactions (store-day transaction volume as a demand proxy)
    txn = pd.read_csv(config.RAW_DIR / "transactions.csv", parse_dates=["date"])
    df = df.merge(txn, on=["date", "store_nbr"], how="left")
    df["transactions"] = df["transactions"].fillna(df["transactions"].median())
    log.info("Merged transactions.csv")

    # ── Keep top 25 families by total sales ─────────────────────────────────
    top_families = (
        df.groupby("family")["sales"].sum().nlargest(25).index.tolist()
    )
    df = df[df["family"].isin(top_families)].copy()
    log.info("Kept top %d families: %s", len(top_families), top_families)

    # ── Add sales_is_missing flag for validation ─────────────────────────────
    df["sales_is_missing"] = df["sales"].isna().astype(int)
    df["sales"] = df["sales"].fillna(0)

    # ── Sort chronologically ─────────────────────────────────────────────────
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    log.info("Final preprocessed shape: %s", df.shape)
    log.info("Stores: %d | Families: %d | Days: %d",
             df.store_nbr.nunique(),
             df.family.nunique(),
             df.date.nunique())

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_features_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a comprehensive, leakage-free feature set for the full dataset.
    All lags and rolling windows use .shift(>=1) to prevent target leakage.
    """
    log.info("=" * 70)
    log.info("STEP 2: Building features")
    log.info("=" * 70)

    df = df.copy()

    # Encode categoricals as integers
    df["store_id"]  = df["store_nbr"].astype("category").cat.codes
    df["family_id"] = df["family"].astype("category").cat.codes
    store_map  = dict(zip(df["store_nbr"],  df["store_id"]))
    family_map = dict(zip(df["family"],     df["family_id"]))

    # Encode store type
    if "type" in df.columns:
        df["store_type"] = df["type"].astype("category").cat.codes
    if "cluster" in df.columns:
        df["cluster"] = df["cluster"].astype(int)

    # Calendar features
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["year"]         = df["date"].dt.year
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]      = df["date"].dt.quarter
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    # Cyclic encoding
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    # Year index (normalized trend feature)
    min_year = df["year"].min()
    df["year_idx"] = df["year"] - min_year

    log.info("Calendar features added")

    # Group for lag/rolling computation
    grp = df.groupby(["store_nbr", "family"], sort=False)["sales"]

    # Lag features (all shifted by ≥1 to prevent leakage)
    for lag in [1, 7, 14, 21, 28, 56]:
        df[f"lag_{lag}"] = grp.shift(lag)

    log.info("Lag features added (1, 7, 14, 21, 28, 56 days)")

    # Rolling statistics (shift-1 so current day not included)
    for w in [7, 14, 30, 90]:
        shifted = grp.shift(1)
        rolled  = df.groupby(["store_nbr", "family"], sort=False).apply(
            lambda g: g["sales"].shift(1).rolling(w, min_periods=1).mean()
        ).reset_index(level=[0, 1], drop=True)
        df[f"roll_mean_{w}"]  = rolled
        df[f"roll_std_{w}"]   = df.groupby(["store_nbr", "family"], sort=False).apply(
            lambda g: g["sales"].shift(1).rolling(w, min_periods=1).std()
        ).reset_index(level=[0, 1], drop=True)

    log.info("Rolling mean/std features added (7, 14, 30, 90 days)")

    # Exponential moving average
    df["ewma_7"] = df.groupby(["store_nbr", "family"], sort=False)["sales"].transform(
        lambda s: s.shift(1).ewm(span=7, min_periods=1).mean()
    )
    df["ewma_28"] = df.groupby(["store_nbr", "family"], sort=False)["sales"].transform(
        lambda s: s.shift(1).ewm(span=28, min_periods=1).mean()
    )

    # Oil momentum (7-day change)
    df["oil_7d_change"] = df["oil_price"] - df["oil_price"].shift(7)
    df["oil_7d_change"] = df["oil_7d_change"].fillna(0)

    # Promotion rolling
    if "onpromotion" in df.columns:
        df["promo_lag1"]  = df.groupby(["store_nbr", "family"], sort=False)["onpromotion"].shift(1)
        df["promo_roll7"] = df.groupby(["store_nbr", "family"], sort=False)["onpromotion"].transform(
            lambda s: s.shift(1).rolling(7, min_periods=1).sum()
        )

    # Store-level mean sales (historical)
    store_mean = df.groupby("store_nbr")["sales"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    df["store_mean_sales"] = store_mean

    # Family-level mean sales (historical)
    family_mean = df.groupby("family")["sales"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    df["family_mean_sales"] = family_mean

    log.info("Advanced features added (EWMAs, oil momentum, promo, store/family means)")

    # Drop rows with NaN targets — these are from lag warm-up period
    before = len(df)
    df = df.dropna(subset=["sales"]).reset_index(drop=True)
    log.info("Dropped %d rows with NaN sales | remaining: %d", before - len(df), len(df))

    # Fill remaining feature NaNs with 0 (lag warm-up for new series)
    feature_cols_to_fill = [c for c in df.columns if c not in ["date", "family", "city", "state", "type"]]
    df[feature_cols_to_fill] = df[feature_cols_to_fill].fillna(0)

    return df, store_map, family_map


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (exclude meta/target columns)."""
    exclude = {
        "date", "sales", "sales_is_missing",
        "store_nbr", "family",        # keep encoded versions store_id/family_id
        "city", "state", "type",       # string categoricals already encoded
    }
    return [c for c in df.columns if c not in exclude and df[c].dtype != object]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: TRAIN / VALIDATION SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(df: pd.DataFrame, train_frac: float = 0.80):
    dates     = sorted(df["date"].unique())
    split_idx = int(len(dates) * train_frac)
    split_dt  = dates[split_idx]

    train = df[df["date"] < split_dt].copy()
    val   = df[df["date"] >= split_dt].copy()

    log.info("Train: %d rows (%s → %s)",
             len(train), train.date.min().date(), train.date.max().date())
    log.info("Val:   %d rows (%s → %s)",
             len(val),   val.date.min().date(),   val.date.max().date())
    return train, val, str(pd.Timestamp(split_dt).date())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: METRICS
# ─────────────────────────────────────────────────────────────────────────────

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def wape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")

def evaluate_model(y_true, q10, q50, q90, model_name: str) -> dict:
    metrics = {
        "model":    model_name,
        "MAE":      round(mae(y_true, q50), 4),
        "RMSE":     round(rmse(y_true, q50), 4),
        "WAPE_%":   round(wape(y_true, q50) * 100, 2),
        "PB_Q10":   round(pinball_loss(y_true, q10, 0.10), 4),
        "PB_Q50":   round(pinball_loss(y_true, q50, 0.50), 4),
        "PB_Q90":   round(pinball_loss(y_true, q90, 0.90), 4),
        "Coverage_80%": round(
            float(np.mean((y_true >= q10) & (y_true <= q90))) * 100, 2
        ),
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────

def fix_crossing(q10, q50, q90):
    stacked = np.sort(np.stack([q10, q50, q90], axis=1), axis=1)
    stacked = np.clip(stacked, 0, None)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


def train_lightgbm(X_train, y_train, X_val, y_val):
    from lightgbm import LGBMRegressor

    models = {}
    t_start = time.time()
    for alpha, name in [(0.10, "q10"), (0.50, "q50"), (0.90, "q90")]:
        log.info("  LightGBM %s…", name)
        mdl = LGBMRegressor(
            objective="quantile", alpha=alpha, metric="quantile",
            n_estimators=600, learning_rate=0.05,
            num_leaves=63, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        mdl.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[],
        )
        models[name] = mdl

    log.info("  LightGBM trained in %.1fs", time.time() - t_start)
    return models


def train_xgboost(X_train, y_train, X_val, y_val):
    from xgboost import XGBRegressor

    models = {}
    t_start = time.time()
    for alpha, name in [(0.10, "q10"), (0.50, "q50"), (0.90, "q90")]:
        log.info("  XGBoost %s…", name)
        mdl = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            n_estimators=600, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1, random_state=42,
            verbosity=0,
        )
        mdl.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        models[name] = mdl

    log.info("  XGBoost trained in %.1fs", time.time() - t_start)
    return models


def train_catboost(X_train, y_train, X_val, y_val):
    from catboost import CatBoostRegressor

    models = {}
    t_start = time.time()
    for alpha, name in [(0.10, "q10"), (0.50, "q50"), (0.90, "q90")]:
        log.info("  CatBoost %s…", name)
        mdl = CatBoostRegressor(
            loss_function=f"Quantile:alpha={alpha}",
            iterations=600, learning_rate=0.05,
            depth=6, subsample=0.8,
            random_seed=42, verbose=0,
        )
        mdl.fit(X_train, y_train, eval_set=(X_val, y_val))
        models[name] = mdl

    log.info("  CatBoost trained in %.1fs", time.time() - t_start)
    return models


def seasonal_naive_predict(val_df: pd.DataFrame) -> tuple:
    """Lag-7 seasonal naive baseline."""
    q50 = val_df["lag_7"].fillna(val_df["lag_1"]).fillna(0).values
    q10 = q50 * 0.75
    q90 = q50 * 1.25
    return fix_crossing(q10, q50, q90)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: ENSEMBLE (DYNAMIC WEIGHTING)
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_predict(model_preds: dict, val_metrics: dict) -> tuple:
    """
    Combine model predictions using inverse-pinball-loss weights.
    Higher pinball loss → lower weight.
    """
    # Only ensemble the ML models (exclude naive)
    ml_models = [m for m in model_preds if m != "naive"]

    weights = {}
    total_w = 0.0
    for m in ml_models:
        loss = val_metrics[m]["PB_Q50"]
        w = 1.0 / (loss + 1e-6)
        weights[m] = w
        total_w += w

    # Normalize
    for m in ml_models:
        weights[m] /= total_w

    log.info("Ensemble weights: %s",
             {m: f"{w:.4f}" for m, w in weights.items()})

    q10 = sum(weights[m] * model_preds[m]["q10"] for m in ml_models)
    q50 = sum(weights[m] * model_preds[m]["q50"] for m in ml_models)
    q90 = sum(weights[m] * model_preds[m]["q90"] for m in ml_models)

    return fix_crossing(q10, q50, q90), weights


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    pipeline_start = time.time()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df_raw = load_real_data()

    # ── 2. Build features ─────────────────────────────────────────────────────
    df, store_map, family_map = build_features_full(df_raw)

    # Save processed data (replaces old demo parquet)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PROCESSED_TRAIN, index=False)
    log.info("Saved processed parquet → %s", config.PROCESSED_TRAIN)

    feature_cols = get_feature_cols(df)
    log.info("Feature count: %d", len(feature_cols))
    log.info("Top features: %s", feature_cols[:20])

    # ── 3. Train/val split ────────────────────────────────────────────────────
    train_df, val_df, split_date = chronological_split(df, train_frac=0.80)

    X_train = train_df[feature_cols].values
    y_train = train_df["sales"].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df["sales"].values

    log.info("\n" + "=" * 70)
    log.info("STEP 3: Training all models")
    log.info("=" * 70)
    log.info("Train rows: %d | Val rows: %d | Features: %d",
             len(X_train), len(X_val), len(feature_cols))

    model_preds  = {}
    all_models   = {}

    # ── 4. Seasonal Naive ──────────────────────────────────────────────────────
    log.info("\n--- Seasonal Naive Baseline ---")
    sn_q10, sn_q50, sn_q90 = seasonal_naive_predict(val_df)
    model_preds["naive"] = {"q10": sn_q10, "q50": sn_q50, "q90": sn_q90}

    # ── 5. LightGBM ───────────────────────────────────────────────────────────
    log.info("\n--- LightGBM ---")
    lgbm_models = train_lightgbm(X_train, y_train, X_val, y_val)
    lgbm_q10, lgbm_q50, lgbm_q90 = fix_crossing(
        lgbm_models["q10"].predict(X_val),
        lgbm_models["q50"].predict(X_val),
        lgbm_models["q90"].predict(X_val),
    )
    model_preds["lightgbm"] = {"q10": lgbm_q10, "q50": lgbm_q50, "q90": lgbm_q90}
    all_models["lightgbm"] = lgbm_models

    # Save LightGBM models (primary slot)
    joblib.dump(lgbm_models["q10"], config.MODEL_Q10_PATH)
    joblib.dump(lgbm_models["q50"], config.MODEL_Q50_PATH)
    joblib.dump(lgbm_models["q90"], config.MODEL_Q90_PATH)
    log.info("Saved LightGBM models")

    # ── 6. XGBoost ────────────────────────────────────────────────────────────
    log.info("\n--- XGBoost ---")
    xgb_models = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_q10, xgb_q50, xgb_q90 = fix_crossing(
        xgb_models["q10"].predict(X_val),
        xgb_models["q50"].predict(X_val),
        xgb_models["q90"].predict(X_val),
    )
    model_preds["xgboost"] = {"q10": xgb_q10, "q50": xgb_q50, "q90": xgb_q90}
    all_models["xgboost"] = xgb_models

    joblib.dump(xgb_models["q10"], config.MODELS_DIR / "xgb_q10.joblib")
    joblib.dump(xgb_models["q50"], config.MODELS_DIR / "xgb_q50.joblib")
    joblib.dump(xgb_models["q90"], config.MODELS_DIR / "xgb_q90.joblib")
    log.info("Saved XGBoost models")

    # ── 7. CatBoost ───────────────────────────────────────────────────────────
    log.info("\n--- CatBoost ---")
    cat_models = train_catboost(X_train, y_train, X_val, y_val)
    cat_q10, cat_q50, cat_q90 = fix_crossing(
        np.array(cat_models["q10"].predict(X_val)),
        np.array(cat_models["q50"].predict(X_val)),
        np.array(cat_models["q90"].predict(X_val)),
    )
    model_preds["catboost"] = {"q10": cat_q10, "q50": cat_q50, "q90": cat_q90}
    all_models["catboost"] = cat_models

    joblib.dump(cat_models["q10"], config.MODELS_DIR / "catboost_q10.joblib")
    joblib.dump(cat_models["q50"], config.MODELS_DIR / "catboost_q50.joblib")
    joblib.dump(cat_models["q90"], config.MODELS_DIR / "catboost_q90.joblib")
    log.info("Saved CatBoost models")

    # ── 8. Evaluate all models ────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("STEP 4: Evaluation on Validation Set")
    log.info("=" * 70)

    val_metrics = {}
    for name, preds in model_preds.items():
        m = evaluate_model(y_val, preds["q10"], preds["q50"], preds["q90"], name)
        val_metrics[name] = m

    # ── 9. Ensemble ───────────────────────────────────────────────────────────
    log.info("\n--- Hybrid Ensemble ---")
    (ens_q10, ens_q50, ens_q90), ens_weights = ensemble_predict(model_preds, val_metrics)
    model_preds["ensemble"] = {"q10": ens_q10, "q50": ens_q50, "q90": ens_q90}
    m_ens = evaluate_model(y_val, ens_q10, ens_q50, ens_q90, "ensemble")
    val_metrics["ensemble"] = m_ens

    # ── 10. Print results table ───────────────────────────────────────────────
    results_df = pd.DataFrame(list(val_metrics.values()))
    results_df = results_df.set_index("model")

    log.info("\n%s", "=" * 70)
    log.info("MODEL COMPARISON RESULTS (Validation Set)")
    log.info("%s", "=" * 70)
    log.info("\n%s", results_df.to_string())

    # ── 11. Determine best model ──────────────────────────────────────────────
    ml_metrics = {m: v for m, v in val_metrics.items() if m != "naive"}
    best_model  = min(ml_metrics, key=lambda m: ml_metrics[m]["WAPE_%"])
    log.info("\nBest model by WAPE: %s (WAPE=%.2f%%)",
             best_model, ml_metrics[best_model]["WAPE_%"])

    # ── 12. SHAP Feature Importance (LightGBM Q50) ───────────────────────────
    log.info("\n" + "=" * 70)
    log.info("STEP 5: SHAP Feature Importance")
    log.info("=" * 70)
    try:
        import shap
        # Use a sample of 2000 rows for speed
        sample_idx = np.random.RandomState(42).choice(len(X_val), min(2000, len(X_val)), replace=False)
        X_sample = X_val[sample_idx]

        explainer = shap.TreeExplainer(lgbm_models["q50"])
        shap_values = explainer.shap_values(X_sample)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature":    feature_cols,
            "mean_|SHAP|": mean_abs_shap,
        }).sort_values("mean_|SHAP|", ascending=False)

        log.info("Top 15 features by SHAP importance:")
        log.info("\n%s", shap_df.head(15).to_string(index=False))

        shap_df.to_csv(config.REPORTS_DIR / "shap_feature_importance.csv", index=False)
        log.info("Saved SHAP importance → reports/shap_feature_importance.csv")
    except Exception as e:
        log.warning("SHAP computation failed: %s", e)
        shap_df = pd.DataFrame()

    # ── 13. Save comprehensive metadata ──────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("STEP 6: Saving unified model metadata")
    log.info("=" * 70)

    # Build val_df with all predictions
    val_out = val_df[["date", "store_nbr", "family", "sales"]].copy()
    val_out["pred_q10"]     = model_preds["lightgbm"]["q10"]
    val_out["pred_q50"]     = model_preds["lightgbm"]["q50"]
    val_out["pred_q90"]     = model_preds["lightgbm"]["q90"]
    val_out["xgb_q50"]      = model_preds["xgboost"]["q50"]
    val_out["cat_q50"]      = model_preds["catboost"]["q50"]
    val_out["ens_q50"]      = model_preds["ensemble"]["q50"]
    val_out["naive_q50"]    = model_preds["naive"]["q50"]

    meta = {
        "feature_cols":   feature_cols,
        "store_map":      store_map,
        "family_map":     family_map,
        "split_date":     split_date,
        "train_rows":     len(train_df),
        "val_rows":       len(val_df),
        "stores":         sorted(df["store_nbr"].unique().tolist()),
        "families":       sorted(df["family"].unique().tolist()),
        "data_source":    "real_favorita_kaggle",
        "best_model":     best_model,
        "ensemble_weights": ens_weights,
        "metrics":        val_metrics,
        "lgbm_params":    {
            "n_estimators": 600, "learning_rate": 0.05,
            "num_leaves": 63, "min_child_samples": 30,
        },
        "data_profile": {
            "total_rows_read":   5_000_000,
            "aggregated_rows":   len(df),
            "date_min":          str(df.date.min().date()),
            "date_max":          str(df.date.max().date()),
            "n_stores":          df.store_nbr.nunique(),
            "n_families":        df.family.nunique(),
        },
    }

    joblib.dump(meta, config.MODEL_META_PATH)
    log.info("Saved model metadata → %s", config.MODEL_META_PATH)

    # Save validation predictions
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    val_out.to_parquet(config.REPORTS_DIR / "validation_predictions.parquet", index=False)
    val_out.head(10000).to_csv(config.REPORTS_DIR / "validation_predictions_sample.csv", index=False)
    results_df.to_csv(config.REPORTS_DIR / "model_comparison.csv")
    log.info("Saved validation predictions and model comparison → reports/")

    # ── 14. Final summary ─────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    log.info("\n" + "=" * 70)
    log.info("TRAINING COMPLETE in %.1f minutes", total_time / 60)
    log.info("=" * 70)
    log.info("\nModel Performance Summary:")
    for m, metrics in val_metrics.items():
        log.info(
            "  %-12s  MAE=%7.2f  RMSE=%7.2f  WAPE=%5.2f%%  PB_Q50=%6.4f  Coverage=%5.1f%%",
            m.upper(),
            metrics["MAE"], metrics["RMSE"],
            metrics["WAPE_%"], metrics["PB_Q50"],
            metrics["Coverage_80%"],
        )
    log.info("\nBest model: %s (WAPE=%.2f%%)",
             best_model.upper(), ml_metrics[best_model]["WAPE_%"])
    log.info("\nAll artifacts saved to:")
    log.info("  models/                     — trained model files")
    log.info("  data/processed/             — preprocessed parquet")
    log.info("  reports/                    — evaluation metrics & predictions")
    log.info("  training_log.txt            — full training log")
    log.info("\nTo launch the dashboard: streamlit run app.py")
    log.info("To launch the API:        uvicorn api:app --port 8000")


if __name__ == "__main__":
    main()
