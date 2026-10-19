"""
config.py — Central configuration for the Retail Demand Forecasting System.

All tunable parameters live here so that modules stay independent of hard-coded
values.  Import this module rather than scattering magic numbers through the code.
"""

from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
SAMPLE_DIR     = DATA_DIR / "sample"
MODELS_DIR     = ROOT_DIR / "models"
REPORTS_DIR    = ROOT_DIR / "reports"

# ── Raw Kaggle files ──────────────────────────────────────────────────────────
TRAIN_CSV       = RAW_DIR / "train.csv"
TEST_CSV        = RAW_DIR / "test.csv"
STORES_CSV      = RAW_DIR / "stores.csv"
HOLIDAYS_CSV    = RAW_DIR / "holidays_events.csv"
OIL_CSV         = RAW_DIR / "oil.csv"
TRANSACTIONS_CSV= RAW_DIR / "transactions.csv"

# ── Processed / cached artefacts ──────────────────────────────────────────────
PROCESSED_TRAIN = PROCESSED_DIR / "train_processed.parquet"
FEATURES_PATH   = PROCESSED_DIR / "features.parquet"

# ── Model artefacts ──────────────────────────────────────────────────────────
MODEL_Q10_PATH  = MODELS_DIR / "lgbm_q10.joblib"
MODEL_Q50_PATH  = MODELS_DIR / "lgbm_q50.joblib"
MODEL_Q90_PATH  = MODELS_DIR / "lgbm_q90.joblib"
MODEL_META_PATH = MODELS_DIR / "model_meta.joblib"

# ── Data subset (keep training tractable on 8 GB RAM) ────────────────────────
# Set to None to use all stores / families.
MAX_STORES      : int | None = 5          # number of stores to include
MAX_FAMILIES    : int | None = 10         # number of product families to include
MAX_TRAIN_ROWS  : int | None = 300_000   # hard cap on training rows

# ── Required columns ─────────────────────────────────────────────────────────
REQUIRED_COLS = ["date", "store_nbr", "family", "sales", "onpromotion"]

# ── Feature engineering ───────────────────────────────────────────────────────
LAG_DAYS          = [1, 7, 14, 30]
ROLLING_WINDOWS   = [7, 14, 30]
EWMA_SPAN         = 7

# Categorical columns that LightGBM handles natively
CAT_COLS = ["store_nbr", "family", "day_of_week", "month"]

# ── Chronological split ───────────────────────────────────────────────────────
# Fraction of timeline used for training; the remainder is the holdout
TRAIN_FRACTION  = 0.80

# ── Quantile levels ───────────────────────────────────────────────────────────
Q_LOWER  = 0.10
Q_MEDIAN = 0.50
Q_UPPER  = 0.90

# ── LightGBM hyperparameters ─────────────────────────────────────────────────
LGBM_PARAMS = {
    "n_estimators"    : 500,
    "learning_rate"   : 0.05,
    "num_leaves"      : 63,
    "min_child_samples": 20,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "n_jobs"          : -1,
    "random_state"    : 42,
    "verbose"         : -1,
}

# ── Inventory defaults ────────────────────────────────────────────────────────
DEFAULT_REVIEW_PERIOD  = 7    # days
DEFAULT_LEAD_TIME      = 3    # days
DEFAULT_SERVICE_LEVEL  = 0.95  # 95 %

# ── Dashboard ─────────────────────────────────────────────────────────────────
FORECAST_HORIZON = 7   # days ahead shown by default

# ── Sample-data generation ────────────────────────────────────────────────────
SAMPLE_STORES    = [1, 2, 3]
SAMPLE_FAMILIES  = ["GROCERY I", "BEVERAGES", "PRODUCE", "CLEANING", "DAIRY"]
SAMPLE_START     = "2017-01-01"
SAMPLE_END       = "2017-08-31"
