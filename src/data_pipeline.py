"""
src/data_pipeline.py — Data ingestion, validation, and preprocessing.

Responsibilities
----------------
* Load raw CSV files from data/raw/.
* Validate required columns and data types.
* Merge optional metadata (stores, holidays, oil).
* Filter to a configurable subset of stores and product families.
* Detect and report anomalous/missing values.
* Save the processed dataset for downstream modules.

No Streamlit imports — this module is independently testable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ── Public helpers ─────────────────────────────────────────────────────────────

def load_csv(path: Path, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    """Load a CSV file and optionally parse date columns.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, parse_dates=parse_dates or [])
    if df.empty:
        raise ValueError(f"CSV file is empty: {path}")

    logger.info("Loaded %s — shape %s", path.name, df.shape)
    return df


def validate_columns(df: pd.DataFrame, required: list[str], source: str = "") -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {source or 'dataframe'}: {missing}"
        )


def parse_dates_inplace(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Convert *col* to datetime if it is not already."""
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")
        n_bad = df[col].isna().sum()
        if n_bad:
            logger.warning("Dropped %d rows with unparseable dates.", n_bad)
            df = df.dropna(subset=[col])
    return df


def drop_duplicates(
    df: pd.DataFrame,
    subset: Optional[list[str]] = None,
    keep: str = "last",
) -> pd.DataFrame:
    """Remove duplicate rows and log the count."""
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows.", dropped)
    return df


def validate_sales(df: pd.DataFrame, col: str = "sales") -> pd.DataFrame:
    """
    Flag and handle invalid sales values.

    Rules
    -----
    * NaN sales are *not* automatically replaced with zero — they are retained
      with a flag column ``sales_is_missing`` so that callers can decide how to
      treat them.
    * Negative sales are replaced with NaN and flagged.
    """
    df = df.copy()
    df["sales_is_missing"] = df[col].isna()

    neg_mask = df[col] < 0
    if neg_mask.any():
        n = int(neg_mask.sum())
        logger.warning("Setting %d negative sales values to NaN.", n)
        df.loc[neg_mask, col] = np.nan
        df.loc[neg_mask, "sales_is_missing"] = True

    return df


def filter_subset(
    df: pd.DataFrame,
    max_stores: Optional[int] = config.MAX_STORES,
    max_families: Optional[int] = config.MAX_FAMILIES,
    store_col: str = "store_nbr",
    family_col: str = "family",
) -> pd.DataFrame:
    """Keep the first *max_stores* stores and *max_families* families by
    total sales volume, to keep training tractable on limited hardware.

    The same filter must be applied consistently to training and validation
    data — callers should pass the exact store/family lists from training.
    """
    if max_stores is not None and store_col in df.columns:
        top_stores = (
            df.groupby(store_col)["sales"].sum().nlargest(max_stores).index.tolist()
        )
        df = df[df[store_col].isin(top_stores)]
        logger.info("Filtered to %d stores.", len(top_stores))

    if max_families is not None and family_col in df.columns:
        top_families = (
            df.groupby(family_col)["sales"].sum().nlargest(max_families).index.tolist()
        )
        df = df[df[family_col].isin(top_families)]
        logger.info("Filtered to %d product families.", len(top_families))

    return df


def merge_stores(df: pd.DataFrame, stores_path: Path = config.STORES_CSV) -> pd.DataFrame:
    """Left-join store metadata (city, state, type, cluster) if the file exists."""
    if not stores_path.exists():
        logger.info("stores.csv not found — skipping store metadata merge.")
        return df

    stores = load_csv(stores_path)
    df = df.merge(stores, on="store_nbr", how="left")
    logger.info("Merged store metadata — new shape %s", df.shape)
    return df


def merge_holidays(df: pd.DataFrame, holidays_path: Path = config.HOLIDAYS_CSV) -> pd.DataFrame:
    """Add a binary ``is_holiday`` flag from the holidays_events file."""
    if not holidays_path.exists():
        logger.info("holidays_events.csv not found — skipping holiday merge.")
        df["is_holiday"] = 0
        return df

    holidays = load_csv(holidays_path, parse_dates=["date"])
    # Keep only rows where the holiday was not transferred/cancelled
    holidays = holidays[holidays["transferred"].eq(False) | holidays["transferred"].isna()]
    holiday_dates = set(holidays["date"].dt.normalize())

    df = df.copy()
    df["is_holiday"] = df["date"].dt.normalize().isin(holiday_dates).astype(int)
    logger.info("Added is_holiday flag.")
    return df


def merge_oil(df: pd.DataFrame, oil_path: Path = config.OIL_CSV) -> pd.DataFrame:
    """Merge daily Brent oil price as a macro feature."""
    if not oil_path.exists():
        logger.info("oil.csv not found — skipping oil price merge.")
        return df

    oil = load_csv(oil_path, parse_dates=["date"])
    oil = oil.rename(columns={"dcoilwtico": "oil_price"})
    oil["oil_price"] = oil["oil_price"].interpolate(method="linear")  # fill small gaps

    df = df.merge(oil[["date", "oil_price"]], on="date", how="left")
    logger.info("Merged oil prices.")
    return df


def load_and_preprocess(
    train_path: Path = config.TRAIN_CSV,
    max_rows: Optional[int] = config.MAX_TRAIN_ROWS,
) -> tuple[pd.DataFrame, dict]:
    """
    Full ingestion pipeline for the Kaggle Favorita dataset.
    Supports both:
    1. Aggregated store-family format (date, store_nbr, family, sales, onpromotion)
    2. Item-level format (date, store_nbr, item_nbr, unit_sales, onpromotion) with items.csv

    Reads chunked to stay within 8 GB RAM limits.
    """
    items_csv = config.RAW_DIR / "items.csv"
    items_map: dict[int, str] = {}
    if items_csv.exists():
        logger.info("Found items.csv — loading item_nbr to family mapping...")
        df_items = pd.read_csv(items_csv, usecols=["item_nbr", "family"])
        items_map = dict(zip(df_items["item_nbr"], df_items["family"]))

    # Peek at header to determine schema
    header_df = pd.read_csv(train_path, nrows=5)
    is_item_level = "item_nbr" in header_df.columns and "family" not in header_df.columns

    if is_item_level:
        logger.info("Detected item-level Kaggle train.csv. Processing in chunks to aggregate by store and family...")
        # To keep training fast and memory safe on 8GB RAM, read the last chunks or filter top stores/families
        chunks = []
        chunk_size = 500_000
        total_rows_read = 0
        limit_rows = 2_000_000  # sample sufficient history (e.g. recent months)

        # Read in chunks and aggregate
        for chunk in pd.read_csv(train_path, chunksize=chunk_size):
            if "unit_sales" in chunk.columns:
                chunk = chunk.rename(columns={"unit_sales": "sales"})

            # Filter negative sales early (returns/cancellations)
            chunk["sales"] = chunk["sales"].clip(lower=0)

            # Map family
            if items_map:
                chunk["family"] = chunk["item_nbr"].map(items_map)
            else:
                chunk["family"] = "ALL"

            # Clean promotion
            if "onpromotion" in chunk.columns:
                chunk["onpromotion"] = chunk["onpromotion"].fillna(False).astype(bool).astype(int)
            else:
                chunk["onpromotion"] = 0

            # Aggregate chunk to date x store_nbr x family level
            agg_chunk = chunk.groupby(["date", "store_nbr", "family"], as_index=False).agg({
                "sales": "sum",
                "onpromotion": "sum"
            })
            chunks.append(agg_chunk)
            total_rows_read += len(chunk)

            if total_rows_read >= limit_rows:
                logger.info("Read %d item-level rows for local training subset.", total_rows_read)
                break

        df_raw = pd.concat(chunks, ignore_index=True)
        # Re-aggregate across chunks
        df = df_raw.groupby(["date", "store_nbr", "family"], as_index=False).agg({
            "sales": "sum",
            "onpromotion": "sum"
        })
    else:
        df = load_csv(train_path, parse_dates=["date"])
        validate_columns(df, config.REQUIRED_COLS, source="train.csv")

    df = parse_dates_inplace(df)
    df = drop_duplicates(df, subset=["date", "store_nbr", "family"])
    df = validate_sales(df)

    # Merge optional metadata
    df = merge_stores(df)
    df = merge_holidays(df)
    df = merge_oil(df)


    # Apply configurable subset BEFORE row cap so the cap is on filtered data
    df = filter_subset(df)

    # Sort chronologically — essential for leakage-free feature engineering
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Hard cap on rows (applied after sorting so we keep the most recent history)
    if max_rows and len(df) > max_rows:
        # Keep the LAST max_rows rows (most recent) for each series
        df = (
            df.groupby(["store_nbr", "family"], group_keys=False)
            .apply(lambda g: g.tail(max_rows // df["store_nbr"].nunique() // df["family"].nunique() + 1))
            .reset_index(drop=True)
        )
        logger.info("Row cap applied — final shape %s", df.shape)

    meta = {
        "stores"        : sorted(df["store_nbr"].unique().tolist()),
        "families"      : sorted(df["family"].unique().tolist()),
        "date_min"      : str(df["date"].min().date()),
        "date_max"      : str(df["date"].max().date()),
        "n_rows"        : len(df),
        "data_source"   : "real",
    }

    return df, meta


def save_processed(df: pd.DataFrame, path: Path = config.PROCESSED_TRAIN) -> None:
    """Save processed DataFrame to parquet for fast reloading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved processed data to %s", path)


def load_processed(path: Path = config.PROCESSED_TRAIN) -> pd.DataFrame:
    """Load the saved processed parquet file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. "
            "Run the training pipeline first."
        )
    return pd.read_parquet(path)


# ── Dataset summary helpers ────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a dict of statistics suitable for display in the Overview tab."""
    return {
        "rows"              : len(df),
        "stores"            : int(df["store_nbr"].nunique()),
        "families"          : int(df["family"].nunique()),
        "date_min"          : str(df["date"].min().date()),
        "date_max"          : str(df["date"].max().date()),
        "missing_sales_pct" : round(df["sales_is_missing"].mean() * 100, 2)
                              if "sales_is_missing" in df.columns else 0.0,
        "promoted_rows_pct" : round((df["onpromotion"] > 0).mean() * 100, 2)
                              if "onpromotion" in df.columns else 0.0,
    }
