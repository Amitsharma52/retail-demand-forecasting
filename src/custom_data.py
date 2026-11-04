"""
src/custom_data.py — Merchant POS / ERP Ingestion & Preprocessing Pipeline.

Empowers real retail merchants and independent shopkeepers to upload their own
historical Point-of-Sale (POS) or inventory CSV/Excel exports into the
AI Demand Forecasting & Replenishment engine.

Features:
- Robust fuzzy column mapping (recognizes varied date/SKU/sales column names)
- Calendar regularization: identifies missing store-open days and fills gaps with 0 sales
- Automated aggregation: collapses multiple transactions per day into daily aggregate sales
- Automated feature generation: builds calendar, lag, and rolling statistics on custom uploads
- Diagnostic data health auditor: checks for missing data, zero-sales ratios, and minimum horizon
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Known Synonyms for Automatic Column Mapping ────────────────────────────────

COLUMN_SYNONYMS = {
    "date": ["date", "timestamp", "transaction_date", "day", "order_date", "sale_date", "dt"],
    "store_nbr": ["store_nbr", "store", "store_id", "branch", "location", "outlet_id", "shop_id"],
    "family": ["family", "product_family", "category", "product_category", "sku", "item", "item_name", "product"],
    "sales": ["sales", "units_sold", "quantity", "qty", "demand", "volume", "units", "sold"],
    "onpromotion": ["onpromotion", "promo", "is_promo", "promotion", "discount_active", "has_discount"],
}


@dataclass
class IngestionReport:
    """Diagnostic validation and data health report for an uploaded dataset."""
    is_valid: bool
    status_message: str
    row_count: int
    store_count: int
    family_count: int
    min_date: str
    max_date: str
    total_days: int
    zero_sales_pct: float
    mapped_columns: Dict[str, str]
    unmapped_columns: List[str]
    warnings: List[str]


def identify_columns(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    """
    Fuzzy-matches DataFrame columns against known retail schemas.
    Returns (mapped_dict, unmapped_list).
    """
    normalized_cols = {col.strip().lower().replace(" ", "_"): col for col in df.columns}
    mapped = {}

    for target_key, synonyms in COLUMN_SYNONYMS.items():
        found = False
        for syn in synonyms:
            if syn in normalized_cols:
                mapped[target_key] = normalized_cols[syn]
                found = True
                break
        if not found and target_key in ("store_nbr", "onpromotion"):
            # Optional columns
            mapped[target_key] = None

    unmapped = [col for col in df.columns if col not in mapped.values() and mapped.get("date") != col]
    return mapped, unmapped


def ingest_merchant_dataset(
    file_or_df: Union[str, bytes, io.BytesIO, pd.DataFrame],
    file_format: str = "csv",
    custom_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, IngestionReport]:
    """
    Load, validate, map, and regularize a merchant's custom sales dataset.

    Parameters
    ----------
    file_or_df : File path, raw bytes, file buffer, or pre-loaded DataFrame
    file_format: 'csv' or 'excel'
    custom_mapping: Optional dictionary explicitly overriding column mappings
                    e.g. {'date': 'TxDate', 'sales': 'PiecesSold', 'family': 'Category'}

    Returns
    -------
    regularized_df : Cleaned daily aggregated DataFrame ready for feature generation
    report         : IngestionReport data health summary
    """
    warnings: List[str] = []

    # 1. Load data
    if isinstance(file_or_df, pd.DataFrame):
        df_raw = file_or_df.copy()
    elif file_format.lower() in ("xlsx", "xls", "excel"):
        df_raw = pd.read_excel(file_or_df)
    else:
        df_raw = pd.read_csv(file_or_df)

    if df_raw.empty:
        return pd.DataFrame(), IngestionReport(
            is_valid=False,
            status_message="Uploaded file contains zero rows.",
            row_count=0, store_count=0, family_count=0,
            min_date="", max_date="", total_days=0, zero_sales_pct=0.0,
            mapped_columns={}, unmapped_columns=[], warnings=["Empty file"],
        )

    # 2. Map Columns
    auto_mapped, unmapped = identify_columns(df_raw)
    final_mapping = auto_mapped.copy()
    if custom_mapping:
        final_mapping.update(custom_mapping)

    # Validate essential columns
    for req in ["date", "family", "sales"]:
        if not final_mapping.get(req):
            return pd.DataFrame(), IngestionReport(
                is_valid=False,
                status_message=f"Missing essential column '{req}'. Could not automatically identify it from: {list(df_raw.columns)}",
                row_count=len(df_raw), store_count=0, family_count=0,
                min_date="", max_date="", total_days=0, zero_sales_pct=0.0,
                mapped_columns=final_mapping, unmapped_columns=unmapped,
                warnings=[f"Column '{req}' not found."],
            )

    # 3. Standardize column names
    rename_dict = {
        final_mapping["date"]: "date",
        final_mapping["family"]: "family",
        final_mapping["sales"]: "sales",
    }
    if final_mapping.get("store_nbr"):
        rename_dict[final_mapping["store_nbr"]] = "store_nbr"
    if final_mapping.get("onpromotion"):
        rename_dict[final_mapping["onpromotion"]] = "onpromotion"

    df = df_raw.rename(columns=rename_dict)

    # Add defaults if optional columns are absent
    if "store_nbr" not in df.columns:
        df["store_nbr"] = 1
        warnings.append("No store column detected. Defaulted all rows to Store #1.")
    else:
        df["store_nbr"] = df["store_nbr"].fillna(1).astype(int)

    if "onpromotion" not in df.columns:
        df["onpromotion"] = 0
    else:
        # Convert to binary int 0/1
        df["onpromotion"] = df["onpromotion"].apply(
            lambda v: 1 if str(v).strip().lower() in ("1", "true", "yes", "y", "t") else 0
        )

    # 4. Clean Date & Types
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    except Exception as e:
        return pd.DataFrame(), IngestionReport(
            is_valid=False,
            status_message=f"Failed to parse date column: {e}",
            row_count=len(df_raw), store_count=0, family_count=0,
            min_date="", max_date="", total_days=0, zero_sales_pct=0.0,
            mapped_columns=final_mapping, unmapped_columns=unmapped,
            warnings=["Date parse failure"],
        )

    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame(), IngestionReport(
            is_valid=False,
            status_message="No valid date values found after datetime conversion.",
            row_count=len(df_raw), store_count=0, family_count=0,
            min_date="", max_date="", total_days=0, zero_sales_pct=0.0,
            mapped_columns=final_mapping, unmapped_columns=unmapped,
            warnings=["All dates null or invalid"],
        )

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0.0)
    df["sales"] = df["sales"].clip(lower=0.0)
    df["family"] = df["family"].astype(str).str.strip().str.upper()

    # 5. Aggregate by (date, store_nbr, family) to handle multiple intraday tickets
    agg_df = (
        df.groupby(["date", "store_nbr", "family"], as_index=False)
        .agg({"sales": "sum", "onpromotion": "max"})
    )

    # 6. Grid Regularization: Ensure unbroken daily timeline for each (store, family)
    min_date = agg_df["date"].min()
    max_date = agg_df["date"].max()
    full_dates = pd.date_range(min_date, max_date, freq="D")
    total_days = len(full_dates)

    if total_days < 28:
        warnings.append(
            f"Dataset spans only {total_days} days. Reliable lag & seasonality modeling typically requires >= 60 days."
        )

    stores = agg_df["store_nbr"].unique()
    families = agg_df["family"].unique()

    # Create full Cartesian index
    idx = pd.MultiIndex.from_product(
        [full_dates, stores, families],
        names=["date", "store_nbr", "family"]
    )
    regularized_df = (
        agg_df.set_index(["date", "store_nbr", "family"])
        .reindex(idx)
        .reset_index()
    )
    regularized_df["sales"] = regularized_df["sales"].fillna(0.0)
    regularized_df["onpromotion"] = regularized_df["onpromotion"].fillna(0).astype(int)
    regularized_df = regularized_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # 7. Compute Health Metrics
    total_records = len(regularized_df)
    zero_sales_count = (regularized_df["sales"] == 0).sum()
    zero_pct = float(zero_sales_count / max(1, total_records))

    if zero_pct > 0.85:
        warnings.append(f"High intermittency: {zero_pct*100:.1f}% of daily records have 0 sales.")

    report = IngestionReport(
        is_valid=True,
        status_message="Dataset successfully ingested, validated, and regularized.",
        row_count=total_records,
        store_count=len(stores),
        family_count=len(families),
        min_date=str(min_date.date()),
        max_date=str(max_date.date()),
        total_days=total_days,
        zero_sales_pct=zero_pct,
        mapped_columns=final_mapping,
        unmapped_columns=unmapped,
        warnings=warnings,
    )

    return regularized_df, report


def generate_custom_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct calendar, lag, and rolling features on a regularized merchant dataset.
    Returns (feature_df, list_of_feature_column_names).
    """
    df_feat = df.copy()
    df_feat["date"] = pd.to_datetime(df_feat["date"])

    # Calendar features
    df_feat["dayofweek"] = df_feat["date"].dt.dayofweek
    df_feat["dayofmonth"] = df_feat["date"].dt.day
    df_feat["month"] = df_feat["date"].dt.month
    df_feat["is_weekend"] = df_feat["dayofweek"].isin([5, 6]).astype(int)
    df_feat["quarter"] = df_feat["date"].dt.quarter

    # Lag and Rolling features grouped by store and family
    grouped = df_feat.groupby(["store_nbr", "family"])

    # Lags
    for lag in [1, 7, 14, 21, 28]:
        df_feat[f"lag_{lag}"] = grouped["sales"].shift(lag)

    # Rolling means and stds (shifted by 1 day to strictly avoid lookahead bias!)
    for window in [7, 14, 30]:
        df_feat[f"rolling_mean_{window}"] = (
            grouped["sales"].transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
        )
        df_feat[f"rolling_std_{window}"] = (
            grouped["sales"].transform(lambda x: x.shift(1).rolling(window, min_periods=3).std()).fillna(0.0)
        )

    # Categorical codes
    df_feat["store_code"] = df_feat["store_nbr"].astype("category").cat.codes
    df_feat["family_code"] = df_feat["family"].astype("category").cat.codes

    feature_cols = [
        "dayofweek", "dayofmonth", "month", "is_weekend", "quarter", "onpromotion",
        "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
        "rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_std_14",
        "rolling_mean_30", "rolling_std_30", "store_code", "family_code"
    ]

    # Drop early warm-up rows containing NaNs from the largest lag/window (28 days)
    df_feat = df_feat.dropna(subset=[f"lag_28", "rolling_mean_14"]).reset_index(drop=True)
    return df_feat, feature_cols
