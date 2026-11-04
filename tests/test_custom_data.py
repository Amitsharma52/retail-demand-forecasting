"""
tests/test_custom_data.py — Unit tests for merchant POS data ingestion and processing.
"""

import io
import pandas as pd
import pytest

from src.custom_data import (
    generate_custom_features,
    identify_columns,
    ingest_merchant_dataset,
)


def test_identify_columns_fuzzy_matching():
    df = pd.DataFrame(columns=["Transaction_Date", "Outlet_ID", "Category", "Units_Sold", "Discount_Active"])
    mapped, unmapped = identify_columns(df)

    assert mapped["date"] == "Transaction_Date"
    assert mapped["store_nbr"] == "Outlet_ID"
    assert mapped["family"] == "Category"
    assert mapped["sales"] == "Units_Sold"
    assert mapped["onpromotion"] == "Discount_Active"


def test_ingest_merchant_dataset_csv_buffer():
    csv_content = """TxDate,Branch,SKU,Quantity
2026-01-01,10,COFFEE,5
2026-01-01,10,COFFEE,3
2026-01-02,10,COFFEE,12
2026-01-04,10,COFFEE,8
"""
    buffer = io.StringIO(csv_content)
    custom_map = {"date": "TxDate", "store_nbr": "Branch", "family": "SKU", "sales": "Quantity"}

    regularized_df, report = ingest_merchant_dataset(buffer, file_format="csv", custom_mapping=custom_map)

    assert report.is_valid is True
    assert report.store_count == 1
    assert report.family_count == 1
    assert report.min_date == "2026-01-01"
    assert report.max_date == "2026-01-04"
    assert report.total_days == 4

    # Day 2026-01-01 aggregated 5 + 3 = 8
    day1_sales = regularized_df[regularized_df["date"] == "2026-01-01"]["sales"].iloc[0]
    assert day1_sales == 8.0

    # Day 2026-01-03 was missing, should be filled with 0.0
    day3_sales = regularized_df[regularized_df["date"] == "2026-01-03"]["sales"].iloc[0]
    assert day3_sales == 0.0


def test_generate_custom_features():
    # Create 35 days of regularized data
    dates = pd.date_range("2026-01-01", periods=45, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "store_nbr": 1,
        "family": "SNACKS",
        "sales": [10.0 + (i % 7) for i in range(45)],
        "onpromotion": [1 if i % 10 == 0 else 0 for i in range(45)],
    })

    feat_df, cols = generate_custom_features(df)
    assert len(feat_df) > 0
    assert "lag_7" in cols
    assert "rolling_mean_7" in cols
    assert "dayofweek" in cols
    assert not feat_df[cols].isna().any().any()
