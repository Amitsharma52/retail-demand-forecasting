"""
tests/test_data_pipeline.py — Unit tests for data ingestion and preprocessing.
"""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import (
    validate_columns,
    parse_dates_inplace,
    drop_duplicates,
    validate_sales,
    filter_subset,
    dataset_summary,
)
from src.sample_data import generate_sample_train


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return generate_sample_train(
        stores=[1, 2],
        families=["GROCERY I", "BEVERAGES"],
        start="2017-01-01",
        end="2017-03-31",
    )


# ── Column validation ──────────────────────────────────────────────────────────

def test_validate_columns_passes(sample_df):
    """Should not raise when all required columns are present."""
    validate_columns(sample_df, ["date", "store_nbr", "family", "sales", "onpromotion"])


def test_validate_columns_raises_on_missing():
    """Should raise ValueError when a column is missing."""
    df = pd.DataFrame({"date": [], "store_nbr": []})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_columns(df, ["date", "store_nbr", "sales"])


# ── Date parsing ───────────────────────────────────────────────────────────────

def test_parse_dates_inplace_converts_string():
    """String dates should be converted to datetime."""
    df = pd.DataFrame({"date": ["2017-01-01", "2017-01-02", "2017-01-03"]})
    df = parse_dates_inplace(df)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_parse_dates_inplace_no_change_if_already_datetime():
    """Already-datetime columns should not be changed."""
    df = pd.DataFrame({"date": pd.date_range("2017-01-01", periods=5)})
    df_after = parse_dates_inplace(df)
    assert pd.api.types.is_datetime64_any_dtype(df_after["date"])


def test_parse_dates_drops_unparseable_rows():
    """Rows with completely unparseable dates should be dropped."""
    df = pd.DataFrame({"date": ["2017-01-01", "not-a-date", "2017-01-03"]})
    df_after = parse_dates_inplace(df)
    assert len(df_after) == 2


# ── Duplicate handling ─────────────────────────────────────────────────────────

def test_drop_duplicates_removes_exact_dupes():
    df = pd.DataFrame({
        "date"    : pd.to_datetime(["2017-01-01", "2017-01-01"]),
        "store_nbr": [1, 1],
        "family"  : ["GROCERY I", "GROCERY I"],
        "sales"   : [10.0, 10.0],
    })
    df_deduped = drop_duplicates(df, subset=["date", "store_nbr", "family"])
    assert len(df_deduped) == 1


def test_drop_duplicates_keeps_last():
    df = pd.DataFrame({
        "date"    : pd.to_datetime(["2017-01-01", "2017-01-01"]),
        "store_nbr": [1, 1],
        "family"  : ["GROCERY I", "GROCERY I"],
        "sales"   : [10.0, 99.0],   # second row has different sales
    })
    df_deduped = drop_duplicates(df, subset=["date", "store_nbr", "family"], keep="last")
    assert df_deduped["sales"].iloc[0] == 99.0


# ── Sales validation ──────────────────────────────────────────────────────────

def test_validate_sales_flags_missing():
    df = pd.DataFrame({"sales": [10.0, np.nan, 5.0]})
    df_v = validate_sales(df)
    assert df_v["sales_is_missing"].iloc[1] == True
    assert df_v["sales_is_missing"].iloc[0] == False


def test_validate_sales_replaces_negative_with_nan():
    df = pd.DataFrame({"sales": [10.0, -3.0, 5.0]})
    df_v = validate_sales(df)
    assert np.isnan(df_v["sales"].iloc[1])
    assert df_v["sales_is_missing"].iloc[1] == True


def test_validate_sales_does_not_replace_zeros():
    """Zero sales are VALID observations — do not replace with NaN."""
    df = pd.DataFrame({"sales": [0.0, 5.0, 10.0]})
    df_v = validate_sales(df)
    assert df_v["sales"].iloc[0] == 0.0
    assert df_v["sales_is_missing"].iloc[0] == False


# ── Chronological ordering ─────────────────────────────────────────────────────

def test_sample_data_is_sorted_chronologically(sample_df):
    """The sample generator should produce sorted data."""
    for (store, fam), grp in sample_df.groupby(["store_nbr", "family"]):
        assert grp["date"].is_monotonic_increasing, \
            f"Data not sorted for store={store}, family={fam}"


# ── Filter subset ─────────────────────────────────────────────────────────────

def test_filter_subset_respects_max_stores(sample_df):
    filtered = filter_subset(sample_df, max_stores=1, max_families=None)
    assert filtered["store_nbr"].nunique() == 1


def test_filter_subset_respects_max_families(sample_df):
    filtered = filter_subset(sample_df, max_stores=None, max_families=1)
    assert filtered["family"].nunique() == 1


# ── Dataset summary ────────────────────────────────────────────────────────────

def test_dataset_summary_returns_correct_types(sample_df):
    df_v = validate_sales(sample_df)
    summary = dataset_summary(df_v)
    assert isinstance(summary["rows"], int)
    assert isinstance(summary["stores"], int)
    assert isinstance(summary["families"], int)
    assert isinstance(summary["missing_sales_pct"], float)
