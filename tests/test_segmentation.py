"""
tests/test_segmentation.py — Unit tests for ABC-XYZ portfolio segmentation.
"""

import numpy as np
import pandas as pd
import pytest

from src.segmentation import (
    SEGMENT_POLICIES,
    classify_abc_xyz,
    generate_segmentation_matrix_summary,
)


def test_classify_abc_xyz():
    # Construct 3 items with distinct volume & volatility profiles:
    # Item 1: High volume, constant sales (AX candidate)
    # Item 2: Medium volume, seasonal variance (BY candidate)
    # Item 3: Tiny volume, erratic spikes (CZ candidate)
    dates = pd.date_range("2026-01-01", periods=30, freq="D")

    records = []
    for d in dates:
        records.append({"date": d, "store_nbr": 1, "family": "STABLE_HIGH", "sales": 100.0})
        records.append({"date": d, "store_nbr": 1, "family": "MED_SEASONAL", "sales": 30.0 + (d.day % 7) * 5})
        records.append({"date": d, "store_nbr": 1, "family": "LOW_ERRATIC", "sales": 1.0 if d.day % 10 == 0 else 0.0})

    df = pd.DataFrame(records)

    seg_df = classify_abc_xyz(df, store_nbr=1)

    assert len(seg_df) == 3
    assert "abc_class" in seg_df.columns
    assert "xyz_class" in seg_df.columns
    assert "segment" in seg_df.columns
    assert "target_service_level" in seg_df.columns

    # STABLE_HIGH should be Class A (largest volume share) and Class X (zero variance)
    stable_row = seg_df[seg_df["family"] == "STABLE_HIGH"].iloc[0]
    assert stable_row["abc_class"] == "A"
    assert stable_row["xyz_class"] == "X"
    assert stable_row["segment"] == "AX"
    assert stable_row["target_service_level"] == 0.98

    # LOW_ERRATIC should be Class C
    erratic_row = seg_df[seg_df["family"] == "LOW_ERRATIC"].iloc[0]
    assert erratic_row["abc_class"] == "C"


def test_segmentation_matrix_summary():
    df_fake = pd.DataFrame({
        "abc_class": ["A", "A", "B", "C"],
        "xyz_class": ["X", "Y", "X", "Z"],
    })
    grid = generate_segmentation_matrix_summary(df_fake)

    assert grid.loc["A", "X"] == 1
    assert grid.loc["A", "Y"] == 1
    assert grid.loc["B", "X"] == 1
    assert grid.loc["Total", "Total"] == 4
