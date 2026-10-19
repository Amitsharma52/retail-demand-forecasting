"""
src/sample_data.py — Synthetic demo data generator.

PURPOSE
-------
When the Kaggle Favorita dataset has not been downloaded, this module produces
a realistic-looking but entirely SYNTHETIC dataset.  The generated data is
clearly labelled in both code and UI so that it is never confused with real
observations.

VISIBILITY RULE
---------------
Every function that returns synthetic data sets  df.attrs["data_source"] = "synthetic"
The dashboard must display a visible banner whenever this attribute is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

RNG = np.random.default_rng(seed=42)


def _weekly_seasonality(day_of_week: np.ndarray) -> np.ndarray:
    """Higher demand on weekends (0=Mon…6=Sun)."""
    weights = np.array([0.85, 0.80, 0.90, 0.95, 1.05, 1.25, 1.20])
    return weights[day_of_week % 7]


def _monthly_trend(month: np.ndarray) -> np.ndarray:
    """Slight Q4 uptick."""
    weights = np.array([0.90, 0.88, 0.92, 0.94, 0.96, 1.00,
                        1.02, 1.05, 1.03, 1.08, 1.12, 1.15])
    return weights[(month - 1) % 12]


def generate_sample_train(
    stores: list[int] | None = None,
    families: list[str] | None = None,
    start: str = config.SAMPLE_START,
    end: str = config.SAMPLE_END,
) -> pd.DataFrame:
    """
    Generate synthetic daily sales data mimicking the Favorita schema.

    Each store × family combination receives:
    * A base demand level drawn from a log-normal distribution.
    * Weekly and monthly multiplicative seasonality.
    * Random promotions (~15 % of days) that boost sales by 10–30 %.
    * Gaussian noise.
    * Occasional zero-sale days (~5 %).

    Returns
    -------
    pd.DataFrame
        Columns: date, store_nbr, family, sales, onpromotion.
        Attribute ``data_source`` == "synthetic".
    """
    stores   = stores   or config.SAMPLE_STORES
    families = families or config.SAMPLE_FAMILIES

    dates = pd.date_range(start=start, end=end, freq="D")
    rows: list[dict] = []

    family_bases = {f: float(RNG.lognormal(mean=4.0, sigma=0.8)) for f in families}

    for store in stores:
        store_scale = float(RNG.uniform(0.7, 1.4))
        for family in families:
            base = family_bases[family] * store_scale
            for date in dates:
                dow   = date.dayofweek
                month = date.month
                promo = int(RNG.random() < 0.15)
                promo_boost = 1 + RNG.uniform(0.10, 0.30) * promo

                demand = (
                    base
                    * _weekly_seasonality(np.array([dow]))[0]
                    * _monthly_trend(np.array([month]))[0]
                    * promo_boost
                    * float(RNG.lognormal(mean=0, sigma=0.15))   # noise
                )

                # ~5 % zero-sale days
                if RNG.random() < 0.05:
                    demand = 0.0

                rows.append({
                    "date"       : date,
                    "store_nbr"  : store,
                    "family"     : family,
                    "sales"      : round(max(0.0, demand), 2),
                    "onpromotion": promo,
                })

    df = pd.DataFrame(rows).sort_values(["store_nbr", "family", "date"])
    df = df.reset_index(drop=True)

    # ⚠ Mark as synthetic — checked by every downstream module and the dashboard
    df.attrs["data_source"] = "synthetic"
    df["is_holiday"] = 0   # simplified — no real holiday calendar
    df["sales_is_missing"] = False

    return df


def get_data_source_label(df: pd.DataFrame) -> str:
    """Return 'real' or 'synthetic' based on the DataFrame attribute."""
    return df.attrs.get("data_source", "unknown")
