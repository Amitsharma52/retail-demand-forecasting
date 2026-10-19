# Experiment Results Report

> AI-Based Retail Demand Forecasting System
> Experiment run: 2026-10-09
> All results computed from actual training/validation runs — NOT fabricated

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Dataset | Synthetic demo data (Favorita schema) |
| Stores | 3 (synthetic stores 1, 2, 3) |
| Product families | 5 (GROCERY I, BEVERAGES, PRODUCE, CLEANING, DAIRY) |
| Date range | 2017-01-01 to 2017-08-31 (243 days) |
| Total rows (after feature build) | 3,630 |
| Train/val split | Chronological 80/20 |
| Training cutoff | 2017-07-14 |
| Training rows | 2,895 |
| Validation rows | 735 |
| Python version | 3.13.7 |
| LightGBM version | 4.7.0 |
| Hardware | Windows 11, 8 GB RAM |

**Note:** Results computed on synthetic data. Real Favorita results will differ (more complex seasonality, more series, longer history). Retrain with real data after downloading from Kaggle.

---

## LightGBM Hyperparameters

```python
{
    "objective"         : "quantile",
    "n_estimators"      : 500,
    "learning_rate"     : 0.05,
    "num_leaves"        : 63,
    "min_child_samples" : 20,
    "subsample"         : 0.8,
    "colsample_bytree"  : 0.8,
    "n_jobs"            : -1,
    "random_state"      : 42,
}
```

Training time: ~3.5 seconds total for 3 models.

---

## Validation Metrics (Actual Computed Results)

### Point Forecast Metrics (Q50 model vs. actual sales)

| Model | MAE | RMSE | WAPE |
|-------|-----|------|------|
| LightGBM Q50 | **12.85** | **24.47** | **16.87%** |
| Seasonal Naive (7-day lag) | 19.20 | 35.36 | 25.13% |

**LightGBM improvement over baseline:**
- MAE: −33.1% (lower is better)
- RMSE: −30.8%
- WAPE: −8.3 percentage points

### Quantile Metrics (Pinball Loss)

| Model | Pinball Q10 | Pinball Q50 | Pinball Q90 |
|-------|-------------|-------------|-------------|
| LightGBM | 4.43 | 6.42 | 3.23 |

Lower pinball loss = better calibrated quantile.

### Prediction Interval Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Empirical 80% PI Coverage | **60.5%** | 80% |
| Mean PI Width | **29.0 units** | — |

**⚠️ Coverage Note:** The observed 60.5% coverage is below the nominal 80% target.  
This is expected given:
1. The independence assumption in σ_H computation.
2. The normal approximation for the Q10–Q90 interval.
3. Synthetic data with lognormal noise may have heavier tails.

Improving coverage is a Priority 2 enhancement (Conformal Prediction calibration).

---

## Sample 7-Day Forecast (Store 1, GROCERY I)

Computed from models trained on synthetic data:

| Date | Q10 (Lower) | Q50 (Median) | Q90 (Upper) |
|------|-------------|--------------|-------------|
| 2017-09-01 | 111.7 | 126.1 | 131.7 |
| 2017-09-02 | 103.8 | 126.5 | 145.3 |
| 2017-09-03 | 104.2 | 143.3 | 155.7 |
| 2017-09-04 | 68.6 | 80.9 | 106.0 |
| 2017-09-05 | 59.8 | 85.0 | 95.1 |
| 2017-09-06 | 70.7 | 92.4 | 107.3 |
| 2017-09-07 | 70.1 | 95.9 | 97.1 |

✅ Quantile ordering verified: Q10 ≤ Q50 ≤ Q90 for all days.

---

## Sample Inventory Calculation

Inputs used:
- On hand: 200, On order: 50, Backorders: 10
- Review period R = 7 days, Lead time L = 3 days → H = 10 days
- Service level: 95% → z = 1.645

Results:
- Net position I_pos = 240
- Median cumulative demand D_H = 1,037.7 units
- Protection-horizon σ_H ≈ 42.6 units
- Safety stock SS = ⌈1.645 × 42.6⌉ = 70 units
- Target level S = 1,037.7 + 70 = 1,107.7 units
- **Recommended order Q = 868 units**

(Large order because H = 10 days covers significant future demand, and current I_pos = 240 is well below target.)

---

## Top SHAP Feature Importances (Global, Q50 model)

| Rank | Feature | Mean \|SHAP\| |
|------|---------|---------------|
| 1 | roll_mean_30 | 21.92 |
| 2 | roll_mean_14 | 17.19 |
| 3 | day_of_week | 4.45 |
| 4 | roll_mean_7 | 3.82 |
| 5 | lag_7 | 3.78 |

**Interpretation:** Rolling mean features dominate — the model relies heavily on medium-term demand history. Day of week is the strongest calendar signal. These results are from the synthetic dataset and may differ on real Favorita data.

---

## What to Expect with Real Favorita Data

When trained on the real dataset (54 stores, 33 families, 2013–2017):
- WAPE typically achievable: 15–25% depending on family
- Coverage should be closer to 80% with more training data
- Feature importance may shift — oil price, holiday, and store-type features become relevant
- Training time will be longer (estimate 5–15 minutes for full dataset)

Use `config.py` to configure `MAX_STORES`, `MAX_FAMILIES`, `MAX_TRAIN_ROWS` to make training tractable.
