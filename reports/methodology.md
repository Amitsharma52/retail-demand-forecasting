# Technical Methodology Report

> AI-Based Retail Demand Forecasting, Inventory Optimization, and Explainable AI
> B.Tech CSE Academic Project

---

## 1. Data Pipeline

### 1.1 Input Data Schema

The Corporación Favorita dataset provides daily sales at the store × product family level.

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Calendar date |
| `store_nbr` | int | Store identifier (1–54) |
| `family` | string | Product family (33 categories) |
| `sales` | float | Daily unit sales |
| `onpromotion` | int | Units on promotional display |

### 1.2 Preprocessing Steps

1. Load CSV with `parse_dates=["date"]`
2. Validate required columns
3. Convert date strings to `datetime64`
4. Drop exact duplicate `(date, store_nbr, family)` rows (keep last)
5. Flag and null-out negative sales; retain genuine zeros
6. Merge optional files: stores, holidays, oil price
7. Filter to configurable subset: first N stores/families by total sales volume
8. Sort chronologically by `(store_nbr, family, date)`
9. Apply row cap for memory management

### 1.3 Missing Values

- Missing `sales` rows receive a `sales_is_missing` flag and are **not** auto-filled with zero.
- Zero sales are treated as genuine observations.
- Oil price gaps are filled with linear interpolation.

---

## 2. Feature Engineering

### 2.1 Anti-Leakage Strategy

All time-series features are computed per `(store_nbr, family)` group. The key rule:

> **No feature for row t may contain any information from time ≥ t.**

Implementation: `.shift(n)` within each group ensures this. Rolling statistics are computed on the **shifted** series so that today's sales value never enters the window.

### 2.2 Feature Set

#### Calendar Features
```
day_of_week ∈ {0,…,6}     # Monday=0
month ∈ {1,…,12}
day_of_month ∈ {1,…,31}
week_of_year ∈ {1,…,53}
quarter ∈ {1,…,4}

# Cyclic encoding (prevents discontinuity at day 6→0 and Dec→Jan)
dow_sin   = sin(2π × day_of_week / 7)
dow_cos   = cos(2π × day_of_week / 7)
month_sin = sin(2π × (month-1) / 12)
month_cos = cos(2π × (month-1) / 12)
```

#### Lag Features (per store×family group)
```
lag_1  = sales(t-1)
lag_7  = sales(t-7)
lag_14 = sales(t-14)
lag_30 = sales(t-30)
```

#### Rolling Features (on shifted series)
```
roll_mean_7  = mean(sales[t-7 … t-1])
roll_mean_14 = mean(sales[t-14 … t-1])
roll_mean_30 = mean(sales[t-30 … t-1])
roll_std_7   = std(sales[t-7 … t-1])
roll_std_14  = std(sales[t-14 … t-1])
roll_std_30  = std(sales[t-30 … t-1])
ewma_7       = EWM(sales[…t-1], span=7)
```

#### Promotion Features
```
is_on_promotion = 1 if onpromotion > 0 else 0
promo_roll_7    = sum(is_on_promotion[t-7 … t-1])
```

### 2.3 Future Feature Generation

For recursive multi-step forecasting (days 1…H):
1. Build the combined history + future skeleton frame.
2. Compute all features using only the available past.
3. After predicting day t, append Q50 as synthetic sales for day t's position.
4. Predicted rows are flagged `is_predicted=True`.

---

## 3. Forecasting Models

### 3.1 Seasonal Naive Baseline

```
baseline(t) = sales(t - 7)
```

Serves as a lower-bound benchmark. No learning — uses only the 7-day lag.

### 3.2 LightGBM Quantile Regression

Three separate models are trained using LightGBM's `quantile` objective, which minimises the **pinball loss**:

```
L_α(y, q) = α·max(y - q, 0) + (1-α)·max(q - y, 0)
```

| Model | α | Objective | Purpose |
|-------|---|-----------|---------|
| Q10   | 0.10 | `quantile` | Lower demand bound |
| Q50   | 0.50 | `quantile` | Central forecast |
| Q90   | 0.90 | `quantile` | Upper demand bound |

### 3.3 Quantile Crossing Correction

At prediction time, values are sorted to enforce Q10 ≤ Q50 ≤ Q90:
```python
stacked = np.sort(np.stack([q10, q50, q90], axis=1), axis=1)
```
This is a safeguard, not a substitute for calibration evaluation.

---

## 4. Model Evaluation

### 4.1 Chronological Split

```
Timeline: ──────────────────────────────────────────────
          [   Training (80%)   ][  Validation (20%)   ]
                               ↑
                          Split date (no shuffle)
```

### 4.2 Metric Definitions

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | `mean(|y - ŷ|)` | Point forecast error |
| RMSE | `sqrt(mean((y - ŷ)²))` | Penalises large errors |
| WAPE | `Σ|y - ŷ| / Σ|y|` | Returns "undefined" when Σy=0 |
| Pinball(α) | `mean(α·max(y-q,0) + (1-α)·max(q-y,0))` | Quantile loss |
| Coverage | `mean(q10 ≤ y ≤ q90)` | Empirical PI coverage |

---

## 5. Uncertainty Estimation

### 5.1 Daily Uncertainty Approximation

Assuming the Q10–Q90 interval approximates an 80% normal prediction interval:

$$P(\mu - 1.2816\sigma \leq Y \leq \mu + 1.2816\sigma) \approx 0.80$$

Therefore:
$$\sigma_t \approx \frac{Q_{90,t} - Q_{10,t}}{2 \times 1.2816}$$

### 5.2 Protection-Horizon Uncertainty

Under the assumption that daily demand errors are **independent**:

$$\sigma_H = \sqrt{\sum_{t=1}^{H} \sigma_t^2}$$

**Documented limitation:** This independence assumption is a simplification. In practice, demand errors may be correlated across days (e.g., due to persistent weather or supply disruptions). A multi-horizon calibrated quantile model would be more accurate (planned Phase 2).

---

## 6. Inventory Optimization

### 6.1 Variables

| Symbol | Meaning |
|--------|---------|
| `I_pos` | Net inventory position |
| `H` | Protection horizon (days) |
| `R` | Inventory review period (days) |
| `L` | Supplier lead time (days) |
| `D_H` | Median cumulative demand over H days |
| `σ_H` | Protection-horizon demand uncertainty |
| `z_p` | Standard normal quantile for service level p |
| `SS` | Safety stock |
| `S` | Target order-up-to level |
| `Q` | Recommended order quantity |

### 6.2 Equations

$$I_{pos} = \text{On-hand} + \text{On-order} - \text{Backorders}$$

$$H = R + L$$

$$D_H = \sum_{t=1}^{H} Q_{50,t}$$

$$\sigma_t = \frac{Q_{90,t} - Q_{10,t}}{2 \times 1.2816}$$

$$\sigma_H = \sqrt{\sum_{t=1}^{H} \sigma_t^2}$$

$$z_p = \Phi^{-1}(p) \quad \text{(standard normal quantile)}$$

$$SS = \max\left(0,\ \left\lceil z_p \cdot \sigma_H \right\rceil\right)$$

$$S = D_H + SS$$

$$Q = \max\left(0,\ \left\lceil S - I_{pos} \right\rceil\right)$$

### 6.3 Service Level Examples

| Service Level p | z_p |
|----------------|-----|
| 90% | 1.282 |
| 95% | 1.645 |
| 99% | 2.326 |

### 6.4 Limitations

- These equations model a **cycle service level** (probability of no stockout in one review cycle), not an OTIF (On-Time In-Full) metric.
- OTIF requires actual fulfilment records that are not in the Favorita dataset.

---

## 7. Explainable AI (SHAP)

### 7.1 Method

`shap.TreeExplainer` is applied to the fitted LightGBM Q50 model. TreeSHAP computes exact Shapley values in polynomial time for tree ensembles.

### 7.2 Interpretation

For each prediction:
- **Global importance**: mean(|SHAP|) across training predictions — features that consistently affect the model.
- **Local explanation**: SHAP values for one specific prediction — shows which features pushed this prediction up or down from the baseline.

### 7.3 Caveats

> SHAP values explain **model behaviour**, not causal relationships. A high SHAP value for `lag_7` means the model relied heavily on this feature for this prediction — it does not prove that last week's sales *caused* this week's sales.
