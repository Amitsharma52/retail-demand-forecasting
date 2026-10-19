# Development Status Report

> AI-Based Retail Demand Forecasting System
> Last updated: 2026-10-09

---

## ✅ Completed (MVP — Priority 1)

### Module A — Data Ingestion & Preprocessing
- [x] CSV loading with date parsing
- [x] Required column validation
- [x] Duplicate row detection and removal
- [x] Negative sales flagging (replaced with NaN, not silently dropped)
- [x] Zero sales preserved as valid observations
- [x] Optional metadata merges: stores, holidays, oil price
- [x] Configurable subset filtering (MAX_STORES, MAX_FAMILIES)
- [x] Chronological sort enforced
- [x] Parquet caching of processed data
- [x] Dataset summary statistics

### Module A (synthetic) — Demo Data Generator
- [x] Synthetic data generator with realistic seasonality and promotions
- [x] Synthetic data clearly labelled with `df.attrs["data_source"] = "synthetic"`
- [x] Dashboard shows visible warning banner when demo data is active

### Module B — Feature Engineering
- [x] Lag features: lag_1, lag_7, lag_14, lag_30
- [x] Rolling mean: 7, 14, 30 days (applied to shifted series — leakage-safe)
- [x] Rolling std: 7, 14, 30 days (leakage-safe)
- [x] EWMA with span=7 (leakage-safe)
- [x] Calendar features: day_of_week, month, day_of_month, week_of_year, quarter
- [x] Cyclic encoding: sin/cos for day-of-week and month
- [x] Promotion features: is_on_promotion, promo_roll_7
- [x] Store and family integer encoding (consistent via saved maps in metadata)
- [x] Future feature builder for recursive multi-step forecasting

### Module C — Forecasting Engine
- [x] Seasonal naive baseline (7-day lag)
- [x] LightGBM Q10 quantile model (alpha=0.10)
- [x] LightGBM Q50 quantile model (alpha=0.50)
- [x] LightGBM Q90 quantile model (alpha=0.90)
- [x] Quantile crossing correction at prediction time
- [x] 7-day recursive ahead forecasting
- [x] Model save/load with Joblib

### Module D — Evaluation
- [x] Chronological train/validation split (no random shuffle)
- [x] MAE metric
- [x] RMSE metric
- [x] WAPE metric (returns "undefined" for zero-demand series)
- [x] Pinball loss for Q10, Q50, Q90
- [x] Empirical coverage of [Q10, Q90] interval
- [x] Baseline vs LightGBM comparison

### Module E — Uncertainty
- [x] Daily σ_t from Q10/Q90 interval with documented approximation
- [x] Protection-horizon σ_H under independence assumption
- [x] Safety stock formula with z_p from scipy.stats.norm.ppf

### Module F — Inventory Optimization
- [x] Net inventory position calculation
- [x] Protection horizon H = R + L
- [x] Cumulative median demand D_H
- [x] Safety stock SS
- [x] Target order-up-to level S
- [x] Recommended order quantity Q
- [x] Step-by-step plain-language explanation
- [x] Assumptions list in UI and documentation
- [x] Zero-order explanation when inventory already sufficient

### Module G — Explainability
- [x] SHAP TreeExplainer with LightGBM Q50 model
- [x] Global feature importance (mean |SHAP|)
- [x] Local SHAP values for individual predictions
- [x] SHAP waterfall bar chart (red = increase, blue = decrease)
- [x] Plain-language rationale generator using actual computed values
- [x] Clear SHAP disclaimer (model behaviour, not causality)
- [x] Graceful degradation if SHAP unavailable

### Module H — Streamlit Dashboard
- [x] Section 1: Overview (data status, model status, Kaggle instructions)
- [x] Section 2: Demand Analysis (historical trend, family comparison, promotions)
- [x] Section 3: Forecasting (interactive chart, forecast table, metrics)
- [x] Section 4: Inventory Optimization (full calculator with all inputs/outputs)
- [x] Section 5: Explainability (SHAP global + local + rationale)
- [x] Section 6: Methodology & Limitations (equations, caveats, assumptions)
- [x] Synthetic data banner on all pages
- [x] Error messages without Python tracebacks shown to users

### Module I — Training Workflow
- [x] One-click training button in Overview page
- [x] Model caching (no retraining on dashboard re-runs)
- [x] Clear instructions when models are missing

### Testing
- [x] 50 automated tests — all passing
- [x] Data pipeline tests (14 tests)
- [x] Feature engineering tests (14 tests)
- [x] Evaluation metric tests (14 tests)
- [x] Inventory calculator tests (9 tests)

### Documentation
- [x] README.md with full installation and usage instructions
- [x] reports/methodology.md — technical pipeline and equations
- [x] reports/development_status.md (this file)
- [x] .gitignore for data, models, environments

---

## ⏳ Pending (Phase 2 — after demonstration)

### Research Enhancements
- [x] Notebook: exploration.ipynb (EDA, feature inspection, model & inventory walkthrough)
- [ ] Walk-forward time-series cross-validation (rolling origin)
- [ ] XGBoost comparison model
- [ ] CatBoost comparison model
- [ ] Conformal prediction intervals (better calibration)
- [ ] Multi-horizon direct calibration of σ_H
- [ ] Historical inventory policy simulation
- [ ] Stockout / censored demand correction interface
  - Blocked by: Favorita dataset lacks on-hand inventory logs
- [ ] FastAPI REST endpoint for production deployment

---

## 🐛 Known Issues / Limitations

1. **Empirical PI coverage**: Validation shows coverage below 80% nominal target. This is expected given the simplicity of the independent-normal approximation and is documented in the methodology.

2. **Recursive forecast propagation**: Predicted values used as inputs for subsequent days may accumulate errors. A direct multi-step model would reduce this.

3. **Integer encoding consistency**: store_nbr and family codes are saved in model metadata. If the dataset changes between training and inference, a retrain is required.

4. **Stockout logs missing**: The Favorita dataset logs observed sales transactions, not historical out-of-stock events. Censored-demand correction is documented as a limitation.

