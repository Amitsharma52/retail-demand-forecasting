# Development Status Report (v2.0 Commercial Edition)

> AI-Based Retail Demand Forecasting, Inventory Optimization & Procurement Microservice  
> Last updated: 2026-11-04  
> Status: **Phase 1 & Phase 2 Complete (Production / Commercial Grade)**

---

## ✅ Phase 1: MVP & Academic Proof-of-Concept (Completed)

### Module A — Data Ingestion & Preprocessing
- [x] CSV loading with robust ISO datetime parsing
- [x] Required column validation (`date`, `store_nbr`, `family`, `sales`, `onpromotion`)
- [x] Duplicate row detection and deduplication
- [x] Negative sales flagging (converted to NaN, not dropped silently)
- [x] Zero sales preserved as genuine zero-demand observations
- [x] Optional metadata merges: stores, holidays, oil price
- [x] Configurable subset filtering (`MAX_STORES`, `MAX_FAMILIES`)
- [x] Chronological sort strictly enforced
- [x] Parquet caching of processed data (`train_processed.parquet`)
- [x] Dataset summary statistics

### Module A (synthetic) — Demo Data Generator
- [x] Synthetic data generator with realistic seasonality and promotions
- [x] Synthetic data clearly labelled with `df.attrs["data_source"] = "synthetic"`
- [x] Dashboard shows visible warning banner when demo data is active

### Module B — Feature Engineering
- [x] Lag features: `lag_1`, `lag_7`, `lag_14`, `lag_30`
- [x] Rolling mean: 7, 14, 30 days (applied to `.shift(1)` — leakage-free)
- [x] Rolling std: 7, 14, 30 days (leakage-free)
- [x] EWMA with span=7 (leakage-free)
- [x] Calendar features: `day_of_week`, `month`, `day_of_month`, `quarter`
- [x] Cyclic encoding: sin/cos for day-of-week and month
- [x] Promotion features: `is_on_promotion`, `promo_roll_7`
- [x] Store and family integer encoding (consistent via saved maps in metadata)
- [x] Future feature builder for recursive multi-step forecasting

### Module C — Forecasting Engine
- [x] Seasonal naive baseline (7-day lag)
- [x] LightGBM Q10 quantile model (alpha=0.10)
- [x] LightGBM Q50 quantile model (alpha=0.50)
- [x] LightGBM Q90 quantile model (alpha=0.90)
- [x] Quantile crossing correction at prediction time ($Q_{10} \le Q_{50} \le Q_{90}$)
- [x] 7-day recursive ahead forecasting
- [x] Model save/load with Joblib

### Module D — Evaluation
- [x] Chronological train/validation split (no random shuffle)
- [x] MAE metric
- [x] RMSE metric
- [x] WAPE metric (returns 0.0 or handles zero-demand series)
- [x] Pinball loss for Q10, Q50, Q90
- [x] Empirical coverage of [Q10, Q90] interval
- [x] Baseline vs LightGBM comparison

### Module E — Uncertainty
- [x] Daily $\sigma_t$ from Q10/Q90 interval with documented approximation
- [x] Protection-horizon $\sigma_H$ under independence assumption
- [x] Safety stock formula with $z_p$ from `scipy.stats.norm.ppf`

### Module F — Inventory Optimization
- [x] Net inventory position calculation: $I_{pos} = \text{On-Hand} + \text{On-Order} - \text{Backorders}$
- [x] Protection horizon $H = R + L$
- [x] Cumulative median demand $D_H$
- [x] Safety stock $SS$
- [x] Target order-up-to level $S$
- [x] Recommended order quantity $Q$
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

---

## ✅ Phase 2: Commercial Viability & Enterprise Extensions (Completed)

### 1. Conformalized Quantile Regression (CQR — `src/conformal.py`)
- [x] Romano et al. (2019) signed non-conformity score engine: $s_i = \max(q_{10,i} - y_i, y_i - q_{90,i})$
- [x] Finite-sample coverage calibration calculation at exact nominal levels (e.g. 80%, 90%)
- [x] Adaptive interval width expansion/contraction without order crossing ($Q_{10} \le Q_{50} \le Q_{90}$)
- [x] Full evaluation metric suite (`raw_coverage`, `calibrated_coverage`, `raw_width`, `calibrated_width`)
- [x] Complete unit testing with 100% pass rate in `tests/test_conformal.py`

### 2. Multi-Model Benchmarking & Walk-Forward CV (`src/benchmarks.py`)
- [x] Expanding-window rolling-origin cross-validation engine (K-fold time-series splits)
- [x] Competitive model ensemble: LightGBM vs. XGBoost (`reg:quantileerror`) vs. Seasonal Naive vs. Rolling Mean
- [x] Multi-metric evaluation matrix: WAPE, MAE, RMSE, Pinball loss, 80% coverage, training duration, and inference latency (ms/sample)
- [x] Integrated unit tests in `tests/test_benchmarks.py`

### 3. Historical Inventory Policy Simulator & Financial Backtesting (`src/simulation.py`)
- [x] Day-by-day discrete-event inventory simulation engine
- [x] Head-to-head comparison of three policies:
  1. StockWise Dynamic Quantile Policy
  2. Classical Static Min-Max $(s, S)$ Policy
  3. 7-Day Moving Average Heuristic Policy
- [x] Comprehensive financial KPI modeling: fill rate %, stockout frequency, holding costs, lost margin penalties, reorder fixed costs, gross revenue, and net operating profit
- [x] Interactive Plotly on-hand inventory trajectory visualization
- [x] Integrated unit tests in `tests/test_simulation.py`

### 4. Commercial Purchase Order (PO) Procurement Engine (`src/po_generator.py`)
- [x] Master case-pack rounding (cases of 6, 12, 24, etc.)
- [x] Minimum Order Quantity (MOQ) enforcement
- [x] Wholesale valuation and delivery ETA scheduling ($T_{delivery} = T_{order} + L$)
- [x] Multi-format export:
  - RFC 4180 CSV export
  - Official commercial ReportLab PDF with corporate headers, line-item grid, totals box, and sign-off block
- [x] Integrated unit tests in `tests/test_po_generator.py`

### 5. Custom Merchant POS Data Ingestion Pipeline (`src/custom_data.py`)
- [x] Merchant CSV and Excel (`.xlsx`) drag-and-drop ingestion
- [x] Fuzzy column matching across diverse retail schemas (Date, SKU, Sales, Store, Promo)
- [x] Daily grid regularization and missing store-open day gap filling
- [x] Automated feature generator (calendar, lags, rolling statistics)
- [x] Data health auditor: span, zero-sales ratio, intermittency, and validation status
- [x] Integrated unit tests in `tests/test_custom_data.py`

### 6. Production REST API Microservice (`api.py`)
- [x] FastAPI production HTTP server with Pydantic v2 schemas
- [x] Production endpoints:
  - `GET /health`
  - `GET /catalog`
  - `POST /forecast` (with optional Conformal Calibration)
  - `POST /replenishment`
  - `POST /purchase-order`
  - `POST /simulate`
- [x] Full integration testing via FastAPI `TestClient` in `tests/test_api.py`

### 7. Commercial Streamlit Command Center (`app.py`)
- [x] 8 specialized operational screens:
  1. Home (At-a-Glance Operational Posture)
  2. Inventory (Stock Register & Days of Supply Left)
  3. Order Suggestions & PO (Commercial Case Rounding, MOQ & PDF/CSV Download)
  4. Policy Simulator (Historical Financial Backtesting & Trajectory Charts)
  5. Custom Store Upload (Merchant POS Drag-and-Drop & Diagnostics)
  6. Sales Trends (Historical Trends & Quantile Intervals)
  7. Advanced Analytics (CQR Conformal, Walk-Forward Benchmarks, SHAP, Formulations)
  8. Settings & REST API Microservice (API docs, cURL examples & Retrain)
- [x] Clean light enterprise styling (`#F7F8F5` background, `#287A50` forest green, `#202820` typography)

---

## 🧪 Test Suite Summary

- **Total Test Cases**: 77
- **Passing**: 77 (100% Pass Rate)
- **Execution Time**: ~6.8 seconds
- **Test Modules**:
  - `tests/test_api.py`: 7 tests (FastAPI microservice endpoints)
  - `tests/test_benchmarks.py`: 4 tests (Multi-model & walk-forward CV)
  - `tests/test_conformal.py`: 5 tests (CQR calibration & nonconformity)
  - `tests/test_custom_data.py`: 3 tests (Merchant POS ingestion & features)
  - `tests/test_data_pipeline.py`: 13 tests (Data hygiene & preprocessing)
  - `tests/test_evaluation.py`: 14 tests (Error metrics & pinball losses)
  - `tests/test_features.py`: 12 tests (Anti-leakage feature engineering)
  - `tests/test_inventory.py`: 9 tests (Replenishment math & safety stock)
  - `tests/test_po_generator.py`: 4 tests (Case rounding, MOQ & ReportLab PDF)
  - `tests/test_simulation.py`: 4 tests (Discrete-event backtesting)
