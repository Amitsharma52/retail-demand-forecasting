# AI-Based Retail Demand Forecasting, Inventory Optimization, and Explainable AI Decision-Support System

> **B.Tech Computer Science and Engineering Academic Project**
> Dataset: [Corporación Favorita Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

---

## Problem Statement

Retailers face two connected challenges:

1. **Demand uncertainty**: Product sales fluctuate due to seasonality, promotions, and external events.  Ordering too much wastes working capital; ordering too little causes stockouts and lost revenue.
2. **Decision opacity**: When an AI system recommends an order, a manager must be able to understand *why* to trust and act on it.

This project builds an end-to-end decision-support system that forecasts demand, estimates uncertainty, recommends inventory replenishment, and explains every recommendation in plain language.

---

## Features

| Feature | Status |
|---------|--------|
| Real data ingestion (Favorita dataset) | ✅ |
| Synthetic demo mode (no download required) | ✅ |
| Leakage-free feature engineering | ✅ |
| Seasonal-naive baseline model | ✅ |
| LightGBM Q10/Q50/Q90 quantile models | ✅ |
| Chronological train/validation split | ✅ |
| MAE / RMSE / WAPE / Pinball / Coverage metrics | ✅ |
| 7-day recursive forecasting | ✅ |
| Prediction interval with crossing correction | ✅ |
| Inventory replenishment calculator | ✅ |
| SHAP global and local feature importance | ✅ |
| Plain-language decision rationale | ✅ |
| Streamlit dashboard (6 sections) | ✅ |
| Automated test suite | ✅ |

---

## Technology Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.13 | Runtime |
| pandas | 2.3.2 | Data manipulation |
| NumPy | 2.3.3 | Numerical computing |
| LightGBM | 4.7.0 | Quantile forecasting |
| scikit-learn | 1.7.2 | Evaluation utilities |
| SHAP | 0.53.0 | Model explanations |
| Streamlit | 1.65.0 | Web dashboard |
| Plotly | 7.1.0 | Interactive charts |
| SciPy | 1.16.2 | Statistics (safety stock) |
| Joblib | 1.5.2 | Model serialisation |
| PyArrow | 25.0.1 | Parquet I/O |
| pytest | 9.1.1 | Automated tests |

---

## Dataset

**Source:** [Kaggle — Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

**Publisher:** Corporación Favorita (Ecuadorian grocery chain)

**Files used:**

| File | Required | Description |
|------|----------|-------------|
| `train.csv` | ✅ Required | Daily sales by store × product family |
| `stores.csv` | Recommended | Store metadata (city, state, type) |
| `holidays_events.csv` | Recommended | Ecuadorian public holidays |
| `oil.csv` | Optional | Daily Brent crude oil price |
| `transactions.csv` | Optional | Daily transaction count |

### Download Instructions

1. Create a free [Kaggle account](https://www.kaggle.com)
2. Accept the competition rules on the [dataset page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
3. Install the Kaggle CLI: `pip install kaggle`
4. Get your API key: [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token → download `kaggle.json`
5. Place `kaggle.json` in `C:\Users\<you>\.kaggle\`
6. Run:

```powershell
kaggle competitions download -c store-sales-time-series-forecasting
Expand-Archive store-sales-time-series-forecasting.zip -DestinationPath data\raw\
```

7. Restart the app

> **Note:** `data/raw/` is in `.gitignore` and will NOT be committed to Git.

---

## Installation

```powershell
# 1. Clone or navigate to the project
cd retail-demand-system

# 2. (Recommended) create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) place Kaggle data in data/raw/
#    If not present, the app automatically uses synthetic demo data.
```

---

## Project Structure

```
retail-demand-system/
├── app.py                    ← Streamlit dashboard (entry point)
├── config.py                 ← Central configuration
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/                  ← Kaggle CSV files (not in Git)
│   ├── processed/            ← Cached parquet files (not in Git)
│   └── sample/               ← Generated sample data
│
├── models/                   ← Saved model artefacts (not in Git)
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py      ← Module A: Data ingestion & preprocessing
│   ├── sample_data.py        ← Synthetic demo data generator
│   ├── features.py           ← Module B: Feature engineering
│   ├── train_model.py        ← Module C/I: Model training & artefacts
│   ├── forecasting.py        ← Module C: Forecast generation
│   ├── evaluation.py         ← Module D: Metrics
│   ├── inventory.py          ← Module F: Inventory calculator
│   └── explainability.py     ← Module G: SHAP + rationale
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_features.py
│   ├── test_inventory.py
│   └── test_evaluation.py
│
├── reports/
│   ├── methodology.md
│   ├── experiment_results.md
│   ├── development_status.md
│   └── demo_script.md
│
└── notebooks/
    └── exploration.ipynb     ← EDA (optional)
```

---

## Commands

### Train the Models

```powershell
# From the project root directory:
python -c "
import sys, logging
logging.basicConfig(level=logging.INFO)
from src.sample_data import generate_sample_train
from src.train_model import run_training_pipeline
df = generate_sample_train()
result = run_training_pipeline(df)
print('Metrics:', result['metrics'])
"
```

Or click **Train Models Now** on the Overview page of the dashboard.

### Start the Dashboard

```powershell
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Run Tests

```powershell
python -m pytest tests/ -v
```

---

## Model Architecture

### Feature Engineering (Module B — `src/features.py`)

All features are computed per `(store_nbr, family)` group and shifted by at least
1 day to prevent target leakage:

| Feature | Description | Anti-leakage |
|---------|-------------|--------------|
| `lag_1/7/14/30` | Sales N days ago | `.shift(N)` |
| `roll_mean_7/14/30` | Rolling average of past sales | `.shift(1).rolling(W).mean()` |
| `roll_std_7/14/30` | Rolling std of past sales | `.shift(1).rolling(W).std()` |
| `ewma_7` | Exponentially weighted moving average | `.shift(1).ewm(span=7)` |
| `dow_sin/cos` | Cyclic day-of-week encoding | No shift needed |
| `month_sin/cos` | Cyclic month encoding | No shift needed |
| `is_on_promotion` | Binary promotion flag | Promotion schedule known |
| `promo_roll_7` | Rolling 7-day promotion count | `.shift(1).rolling(7).sum()` |

### Quantile Regression (Module C — `src/train_model.py`)

Three separate LightGBM models are trained:

```
Objective: quantile (pinball loss)
Q10 model: alpha = 0.10  → lower demand bound
Q50 model: alpha = 0.50  → central forecast
Q90 model: alpha = 0.90  → upper demand bound
```

**Crossing correction** ensures Q10 ≤ Q50 ≤ Q90 at prediction time.

---

## Uncertainty and Inventory Equations

See `reports/methodology.md` for full derivation.

**Key equations:**

```
σ_t  = (Q90_t - Q10_t) / 2.564         # daily uncertainty
σ_H  = sqrt(Σ σ_t²)                    # protection-horizon uncertainty
SS   = ceil(z_p × σ_H)                 # safety stock
S    = D_H + SS                         # target order-up-to level
Q    = max(0, ceil(S - I_pos))          # recommended order quantity
```

---

## Model Evaluation

- **Split:** Chronological holdout (no random split)
- **Training:** First 80% of the timeline
- **Validation:** Last 20% of the timeline
- **Metrics:** MAE, RMSE, WAPE, Pinball loss (Q10/Q50/Q90), 80% PI Coverage

---

## SHAP Methodology

SHAP (SHapley Additive exPlanations) using `shap.TreeExplainer` on the Q50 model.

> **Important:** SHAP values explain model *behaviour* — they show which features
> contributed to a specific prediction relative to the training average.
> They are NOT proof of causal relationships.

---

## Data Limitations

1. **Stockout correction** — The Favorita dataset does not contain on-hand inventory
   records. Censored demand recovery is therefore NOT implemented.
   The development status report tracks this as a Phase 2 feature.

2. **Independence assumption** — The safety stock calculation assumes daily demand
   errors are independent. This is documented and is a known simplification.

3. **Promotion data** — Future promotion schedules are known for the Favorita
   test set but unknown for arbitrary future dates. Zero promotion is assumed
   when this information is unavailable.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: train.csv` | Download the Kaggle dataset — or use the synthetic demo mode |
| `streamlit: command not found` | `pip install streamlit` |
| Dashboard re-trains on every refresh | Models are cached via `@st.cache_resource` — check model files in `models/` |
| SHAP not available | `pip install shap` |
| `ModuleNotFoundError: lightgbm` | `pip install lightgbm` |

---

## Future Improvements (Phase 2)

1. Walk-forward cross-validation for more robust evaluation
2. Conformal prediction for better-calibrated intervals
3. XGBoost and CatBoost model comparison
4. Stockout / censored demand correction interface
5. Historical inventory policy simulation
6. FastAPI REST endpoint for production serving
