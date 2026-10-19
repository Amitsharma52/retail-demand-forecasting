# Demo Script — AI Retail Demand Forecasting System

> **For use during your 13 October 2026 project presentation**
> Practice this walkthrough at least twice before the demo.

---

## Before the Demo (5 minutes before)

```powershell
# Navigate to the project
cd C:\Users\sharm\.gemini\antigravity\scratch\retail-demand-system

# Start the dashboard
streamlit run app.py
```

Open http://localhost:8501 in Chrome. Make the browser window fullscreen.

**Confirm:** The yellow banner at the top says "⚠️ DEMO MODE — All data shown is SYNTHETIC".  
If you have downloaded the Kaggle dataset, the banner will not appear — this means real data is loaded.

---

## Demo Walkthrough (~15 minutes)

### 1. Introduction (1 minute)

> "This project addresses a core problem in retail operations: how should a store manager decide how much inventory to order, given uncertain future demand?
>
> Our system answers four questions automatically:
> - What will demand be over the next 7 days?
> - How uncertain is that forecast?
> - How much should we order given our current stock and supplier lead time?
> - Why did the AI make that recommendation?"

### 2. Overview Page (2 minutes)

Navigate to **🏠 Overview**.

Point out:
- **Data Source indicator** — "Real" or "SYNTHETIC DEMO"
- **Dataset statistics** — rows, stores, families, date range
- **Model Status** — green tick if models are trained

> "The system either uses the Kaggle Corporación Favorita dataset — a real Ecuadorian grocery chain — or, if it's not downloaded, it uses clearly labelled synthetic demo data."

If models are not trained → click **🚀 Train Models Now**.

> "Training takes about 60 seconds on this machine. It trains three LightGBM models — one for the 10th percentile, one for the median, and one for the 90th percentile of demand."

### 3. Demand Analysis (2 minutes)

Navigate to **📈 Demand Analysis**.

- Select **Store 1** and **GROCERY I**
- Show the historical sales trend chart
- Point out the orange diamond markers: "These are days where a promotion was active"
- Show the family comparison bar chart: "GROCERY I has the highest total volume"
- Show the promotion vs. no-promotion comparison

> "Before we build any model, we need to understand our data. This shows us seasonality, promotion effects, and data quality."

### 4. Forecasting (4 minutes)

Navigate to **🔮 Forecasting**.

Select:
- Store: **1**
- Product Family: **GROCERY I**
- Horizon: **7 days**

Point out the chart:
- **Blue line** = last 60 days of actual sales
- **Red dashed line** = our LightGBM median forecast
- **Orange shaded band** = 80% prediction interval (Q10 to Q90)
- **Green dotted line** = seasonal naive baseline (just repeating last week's sales)

> "The orange band says: we're 80% confident the true demand will fall in this range. The wider the band, the more uncertain the forecast."

Show the **Forecast Values table** below the chart.

Show the **Validation Metrics table**:
> "On held-out historical data — data the model never saw during training — LightGBM achieves a WAPE of about 17%, compared to the naive baseline's 25%. That's a meaningful improvement."

### 5. Inventory Optimization (4 minutes)

Navigate to **📦 Inventory Optimization**.

Set these example inputs (explain each one):
- **Store**: 1 | **Family**: GROCERY I
- **Stock On Hand**: 150 units
- **Units On Order**: 20 units
- **Backorders**: 5 units
- **Review Period**: 7 days
- **Lead Time**: 3 days
- **Service Level**: 95%

> "The service level means: we want a 95% probability of not running out of stock during the protection horizon."

Walk through the **Step-by-Step Calculation**:

> "Step 1: Net position = 150 + 20 − 5 = 165 units currently available.
>
> Step 2: Protection horizon = 7 + 3 = 10 days. That's the window we must cover.
>
> Step 3: The median forecast predicts [X] units over those 10 days.
>
> Step 4: The uncertainty estimate gives us σ_H = [X] units standard deviation.
>
> Step 5: For 95% service level, z = 1.645, so safety stock = ⌈1.645 × [X]⌉ = [X] units.
>
> Step 6: Target level = median demand + safety stock.
>
> Step 7: Recommended order = max(0, target − current position) = **[X] units**."

Change the service level to **99%** → show that safety stock increases.

> "Higher service level = more safety stock = higher holding cost. This is the classic service-level tradeoff."

### 6. Explainability (2 minutes)

Navigate to **🧠 Explainability**.

Show the **Global Feature Importance** chart:
> "These are the most important features for the model overall. Lag features dominate — meaning recent sales history is the strongest signal."

Show the **Local SHAP** chart:
> "For this specific prediction — Store 1, GROCERY I, tomorrow — these are the features that pushed the forecast up (red) or down (blue) from the average."

Show the **AI Decision Rationale**:
> "The system generates a plain-language explanation tracing from the forecast, through uncertainty, to the final order recommendation. This is critical for trust — a manager should understand *why* the system recommends what it does."

### 7. Methodology (30 seconds)

Navigate to **📖 Methodology & Limitations**.

> "We're transparent about assumptions. The safety stock formula approximates uncertainty from marginal quantiles — we document this. Stockout correction requires inventory data we don't have in this dataset — we document that too. Academic honesty about limitations is as important as the results."

---

## Likely Questions from Your Supervisor

**Q: Why LightGBM and not a neural network?**
> LightGBM is well-suited for tabular data with interpretable features. It trains in seconds, supports quantile regression natively, and works with SHAP explanations. Neural approaches like LSTM are a Phase 2 enhancement.

**Q: How do you prevent data leakage?**
> All lag and rolling features are computed with `.shift(1)` before the rolling window, so today's sales never enter any feature for today's prediction. I can show the exact code in `src/features.py`.

**Q: What is the 80% prediction interval?**
> The Q10 and Q90 models target the 10th and 90th percentiles of demand. The gap between them is a *nominal* 80% interval. The validation shows our actual empirical coverage — which we display honestly, even if it differs from 80%.

**Q: Why not ARIMA or Prophet?**
> The Favorita dataset has 54 stores × 33 product families = 1,782 series. ARIMA requires fitting per series; LightGBM learns across all series jointly and captures cross-series patterns through store and family features.

**Q: What is SHAP?**
> SHapley Additive exPlanations — a mathematically principled method for attributing a prediction to its input features. TreeSHAP computes exact values for tree models in polynomial time. I emphasise that SHAP explains *model behaviour*, not causality.

---

## Emergency Recovery

| Problem | Fix |
|---------|-----|
| `streamlit: not found` | `pip install streamlit` then restart |
| Models not trained | Click "Train Models Now" on Overview page |
| Chart not loading | Refresh browser (F5) |
| Dashboard crashes | `streamlit run app.py` in terminal again |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
