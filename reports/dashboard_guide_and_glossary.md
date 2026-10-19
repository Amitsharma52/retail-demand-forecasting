# Retail AI Decision-Support System: Dashboard Guide & Terminology Handbook

This guide explains how the dashboard works, why each feature exists, and the plain-English meaning of every technical term and formula used in the application.

---

## 1. How the Application Works (High-Level Architecture)

The dashboard is designed as an **End-to-End Decision Support System**. Instead of just predicting numbers, it connects raw data to real business decisions in four steps:

```
[Real Historical Sales Data]
             ↓
[Leakage-Free Feature Engineering] (Lags, Rolling Means, Calendar)
             ↓
[Quantile Machine Learning Engine] (LightGBM Q10, Q50, Q90)
             ↓
[Inventory Replenishment Optimizer] (Safety Stock, Net Position, Target S)
             ↓
[Explainability & Decision Rationale] (SHAP Feature Importance & Plain English Logic)
```

### The 4 Dashboard Tabs

1. **📦 Demand Forecast & Inventory (Main Action Screen)**:
   - You select your store location and product family.
   - You input current operational constraints (current stock, supplier lead time, review cycle).
   - The system immediately displays the **Recommended Order Quantity** in a prominent hero banner, alongside the 7-day quantile forecast chart.
2. **📈 Demand History & Trends (Exploratory Data Analysis)**:
   - Visualizes actual historical sales spikes, promotion days, and sales volume comparisons across different categories.
3. **🧠 Explainable AI (SHAP)**:
   - Answers *why* the model made that specific forecast (e.g., did yesterday's sales or promotional status push the prediction up?).
4. **📖 Model Details & Verification (Academic & Evaluation)**:
   - Compares LightGBM's performance against the historical Seasonal Naive benchmark and displays the mathematical formulas governing the inventory logic.

---

## 2. Core Terminology Glossary

### A. Forecasting & Machine Learning Terms

| Term | What It Means | Why It Matters |
| :--- | :--- | :--- |
| **Median Forecast ($Q_{50}$)** | The 50th percentile prediction. There is an equal (50%) chance that actual sales will be higher or lower than this value. | This serves as our **expected central demand**. Unlike a standard mean (average), the median is not skewed by extreme sales spikes. |
| **Lower Bound ($Q_{10}$)** | The 10th percentile prediction. There is only a 10% chance that demand will drop below this floor. | Represents low-demand scenarios (slow days). |
| **Upper Bound ($Q_{90}$)** | The 90th percentile prediction. There is a 90% chance that demand will be at or below this ceiling. | Represents surge demand scenarios (busy days). |
| **Prediction Interval (PI)** | The range between $Q_{10}$ and $Q_{90}$. In this system, this represents an **80% Prediction Interval**. | Communicates **uncertainty**. A narrow band means the model is confident; a wide band indicates high volatility. |
| **Quantile Crossing** | An anomaly where an upper quantile model outputs a number smaller than the median ($Q_{90} < Q_{50}$). | Our system implements an **order-preserving safeguard** at inference time so $Q_{10} \le Q_{50} \le Q_{90}$ is always maintained. |
| **Seasonal Naive Benchmark** | A baseline forecast that assumes demand this week will be identical to demand exactly 7 days ago. | Used as a benchmark. If a machine learning model cannot beat this simple rule, it is not worth deploying. |
| **Data Leakage** | An error where future information is accidentally included in model training features. | All rolling averages and lags in this system strictly use `.shift(1)` so today's actual sales are never used to predict today. |

---

### B. Inventory Management & Supply Chain Terms

| Term | Symbol | What It Means | Formula |
| :--- | :---: | :--- | :--- |
| **Stock on Hand** | - | Physical inventory currently sitting on shelves or in the backroom. | Input value |
| **Units on Order** | - | Products already purchased from the supplier that are in transit and have not yet arrived. | Input value |
| **Backorders** | - | Customer orders that have been placed but cannot be fulfilled immediately due to lack of stock. | Input value |
| **Net Inventory Position** | $I_{pos}$ | The total effective stock currently available or committed. | $I_{pos} = \text{On-Hand} + \text{On-Order} - \text{Backorders}$ |
| **Review Period** | $R$ | The number of days between regular inventory ordering reviews (e.g., checking stock every 7 days). | Input value |
| **Supplier Lead Time** | $L$ | The number of days it takes for an order to arrive after placing it with the supplier. | Input value |
| **Protection Horizon** | $H$ | The total window of time that an inventory order must protect against stockouts ($R + L$). | $H = R + L$ |
| **Cumulative Demand** | $D_H$ | The total expected demand summed across the entire protection horizon. | $D_H = \sum_{t=1}^{H} Q_{50,t}$ |
| **Cycle Service Level** | $p$ | The target probability of **not stocking out** during a replenishment cycle (e.g., 95% or 99%). | Input value |
| **Z-Score Multiplier** | $z_p$ | The standard normal distribution factor corresponding to the chosen service level ($z_{0.95} \approx 1.645$, $z_{0.99} \approx 2.326$). | $\Phi^{-1}(p)$ |
| **Demand Uncertainty** | $\sigma_H$ | The aggregate standard deviation of forecast error across the entire protection horizon. | $\sigma_H = \sqrt{\sum_{t=1}^H \left(\frac{Q_{90,t} - Q_{10,t}}{2.564}\right)^2}$ |
| **Safety Stock** | $SS$ | Buffer inventory kept to absorb unexpected spikes in demand during lead times. | $SS = \max(0, \lceil z_p \cdot \sigma_H \rceil)$ |
| **Order-Up-To Level** | $S$ | The maximum target inventory position required to satisfy expected demand plus safety buffer. | $S = D_H + SS$ |
| **Recommended Order Quantity** | $Q$ | The final whole-unit quantity that should be ordered right now. | $Q = \max(0, \lceil S - I_{pos} \rceil)$ |

---

### C. Explainable AI & Evaluation Terms

| Term | What It Means | How to Interpret It on the Dashboard |
| :--- | :--- | :--- |
| **SHAP (SHapley Additive exPlanations)** | A game-theoretic technique that measures how much each feature contributed to moving a prediction away from the baseline average. | Explains **model behaviour**, not physical causality. |
| **Global Feature Importance** | The average magnitude of influence a feature has across all training samples ($\text{Mean } \|SHAP\|$). | High values (like `roll_mean_30`) mean the model relies heavily on recent 30-day sales trends across all products. |
| **Local SHAP Attribution** | Feature breakdown for **one specific prediction** (e.g., tomorrow's sales at Store 1 for Groceries). | **Red bars** indicate features that pushed the forecast higher; **blue bars** indicate features that dragged the forecast lower. |
| **MAE (Mean Absolute Error)** | The average absolute error between predicted sales and actual sales in units. | Lower is better. |
| **RMSE (Root Mean Squared Error)** | Similar to MAE, but squares errors before averaging, penalizing large forecasting blunders much more severely. | Lower is better. |
| **WAPE (Weighted Absolute Percentage Error)** | Total absolute error divided by total actual demand: $\frac{\sum \|y - \hat{y}\|}{\sum \|y\|}$. | Preferred over MAPE for retail because it handles zero-sales days safely without division-by-zero errors. |
| **Pinball Loss** | The asymmetric loss function used to evaluate quantile regression accuracy. | Evaluates whether $Q_{10}$, $Q_{50}$, and $Q_{90}$ are accurately positioned. |
| **Empirical Coverage** | The actual percentage of historical validation dates where actual sales fell between $Q_{10}$ and $Q_{90}$. | Ideally close to 80%. |
| **Censored Demand Limitation** | When a store runs out of stock, sales drop to 0, masking true consumer demand. | The Favorita dataset contains sales, not stock levels. Thus, the system documents that true censored demand recovery is omitted due to missing inventory logs. |

---

## 3. Walkthrough: How an Order Quantity is Calculated

Suppose you are a store manager reviewing **Store 1 — Beverages**:

1. **You input your status**:
   - Stock on Hand = 150 units
   - Units on Order = 50 units
   - Backorders = 0 units
   - Review Period = 7 days, Lead Time = 3 days $\rightarrow$ **Protection Horizon $H = 10$ days**.
   - Target Service Level = 95% $\rightarrow z_{0.95} = 1.645$.

2. **The system calculates**:
   - **Net Position**: $150 + 50 - 0 = 200\text{ units}$.
   - **Forecast**: Summing daily $Q_{50}$ forecasts over the 10-day window gives $D_H = 1,038\text{ units}$.
   - **Uncertainty ($\sigma_H$)**: Using the spread between $Q_{10}$ and $Q_{90}$ gives $\sigma_H = 42.6\text{ units}$.
   - **Safety Stock**: $\lceil 1.645 \times 42.6 \rceil = 70\text{ units}$.
   - **Target Level ($S$)**: $1,038 + 70 = 1,108\text{ units}$.
   - **Final Order Recommendation ($Q$)**: $1,108 - 200 = \mathbf{908\text{ units}}$.

3. **Plain-Language Summary**:
   > *"You need 1,108 units to cover expected sales and safety stock over the next 10 days. Since you already have 200 units available or on the way, place an order for 908 units."*
