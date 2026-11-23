"""
app.py — AI-Based Retail Demand Forecasting, Inventory Optimization, and Explainable AI Decision-Support System.
"Research-Grade Enterprise Retail Intelligence, Probabilistic Inventory Optimization & Explainable AI Decision Microservice."

Features:
- Multi-Model Quantile Ensembles (LightGBM + XGBoost + CatBoost + Seasonal Naive)
- Dynamic Validation-Weighted Ensemble with Inverse Loss Optimization
- Finite-Sample Conformalized Quantile Regression (CQR) Calibration
- Probabilistic Inventory Optimization: Tail Stockout Risk P(Demand > I_pos), Expected Fill Rate, Overstock Expectation
- ABC-XYZ 9-Box Portfolio Segmentation with Strategy Recommendations
- Real-Time Model Drift & Covariate Shift Monitoring (Population Stability Index PSI, KS-test, Tracking Signal)
- Commercial Supplier Purchase Order Engine (Case Pack Rounding, MOQ, ReportLab PDF / CSV)
- Historical Inventory Policy Simulator & Financial Backtesting
- Custom Merchant POS Ingestion Pipeline & Diagnostic Health Auditor
- Game-Theoretic TreeSHAP Explainability & Grounded Plain-Language Rationales
- Production REST API Microservice Integration
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.sample_data import generate_sample_train
from src.data_pipeline import (
    load_and_preprocess,
    load_processed,
    save_processed,
)
from src.train_model import (
    run_training_pipeline,
    load_models,
    models_exist,
)
from src.forecasting import generate_forecast, seasonal_naive_forecast
from src.evaluation import format_metrics_table
from src.inventory import InventoryInputs, compute_inventory
from src.explainability import (
    is_shap_available,
    global_feature_importance,
    local_shap_explanation,
    generate_rationale,
)
from src.conformal import (
    fit_conformal_calibrator,
    apply_conformal_calibration,
    calibrate_forecast_dataframe,
    evaluate_conformal_calibration,
)
from src.simulation import (
    CostParameters,
    PolicyType,
    run_comparative_simulation,
    format_simulation_comparison_table,
    simulate_policy,
)
from src.benchmarks import (
    run_walk_forward_benchmark,
    XGB_AVAILABLE,
)
from src.po_generator import (
    SupplierInfo,
    PurchaseOrderItem,
    PurchaseOrder,
    create_purchase_order,
    export_po_to_dataframe,
    export_po_to_csv_string,
    generate_po_pdf,
)
from src.custom_data import (
    ingest_merchant_dataset,
    generate_custom_features,
    identify_columns,
)
from src.segmentation import (
    classify_abc_xyz,
    generate_segmentation_matrix_summary,
    SEGMENT_POLICIES,
)
from src.probabilistic_inventory import (
    ProbabilisticInventoryInputs,
    compute_probabilistic_inventory,
)
from src.drift_monitor import (
    audit_dataset_drift,
    calculate_psi,
    calculate_tracking_signal,
)
from src.forecasting_ensemble import (
    HybridQuantileEnsemble,
    CATBOOST_AVAILABLE,
)
from src.feature_selection import (
    run_automated_feature_selection,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Based Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── StockWise Visual Design System (CSS) ───────────────────────────────────────
st.markdown("""
<style>
    /* Force Light Streamlit Top Header */
    header[data-testid="stHeader"] {
        background-color: #F7F8F5 !important;
        color: #202820 !important;
    }
    header[data-testid="stHeader"] svg,
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a {
        color: #202820 !important;
        fill: #202820 !important;
    }
    
    /* Base theme background & text */
    .stApp {
        background-color: #F7F8F5 !important;
        color: #202820 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Streamlit Main Markdown & Paragraphs default text color */
    .stApp p, .stApp span, .stApp label, .stApp div {
        color: #202820;
    }
    
    /* Top padding reduction */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1180px;
    }
    
    /* Sidebar container styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E4E9E2 !important;
    }
    [data-testid="stSidebar"] * {
        color: #202820 !important;
    }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] caption {
        color: #70796F !important;
    }

    /* Sidebar Radio Navigation Item Styling */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #202820 !important;
        font-weight: 500 !important;
        padding: 0.35rem 0.5rem !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #F0F4EF !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-checked="true"] {
        color: #287A50 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.15rem !important;
    }

    /* Clean Cards */
    .sw-card {
        background-color: #FFFFFF;
        border: 1px solid #E4E9E2;
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 1px 3px rgba(32, 40, 32, 0.04);
        margin-bottom: 1rem;
    }
    
    .sw-card-highlight {
        background-color: #FFFFFF;
        border: 1px solid #CDE2D3;
        border-left: 5px solid #287A50;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 6px rgba(40, 122, 80, 0.06);
        margin-bottom: 1.2rem;
    }
    
    .sw-card-warning {
        background-color: #FFFFFF;
        border: 1px solid #F3E0C4;
        border-left: 5px solid #D99024;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 6px rgba(217, 144, 36, 0.06);
        margin-bottom: 1.2rem;
    }

    /* Metric Boxes */
    .sw-metric-box {
        background-color: #FFFFFF;
        border: 1px solid #E4E9E2;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        height: 100%;
        box-shadow: 0 1px 2px rgba(32, 40, 32, 0.03);
    }
    .sw-metric-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #70796F;
        margin-bottom: 0.35rem;
    }
    .sw-metric-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #202820;
        line-height: 1.15;
    }
    .sw-metric-sub {
        font-size: 0.8rem;
        color: #70796F;
        margin-top: 0.35rem;
    }

    /* Badges */
    .sw-badge-green {
        display: inline-block;
        background-color: #EBF5EE;
        color: #287A50;
        border: 1px solid #CDE2D3;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        text-transform: uppercase;
    }
    .sw-badge-orange {
        display: inline-block;
        background-color: #FDF4E7;
        color: #B87214;
        border: 1px solid #F3E0C4;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        text-transform: uppercase;
    }
    .sw-badge-red {
        display: inline-block;
        background-color: #FDF2F2;
        color: #D94F45;
        border: 1px solid #F8D7D5;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        text-transform: uppercase;
    }
    .sw-badge-blue {
        display: inline-block;
        background-color: #EDF2FD;
        color: #2B57B0;
        border: 1px solid #CDDBF8;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        text-transform: uppercase;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background-color: #287A50 !important;
        border-color: #287A50 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.45rem 1rem !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #206240 !important;
        border-color: #206240 !important;
        color: #FFFFFF !important;
    }
    
    /* Headings hierarchy */
    h1, h2, h3, h4, h5, h6 {
        color: #202820 !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    .sw-page-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #202820 !important;
        margin-bottom: 0.2rem;
    }
    .sw-page-subtitle {
        font-size: 0.95rem;
        color: #70796F !important;
        margin-bottom: 1.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cached Data & Model Loading
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading store records...")
def get_data() -> tuple[pd.DataFrame, dict, str]:
    if config.PROCESSED_TRAIN.exists():
        try:
            df = load_processed()
            meta = {
                "stores": sorted(df["store_nbr"].unique().tolist()),
                "families": sorted(df["family"].unique().tolist()),
                "date_min": str(df["date"].min().date()),
                "date_max": str(df["date"].max().date()),
                "n_rows": len(df),
                "data_source": "real",
            }
            return df, meta, "real"
        except Exception:
            pass

    if config.TRAIN_CSV.exists():
        try:
            df, meta = load_and_preprocess()
            save_processed(df)
            return df, meta, "real"
        except Exception as exc:
            st.error(f"Failed to load Kaggle data: {exc}")

    df = generate_sample_train()
    meta = {
        "stores": config.SAMPLE_STORES,
        "families": config.SAMPLE_FAMILIES,
        "date_min": config.SAMPLE_START,
        "date_max": config.SAMPLE_END,
        "n_rows": len(df),
        "data_source": "sample",
    }
    return df, meta, "sample"


@st.cache_resource(show_spinner="Initializing forecasting models...")
def get_models():
    if models_exist():
        try:
            return load_models()
        except Exception:
            pass

    df, _, _ = get_data()
    run_training_pipeline(df)
    return load_models()


df, meta, data_source = get_data()


# ══════════════════════════════════════════════════════════════════════════════
# Session State Initialization
# ══════════════════════════════════════════════════════════════════════════════

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

if "on_hand" not in st.session_state:
    st.session_state.on_hand = 150.0

if "on_order" not in st.session_state:
    st.session_state.on_order = 50.0

if "backorders" not in st.session_state:
    st.session_state.backorders = 0.0

if "review_period" not in st.session_state:
    st.session_state.review_period = 7

if "lead_time" not in st.session_state:
    st.session_state.lead_time = 3

if "service_level" not in st.session_state:
    st.session_state.service_level = 0.95

if "pack_size" not in st.session_state:
    st.session_state.pack_size = 12

if "moq" not in st.session_state:
    st.session_state.moq = 24

if "unit_cost" not in st.session_state:
    st.session_state.unit_cost = 8.50

if "supplier_name" not in st.session_state:
    st.session_state.supplier_name = "Andean Consumer Goods Logistics S.A."


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — Navigation & Store Selector
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.4rem 0 0.8rem 0;">
        <div style="font-size: 1.25rem; font-weight: 800; color: #287A50; letter-spacing: -0.02em; line-height: 1.25;">🛒 AI-Based Retail Demand Forecasting</div>
        <div style="font-size: 0.8rem; color: #70796F; font-weight: 500; margin-top: 0.2rem;">Inventory Intelligence & XAI Decision Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    nav_options = [
        "Home",
        "Inventory & Risk",
        "Order Suggestions & PO",
        "ABC-XYZ Segmentation",
        "Policy Simulator",
        "Model Drift & Monitor",
        "Custom Store Upload",
        "Sales Trends",
        "Advanced Analytics",
        "Settings & API Microservice",
    ]
    
    def nav_format(opt):
        icons = {
            "Home": "🏠 Home",
            "Inventory & Risk": "📦 Inventory & Risk",
            "Order Suggestions & PO": "📋 Order Suggestions & PO",
            "ABC-XYZ Segmentation": "📊 ABC-XYZ Segmentation",
            "Policy Simulator": "⚡ Policy Simulator",
            "Model Drift & Monitor": "🛡️ Model Drift & Monitor",
            "Custom Store Upload": "📂 Custom Store Upload",
            "Sales Trends": "📈 Sales Trends",
            "Advanced Analytics": "🔬 Advanced Analytics",
            "Settings & API Microservice": "⚙️ Settings & REST API",
        }
        return icons.get(opt, opt)

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=nav_options.index(st.session_state.nav_page) if st.session_state.nav_page in nav_options else 0,
        format_func=nav_format,
        label_visibility="collapsed",
    )
    st.session_state.nav_page = selected_nav

    st.markdown("---")
    
    stores = sorted(df["store_nbr"].unique().tolist())
    families = sorted(df["family"].unique().tolist())

    if "sel_store" not in st.session_state:
        st.session_state.sel_store = stores[0]
    if "sel_family" not in st.session_state:
        st.session_state.sel_family = families[0]

    st.markdown(f"""
    <div style="background-color: #F7F8F5; border: 1px solid #E4E9E2; border-radius: 8px; padding: 0.7rem 0.85rem; margin-top: 0.5rem;">
        <div style="font-size: 0.72rem; font-weight: 700; color: #70796F; text-transform: uppercase; letter-spacing: 0.04em;">Active Category</div>
        <div style="font-size: 0.92rem; font-weight: 700; color: #202820; margin-top: 0.15rem;">Store {st.session_state.sel_store} · {st.session_state.sel_family}</div>
        <div style="font-size: 0.75rem; color: #70796F; margin-top: 0.1rem;">Timeline: {meta['date_min']} to {meta['date_max']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 0.75rem; color: #70796F; text-align: center; font-weight: 500;'>Retail Demand AI v2.0 · Commercial Grade</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Shared Computations (Forecast, Probabilistic Inventory, and Risk Engine)
# ══════════════════════════════════════════════════════════════════════════════

sel_store = st.session_state.sel_store
sel_family = st.session_state.sel_family

H = int(st.session_state.review_period + st.session_state.lead_time)
calc_horizon = max(H, 7)

if not models_exist():
    st.warning("Models not yet loaded. Initializing...")
    run_training_pipeline(df)

q10_model, q50_model, q90_model, model_meta = get_models()

try:
    fc_df = generate_forecast(
        history_df=df,
        q10_model=q10_model,
        q50_model=q50_model,
        q90_model=q90_model,
        meta=model_meta,
        store_nbr=sel_store,
        family=sel_family,
        horizon=calc_horizon,
    )
    base_df = seasonal_naive_forecast(df, sel_store, sel_family, horizon=calc_horizon)

    # Probabilistic Inventory Calculations
    prob_inputs = ProbabilisticInventoryInputs(
        on_hand=st.session_state.on_hand,
        on_order=st.session_state.on_order,
        backorders=st.session_state.backorders,
        review_period=st.session_state.review_period,
        lead_time=st.session_state.lead_time,
        target_service_level=st.session_state.service_level,
        unit_cost=st.session_state.unit_cost,
        selling_price=st.session_state.unit_cost * 1.6,
    )
    prob_res = compute_probabilistic_inventory(prob_inputs, fc_df)
    
    seven_day_demand = float(fc_df["pred_q50"].iloc[:7].sum())
    daily_rate = max(0.1, prob_res.expected_demand_horizon_DH / max(1, prob_res.protection_horizon_H))
    days_remaining = round(st.session_state.on_hand / daily_rate, 1)

except Exception as err:
    st.error(f"Error computing store forecasts: {err}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1: Home — Operational Posture & Risk Alerts
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.nav_page == "Home":
    st.markdown('<div class="sw-page-title">Operational Posture & Inventory Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Real-time stock posture, tail stockout probabilities, upcoming demand, and procurement action items.</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background-color: #FFFFFF; border: 1px solid #E4E9E2; border-radius: 10px; padding: 0.75rem 1.1rem; margin-bottom: 1.25rem; font-size: 0.9rem; display: flex; align-items: center; justify-content: space-between;">
        <div>
            <span style="color: #70796F;">Active selection:</span> 
            <strong>Store {sel_store}</strong> — <strong>{sel_family}</strong>
        </div>
        <div style="color: #70796F; font-size: 0.82rem;">
            Protection Horizon: <strong>{prob_res.protection_horizon_H} Days</strong> (Review: {st.session_state.review_period}d + Lead: {st.session_state.lead_time}d)
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        needs_order = prob_res.recommended_order_quantity > 0
        status_badge = '<span class="sw-badge-orange">Needs Reorder</span>' if needs_order else '<span class="sw-badge-green">Stock Healthy</span>'
        st.markdown(f"""
        <div class="sw-metric-box">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="sw-metric-label">Recommended Order</div>
                {status_badge}
            </div>
            <div class="sw-metric-val">{prob_res.recommended_order_quantity:,.0f} <span style="font-size: 1rem; font-weight: 500; color: #70796F;">units</span></div>
            <div class="sw-metric-sub">Target S = {prob_res.target_level_S:.0f} (Reorder pt s = {prob_res.reorder_point_s:.0f})</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Current Stock on Hand</div>
            <div class="sw-metric-val">{st.session_state.on_hand:,.0f} <span style="font-size: 1rem; font-weight: 500; color: #70796F;">units</span></div>
            <div class="sw-metric-sub">Physical warehouse & shelf inventory</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        risk_badge = '<span class="sw-badge-red">Elevated</span>' if prob_res.stockout_probability_pct > 20 else '<span class="sw-badge-green">Low</span>'
        st.markdown(f"""
        <div class="sw-metric-box">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="sw-metric-label">Tail Stockout Risk</div>
                {risk_badge}
            </div>
            <div class="sw-metric-val">{prob_res.stockout_probability_pct:.1f}%</div>
            <div class="sw-metric-sub">Expected Fill Rate: <strong>{prob_res.expected_fill_rate_pct:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">7-Day Projected Sales</div>
            <div class="sw-metric-val">~{seven_day_demand:,.0f} <span style="font-size: 1rem; font-weight: 500; color: #70796F;">units</span></div>
            <div class="sw-metric-sub">LightGBM Q50 median demand</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    action_card_class = "sw-card-highlight" if prob_res.recommended_order_quantity > 0 else "sw-card-highlight"
    action_title = "PROCUREMENT DECISION ADVICE"
    action_headline = prob_res.risk_urgency_tier
    action_desc = prob_res.detailed_explanation

    st.markdown(f"""
    <div class="{action_card_class}">
        <div class="sw-action-title">{action_title}</div>
        <div class="sw-action-main">{action_headline}</div>
        <div class="sw-action-desc">{action_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    btn_col1, btn_col2, btn_col3, _ = st.columns([2, 2, 2, 2])
    with btn_col1:
        if st.button("Generate Purchase Order (PO) →", type="primary", use_container_width=True):
            st.session_state.nav_page = "Order Suggestions & PO"
            st.rerun()
    with btn_col2:
        if st.button("Explore ABC-XYZ Matrix →", use_container_width=True):
            st.session_state.nav_page = "ABC-XYZ Segmentation"
            st.rerun()
    with btn_col3:
        if st.button("Check Model Drift Status →", use_container_width=True):
            st.session_state.nav_page = "Model Drift & Monitor"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Probabilistic Multi-Step Demand Projection")
    st.caption("Green depicts observed store transactions; blue illustrates upcoming probabilistic quantile uncertainty bands [Q10 to Q90].")

    hist_tail = df[(df["store_nbr"] == sel_store) & (df["family"] == sel_family)].sort_values("date").tail(21)
    fc_preview = fc_df.iloc[:7]

    fig_prev = go.Figure()
    fig_prev.add_trace(go.Scatter(
        x=hist_tail["date"],
        y=hist_tail["sales"],
        mode="lines",
        name="Observed Store Sales",
        line=dict(color="#287A50", width=2.2),
    ))
    fig_prev.add_trace(go.Scatter(
        x=pd.concat([fc_preview["date"], fc_preview["date"][::-1]]),
        y=pd.concat([fc_preview["pred_q90"], fc_preview["pred_q10"][::-1]]),
        fill="toself",
        fillcolor="rgba(79, 107, 237, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Prediction Interval [Q10, Q90]",
        hoverinfo="skip",
    ))
    fig_prev.add_trace(go.Scatter(
        x=fc_preview["date"],
        y=fc_preview["pred_q50"],
        mode="lines+markers",
        name="Median Forecast (Q50)",
        line=dict(color="#4F6BED", width=2.5),
        marker=dict(size=6, color="#4F6BED"),
    ))
    fig_prev.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#F0F2ED", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#F0F2ED", zeroline=False, title="Daily Units"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_prev, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2: Inventory & Risk — Probabilistic Optimization
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Inventory & Risk":
    st.markdown('<div class="sw-page-title">Probabilistic Inventory Register & Tail-Risk Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Quantify tail stockout probabilities, expected shortage, and overstock exposure under probabilistic demand.</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Stockout Probability</div>
            <div class="sw-metric-val" style="color: {'#D94F45' if prob_res.stockout_probability_pct > 20 else '#287A50'};">{prob_res.stockout_probability_pct:.1f}%</div>
            <div class="sw-metric-sub">P(Demand > Net Position)</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Expected Fill Rate</div>
            <div class="sw-metric-val">{prob_res.expected_fill_rate_pct:.1f}%</div>
            <div class="sw-metric-sub">Expected demand satisfied</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Dynamic Reorder Point (s)</div>
            <div class="sw-metric-val">{prob_res.reorder_point_s:.0f} <span style="font-size: 0.9rem; font-weight: 500; color: #70796F;">units</span></div>
            <div class="sw-metric-sub">Lead time demand + buffer</div>
        </div>
        """, unsafe_allow_html=True)
    with r4:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Target Level (S)</div>
            <div class="sw-metric-val" style="color: #287A50;">{prob_res.target_level_S:.0f} <span style="font-size: 0.9rem; font-weight: 500; color: #70796F;">units</span></div>
            <div class="sw-metric-sub">Horizon demand + Safety stock ({prob_res.safety_stock}u)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Planning Parameters & Cost Factors")
    with st.form("inventory_form"):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            new_on_hand = st.number_input(
                "Current stock on hand (units)",
                min_value=0.0,
                value=float(st.session_state.on_hand),
                step=10.0,
            )
            new_on_order = st.number_input(
                "Units already on order",
                min_value=0.0,
                value=float(st.session_state.on_order),
                step=10.0,
            )
        with col_s2:
            new_backorders = st.number_input(
                "Outstanding customer backorders",
                min_value=0.0,
                value=float(st.session_state.backorders),
                step=5.0,
            )
            new_review = st.number_input(
                "Review period (days, R)",
                min_value=1,
                max_value=30,
                value=int(st.session_state.review_period),
                step=1,
            )
        with col_s3:
            new_lead = st.number_input(
                "Supplier lead time (days, L)",
                min_value=1,
                max_value=30,
                value=int(st.session_state.lead_time),
                step=1,
            )
            new_svc = st.select_slider(
                "Cycle service level target",
                options=[0.85, 0.90, 0.95, 0.98, 0.99],
                value=float(st.session_state.service_level),
                format_func=lambda x: f"{int(x*100)}% Service Level",
            )

        submit_btn = st.form_submit_button("Recalculate Probabilistic Plan", type="primary")
        if submit_btn:
            st.session_state.on_hand = new_on_hand
            st.session_state.on_order = new_on_order
            st.session_state.backorders = new_backorders
            st.session_state.review_period = new_review
            st.session_state.lead_time = new_lead
            st.session_state.service_level = new_svc
            st.success("Parameters saved and probabilistic risks recalculated!")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3: Order Suggestions & Purchase Order Generator
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Order Suggestions & PO":
    st.markdown('<div class="sw-page-title">Order Suggestions & Supplier PO Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Algorithmic replenishment transformed into commercial, supplier-ready purchase orders.</div>', unsafe_allow_html=True)

    order_qty = prob_res.recommended_order_quantity
    
    st.markdown(f"""
    <div class="sw-card-highlight">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div class="sw-action-title">ALGORITHMIC REORDER RECOMMENDATION FOR {sel_family}</div>
                <div style="font-size: 2.6rem; font-weight: 800; color: #287A50; line-height: 1.1;">
                    {order_qty:,.0f} <span style="font-size: 1.25rem; font-weight: 500; color: #70796F;">units</span>
                </div>
            </div>
            <span class="sw-badge-green" style="font-size: 0.9rem; padding: 0.4rem 0.8rem;">
                { 'PO Recommended' if order_qty > 0 else 'Stock Adequate' }
            </span>
        </div>
        <div style="font-size: 0.95rem; color: #485247; margin-top: 0.8rem; line-height: 1.5;">
            Dynamic Reorder Point <em>s</em> = <strong>{prob_res.reorder_point_s:.0f} units</strong>. Target level <em>S</em> = <strong>{prob_res.target_level_S:.0f} units</strong>. Current position is <strong>{prob_res.net_inventory_position:.0f} units</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Commercial Procurement Constraints")
    with st.expander("⚙️ Distributor & Packaging Configuration", expanded=True):
        po_c1, po_c2, po_c3, po_c4 = st.columns(4)
        with po_c1:
            pack_size = st.number_input("Case Pack Size (units/case)", min_value=1, value=int(st.session_state.pack_size), step=1)
        with po_c2:
            moq = st.number_input("Minimum Order Qty (MOQ)", min_value=0, value=int(st.session_state.moq), step=12)
        with po_c3:
            unit_cost = st.number_input("Distributor Unit Cost ($)", min_value=0.1, value=float(st.session_state.unit_cost), step=0.5)
        with po_c4:
            sup_name = st.text_input("Supplier Profile", value=st.session_state.supplier_name)

        st.session_state.pack_size = pack_size
        st.session_state.moq = moq
        st.session_state.unit_cost = unit_cost
        st.session_state.supplier_name = sup_name

    supplier = SupplierInfo(name=sup_name)
    po_item_dict = {
        "family": sel_family,
        "raw_recommended_qty": order_qty,
        "pack_size": pack_size,
        "moq": moq,
        "unit_cost": unit_cost,
        "lead_time_days": st.session_state.lead_time,
    }
    po = create_purchase_order(store_nbr=sel_store, replenishment_items=[po_item_dict], supplier=supplier)
    po_item = po.items[0]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Official Purchase Order Preview")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Cases Ordered</div>
            <div class="sw-metric-val">{po_item.cases_ordered:,} <span style="font-size: 0.9rem; font-weight: 500; color: #70796F;">cases</span></div>
            <div class="sw-metric-sub">{pack_size} units per case</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Final Units Transmitted</div>
            <div class="sw-metric-val">{po_item.rounded_units:,} <span style="font-size: 0.9rem; font-weight: 500; color: #70796F;">units</span></div>
            <div class="sw-metric-sub">MOQ: {moq} units enforced</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Total Purchase Valuation</div>
            <div class="sw-metric-val">${po.total_cost:,.2f}</div>
            <div class="sw-metric-sub">${unit_cost:.2f} per unit wholesale</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Expected Delivery ETA</div>
            <div class="sw-metric-val" style="font-size: 1.4rem; color: #287A50;">{po_item.expected_delivery_date}</div>
            <div class="sw-metric-sub">{st.session_state.lead_time} days supplier lead time</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    po_df = export_po_to_dataframe(po)
    st.dataframe(po_df, hide_index=True, use_container_width=True)

    down_col1, down_col2, _ = st.columns([2.5, 2, 3])
    with down_col1:
        pdf_bytes = generate_po_pdf(po)
        st.download_button(
            label="📄 Download Official Supplier PO (PDF)",
            data=pdf_bytes,
            file_name=f"{po.po_number}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
    with down_col2:
        csv_str = export_po_to_csv_string(po)
        st.download_button(
            label="📊 Export PO as CSV",
            data=csv_str,
            file_name=f"{po.po_number}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 4: ABC-XYZ Portfolio Segmentation Matrix (NEW)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "ABC-XYZ Segmentation":
    st.markdown('<div class="sw-page-title">ABC-XYZ Product Portfolio Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">9-Box classification mapping sales volume velocity (ABC) against demand volatility (XYZ) to drive differentiated replenishment strategies.</div>', unsafe_allow_html=True)

    seg_df = classify_abc_xyz(df, store_nbr=sel_store)
    grid_summary = generate_segmentation_matrix_summary(seg_df)

    st.markdown("### 3×3 Portfolio Classification Grid")
    st.dataframe(grid_summary, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Portfolio Volatility Scatter (Volume Share % vs. Demand CV)")
    
    fig_seg = go.Figure()

    colors_map = {
        "AX": "#287A50", "AY": "#4F6BED", "AZ": "#D94F45",
        "BX": "#63A375", "BY": "#7B92F2", "BZ": "#E27B73",
        "CX": "#9BC4A7", "CY": "#A6B6F7", "CZ": "#EAA7A2",
    }

    for segment_name, group in seg_df.groupby("segment"):
        fig_seg.add_trace(go.Scatter(
            x=group["cum_share"] * 100.0,
            y=group["cv"],
            mode="markers+text",
            text=group["family"],
            textposition="top center",
            name=f"Segment {segment_name}",
            marker=dict(
                size=14,
                color=colors_map.get(segment_name, "#555555"),
                line=dict(width=1, color="#202820"),
            ),
        ))

    # Cutoff reference lines
    fig_seg.add_vline(x=80, line_dash="dash", line_color="#70796F", annotation_text="Class A Cutoff (80%)")
    fig_seg.add_vline(x=95, line_dash="dash", line_color="#70796F", annotation_text="Class B Cutoff (95%)")
    fig_seg.add_hline(y=0.50, line_dash="dash", line_color="#70796F", annotation_text="Class X Cutoff (CV=0.5)")
    fig_seg.add_hline(y=1.00, line_dash="dash", line_color="#70796F", annotation_text="Class Y Cutoff (CV=1.0)")

    fig_seg.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=10, b=10),
        height=450,
        xaxis=dict(title="Cumulative Volume Share (%)", showgrid=True, gridcolor="#F0F2ED"),
        yaxis=dict(title="Demand Volatility (Coefficient of Variation CV = σ / μ)", showgrid=True, gridcolor="#F0F2ED"),
    )
    st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Differentiated Policy Rules by Category")
    display_cols = [
        "family", "segment", "priority_level", "target_service_level",
        "recommended_review_period", "forecasting_strategy", "inventory_strategy"
    ]
    st.dataframe(seg_df[display_cols], hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 5: Policy Simulator & Financial Backtesting
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Policy Simulator":
    st.markdown('<div class="sw-page-title">Historical Inventory Policy Simulator & Financial Backtesting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Backtest StockWise Dynamic Quantile against Static Min-Max (s, S) and Moving Average heuristics over historical timelines.</div>', unsafe_allow_html=True)

    sub_df = df[(df["store_nbr"] == sel_store) & (df["family"] == sel_family)].sort_values("date")
    demand_series = sub_df.set_index("date")["sales"]

    st.markdown("### Simulation Parameters & Cost Modeling")
    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
    with sim_c1:
        sim_unit_cost = st.number_input("Wholesale Unit Cost ($)", value=10.0, step=1.0)
        sim_sell_price = st.number_input("Retail Selling Price ($)", value=16.0, step=1.0)
    with sim_c2:
        sim_holding_rate = st.number_input("Annualized Holding Cost Rate (%)", value=20.0, step=5.0) / 100.0
        sim_stockout_penalty = st.number_input("Stockout Penalty / Lost Unit ($)", value=8.0, step=1.0)
    with sim_c3:
        sim_fixed_order_cost = st.number_input("Fixed Order Cost ($/order)", value=25.0, step=5.0)
        sim_initial_stock = st.number_input("Initial On-Hand Stock (units)", value=100.0, step=20.0)
    with sim_c4:
        sim_service_level = st.selectbox("Simulated Service Level", [0.90, 0.95, 0.99], index=1)
        sim_days_to_run = st.slider("History Length (Days)", min_value=30, max_value=len(demand_series), value=min(len(demand_series), 45))

    truncated_demand = demand_series.tail(sim_days_to_run)
    costs = CostParameters(
        unit_cost=sim_unit_cost,
        selling_price=sim_sell_price,
        holding_cost_rate_annual=sim_holding_rate,
        stockout_penalty_per_unit=sim_stockout_penalty,
        order_fixed_cost=sim_fixed_order_cost,
    )

    if st.button("Run Multi-Policy Discrete-Event Simulation", type="primary"):
        with st.spinner("Simulating daily store operations across policies..."):
            sim_results = run_comparative_simulation(
                daily_demand_series=truncated_demand,
                review_period=st.session_state.review_period,
                lead_time=st.session_state.lead_time,
                service_level=sim_service_level,
                initial_stock=sim_initial_stock,
                cost_params=costs,
            )

            st.markdown("---")
            st.markdown("### Policy Benchmark Scorecard")
            comparison_table = format_simulation_comparison_table(sim_results)
            st.dataframe(comparison_table, hide_index=True, use_container_width=True)

            st.markdown("### Daily On-Hand Stock Trajectory")
            fig_sim = go.Figure()

            policy_colors = {
                PolicyType.STOCKWISE_DYNAMIC: "#287A50",
                PolicyType.STATIC_MIN_MAX: "#4F6BED",
                PolicyType.MOVING_AVERAGE: "#D99024",
            }

            for p_type, run_res in sim_results.items():
                fig_sim.add_trace(go.Scatter(
                    x=run_res.daily_log["date"],
                    y=run_res.daily_log["on_hand_end"],
                    mode="lines",
                    name=p_type.value,
                    line=dict(color=policy_colors.get(p_type, "#555555"), width=2.2),
                ))

            fig_sim.add_hline(y=0, line_dash="dash", line_color="#D94F45", annotation_text="Stockout Threshold (0 units)")

            fig_sim.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin=dict(l=10, r=10, t=10, b=10),
                height=360,
                xaxis=dict(showgrid=True, gridcolor="#F0F2ED", title="Simulation Date"),
                yaxis=dict(showgrid=True, gridcolor="#F0F2ED", title="Ending Stock On-Hand (Units)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_sim, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 6: Model Drift & System Health Monitoring (NEW)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Model Drift & Monitor":
    st.markdown('<div class="sw-page-title">Advanced Model Drift & System Health Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Track feature distribution shifts (PSI, Kolmogorov-Smirnov) and forecast tracking signal bias to safeguard production reliability.</div>', unsafe_allow_html=True)

    feat_cols = model_meta.get("feature_cols", [])
    split_idx = int(len(df) * 0.80)
    df_ref = df.iloc[:split_idx]
    df_cur = df.iloc[split_idx:]

    report = audit_dataset_drift(df_ref, df_cur, feature_cols=feat_cols, target_col="sales")

    dh1, dh2, dh3 = st.columns(3)
    with dh1:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">System Drift Status</div>
            <div class="sw-metric-val" style="color: {'#287A50' if report.overall_status == 'HEALTHY' else '#D94F45'};">{report.overall_status}</div>
            <div class="sw-metric-sub">{report.recommendation_summary}</div>
        </div>
        """, unsafe_allow_html=True)
    with dh2:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Mean Population Stability Index</div>
            <div class="sw-metric-val">{report.mean_psi_score:.3f}</div>
            <div class="sw-metric-sub">PSI &lt; 0.10: Stable | &gt; 0.25: Drift</div>
        </div>
        """, unsafe_allow_html=True)
    with dh3:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Retraining Recommendation</div>
            <div class="sw-metric-val" style="color: {'#D94F45' if report.retrain_recommended else '#287A50'};">{'ACTION NEEDED' if report.retrain_recommended else 'NOT REQUIRED'}</div>
            <div class="sw-metric-sub">Tracking Signal: {report.tracking_signal_status}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Feature-Level Stability & Divergence Scorecard")

    drift_rows = []
    for feat, f_res in report.feature_details.items():
        drift_rows.append({
            "Feature": feat,
            "PSI Score": f"{f_res.psi_score:.4f}",
            "KS Statistic": f"{f_res.ks_statistic:.4f}",
            "KS p-value": f"{f_res.ks_p_value:.4e}",
            "Drift Status": f_res.drift_status,
        })
    st.dataframe(pd.DataFrame(drift_rows), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 7: Custom Store Upload
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Custom Store Upload":
    st.markdown('<div class="sw-page-title">Custom Store / Merchant POS Ingestion</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Upload your own Point-of-Sale (POS) or inventory CSV/Excel exports for automated forecasting and replenishment planning.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Sales / Transaction File (.csv, .xlsx)",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is not None:
        try:
            fmt = "excel" if uploaded_file.name.endswith((".xlsx", ".xls")) else "csv"
            raw_bytes = uploaded_file.read()
            reg_df, report = ingest_merchant_dataset(io.BytesIO(raw_bytes), file_format=fmt)

            st.markdown("### Ingestion & Data Health Audit")
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                status_color = "sw-badge-green" if report.is_valid else "sw-badge-orange"
                st.markdown(f"""
                <div class="sw-metric-box">
                    <div class="sw-metric-label">Pipeline Status</div>
                    <div style="margin-top:0.3rem;"><span class="{status_color}">{'Validated' if report.is_valid else 'Failed'}</span></div>
                    <div class="sw-metric-sub">{report.status_message}</div>
                </div>
                """, unsafe_allow_html=True)
            with h2:
                st.markdown(f"""
                <div class="sw-metric-box">
                    <div class="sw-metric-label">Timeline Span</div>
                    <div class="sw-metric-val">{report.total_days} <span style="font-size: 0.9rem; font-weight: 500; color: #70796F;">days</span></div>
                    <div class="sw-metric-sub">{report.min_date} to {report.max_date}</div>
                </div>
                """, unsafe_allow_html=True)
            with h3:
                st.markdown(f"""
                <div class="sw-metric-box">
                    <div class="sw-metric-label">Categories / Stores</div>
                    <div class="sw-metric-val">{report.family_count} <span style="font-size: 0.9rem; font-weight: 500; color: #70796F;">items</span></div>
                    <div class="sw-metric-sub">Across {report.store_count} store location(s)</div>
                </div>
                """, unsafe_allow_html=True)
            with h4:
                st.markdown(f"""
                <div class="sw-metric-box">
                    <div class="sw-metric-label">Zero-Sales Ratio</div>
                    <div class="sw-metric-val">{report.zero_sales_pct*100:.1f}%</div>
                    <div class="sw-metric-sub">Intermittent demand factor</div>
                </div>
                """, unsafe_allow_html=True)

            if report.warnings:
                for w in report.warnings:
                    st.warning(f"Auditor Note: {w}")

            st.markdown("### Regularized Merchant Dataset Preview")
            st.dataframe(reg_df.head(20), use_container_width=True)

        except Exception as e:
            st.error(f"Failed to process custom file: {e}")
    else:
        st.info("💡 **Ready for real data**: Export recent sales transactions from Square, Shopify, Lightspeed, or SAP and upload them directly.")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 8: Sales Trends
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Sales Trends":
    st.markdown('<div class="sw-page-title">Sales Trends & Historical Volume</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Analyze historical demand patterns and upcoming multi-day forecasts.</div>', unsafe_allow_html=True)

    series_df = df[(df["store_nbr"] == sel_store) & (df["family"] == sel_family)].sort_values("date")
    hist_view = series_df.tail(60)

    fig_trends = go.Figure()
    fig_trends.add_trace(go.Scatter(
        x=hist_view["date"],
        y=hist_view["sales"],
        mode="lines",
        name="Observed Store Sales",
        line=dict(color="#287A50", width=2.2),
    ))
    fig_trends.add_trace(go.Scatter(
        x=pd.concat([fc_df["date"][:7], fc_df["date"][:7][::-1]]),
        y=pd.concat([fc_df["pred_q90"][:7], fc_df["pred_q10"][:7][::-1]]),
        fill="toself",
        fillcolor="rgba(79, 107, 237, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Prediction Interval [Q10, Q90]",
        hoverinfo="skip",
    ))
    fig_trends.add_trace(go.Scatter(
        x=fc_df["date"][:7],
        y=fc_df["pred_q50"][:7],
        mode="lines+markers",
        name="Median Forecast (Q50)",
        line=dict(color="#4F6BED", width=2.6),
        marker=dict(size=6, color="#4F6BED"),
    ))

    fig_trends.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#F0F2ED", zeroline=False, title="Date"),
        yaxis=dict(showgrid=True, gridcolor="#F0F2ED", zeroline=False, title="Units Sold"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_trends, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 9: Advanced Analytics & Research Suite
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Advanced Analytics":
    st.markdown('<div class="sw-page-title">Advanced Analytics & Model Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Hybrid Ensemble optimization, Conformal calibration (CQR), TreeSHAP feature attributions, and mathematical formulations.</div>', unsafe_allow_html=True)

    tech_tab1, tech_tab2, tech_tab3, tech_tab4, tech_tab5, tech_tab6 = st.tabs([
        "🤝 Hybrid Ensemble (Validation-Weighted)",
        "🎯 Conformal Prediction (CQR)",
        "🏁 Model Benchmarks (Walk-Forward CV)",
        "🔬 Explainable AI (TreeSHAP)",
        "📊 Automated Feature Selection",
        "📐 Mathematical Formulations",
    ])

    # Tab 1: Hybrid Ensemble
    with tech_tab1:
        st.markdown("#### Hybrid Quantile Ensemble (LightGBM + XGBoost + CatBoost + Seasonal Naive)")
        st.markdown("""
        Combines predictions using out-of-sample validation loss weights:
        $$w_m = \\frac{1 / (\\text{Loss}_m^p)}{\\sum_j 1 / (\\text{Loss}_j^p)}$$
        Rather than relying on arbitrary fixed weights, the system dynamically shifts weights toward the most accurate algorithm.
        """)

        if st.button("Fit & Benchmark Hybrid Ensemble"):
            with st.spinner("Training candidate models and evaluating out-of-sample weights..."):
                try:
                    sub_work = df[(df["store_nbr"] == sel_store) & (df["family"] == sel_family)].sort_values("date").reset_index(drop=True)
                    val_len = max(10, int(len(sub_work) * 0.20))
                    train_w = sub_work.iloc[:-val_len]
                    val_w = sub_work.iloc[-val_len:]

                    feat_cols = model_meta.get("feature_cols", [])
                    from src.features import build_features
                    train_w_feat = build_features(train_w)
                    val_w_feat = build_features(val_w)

                    X_tr = train_w_feat[feat_cols]
                    y_tr = train_w_feat["sales"].to_numpy()
                    X_va = val_w_feat[feat_cols]
                    y_va = val_w_feat["sales"].to_numpy()

                    ens = HybridQuantileEnsemble(mode="weighted_blend", include_catboost=True, include_xgboost=True)
                    e_rep = ens.fit(X_tr, y_tr, X_va, y_va, df_train_full=train_w, df_val_full=val_w)

                    st.success(f"Best Performing Individual Model: **{e_rep.best_single_model}** | Ensemble Validation WAPE: **{e_rep.ensemble_wape*100:.2f}%**")

                    score_rows = []
                    for m_name, c_m in e_rep.candidate_metrics.items():
                        score_rows.append({
                            "Candidate Model": m_name,
                            "Validation WAPE (%)": f"{c_m.wape*100:.2f}%",
                            "MAE": f"{c_m.mae:.2f}",
                            "RMSE": f"{c_m.rmse:.2f}",
                            "Pinball Loss": f"{c_m.pinball_avg:.3f}",
                            "Dynamic Weight (%)": f"{c_m.weight*100:.1f}%",
                            "Train Time": f"{c_m.training_time_sec:.2f}s",
                        })
                    st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)

                except Exception as ens_e:
                    st.error(f"Ensemble fitting error: {ens_e}")

    # Tab 2: Conformal CQR
    with tech_tab2:
        st.markdown("#### Conformalized Quantile Regression (Romano et al., 2019)")
        feat_cols = model_meta.get("feature_cols", [])
        if feat_cols and all(c in df.columns for c in feat_cols):
            val_tail = df.tail(min(300, len(df)))
            X_cal = val_tail[feat_cols]
            y_cal = val_tail["sales"].to_numpy()
            q10_raw = q10_model.predict(X_cal)
            q50_raw = q50_model.predict(X_cal)
            q90_raw = q90_model.predict(X_cal)

            cqr_res = evaluate_conformal_calibration(
                y_test=y_cal,
                q10_test=q10_raw,
                q50_test=q50_raw,
                q90_test=q90_raw,
                conformal_delta=fit_conformal_calibrator(y_cal, q10_raw, q90_raw, target_coverage=0.80),
                target_coverage=0.80,
            )

            cq1, cq2, cq3 = st.columns(3)
            with cq1:
                st.metric("Nominal Target Coverage", f"{cqr_res.nominal_coverage*100:.0f}%")
            with cq2:
                st.metric("Raw LightGBM Coverage", f"{cqr_res.raw_coverage*100:.1f}%")
            with cq3:
                st.metric("Conformal Calibrated Coverage", f"{cqr_res.calibrated_coverage*100:.1f}%", delta=f"{(cqr_res.calibrated_coverage - cqr_res.raw_coverage)*100:+.1f}%")

            st.info(f"Conformal non-conformity adjustment parameter: **Δ = {cqr_res.conformal_adjustment:.2f} units**. Guarantees finite-sample coverage at nominal target.")

    # Tab 3: Walk-Forward CV
    with tech_tab3:
        st.markdown("#### Expanding-Window Walk-Forward Cross Validation")
        if st.button("Run Walk-Forward CV Benchmark Suite"):
            with st.spinner("Executing time-series cross validation across models..."):
                try:
                    feat_cols = model_meta.get("feature_cols", [])
                    bench_report = run_walk_forward_benchmark(
                        feature_df=df,
                        feature_cols=feat_cols,
                        n_splits=3,
                        test_horizon_days=14,
                        include_xgboost=XGB_AVAILABLE,
                    )
                    st.dataframe(bench_report.summary_df, hide_index=True, use_container_width=True)
                except Exception as b_err:
                    st.error(f"Benchmark error: {b_err}")

    # Tab 4: TreeSHAP
    with tech_tab4:
        st.markdown("#### Feature Influence Analysis (TreeSHAP)")
        if not is_shap_available():
            st.warning("SHAP package not installed. Run `pip install shap`.")
        else:
            feature_cols = model_meta.get("feature_cols", [])
            sh_col1, sh_col2 = st.columns(2)

            with sh_col1:
                st.markdown("##### Global Feature Drivers")
                try:
                    imp_df = global_feature_importance(q50_model, feature_cols, top_n=10)
                    fig_imp = go.Figure(go.Bar(
                        x=imp_df["importance"],
                        y=imp_df["feature"],
                        orientation="h",
                        marker_color="#4F6BED",
                    ))
                    fig_imp.update_layout(
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=350,
                        yaxis=dict(autorange="reversed"),
                        xaxis=dict(title="Importance (Mean |SHAP|)", showgrid=True, gridcolor="#F0F2ED"),
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                except Exception as e:
                    st.info(f"Global importance: {e}")

            with sh_col2:
                st.markdown("##### Plain-Language Decision Rationale")
                try:
                    rat = generate_rationale(
                        forecast_df=fc_df,
                        shap_df=None,
                        inventory_result=inv_res,
                        data_source="real",
                    )
                    st.info(rat)
                except Exception:
                    st.write(inv_res.explanation)

    # Tab 5: Feature Selection
    with tech_tab5:
        st.markdown("#### Automated Feature Selection & Redundancy Pruning")
        if st.button("Run Feature Selection Pipeline"):
            with st.spinner("Pruning low variance and collinear features..."):
                try:
                    feat_cols = model_meta.get("feature_cols", [])
                    X_raw = df.dropna(subset=["sales"])[feat_cols]
                    y_raw = df.dropna(subset=["sales"])["sales"].to_numpy()

                    X_pruned, fs_rep = run_automated_feature_selection(X_raw, y_raw, top_k=15)
                    st.success(f"Reduced feature space from {fs_rep.initial_feature_count} to {fs_rep.selected_feature_count} high-signal features.")

                    st.markdown(f"**Dropped Collinear Features**: `{fs_rep.dropped_collinear}`")
                    st.markdown(f"**Selected Top Features**: `{fs_rep.selected_features}`")
                except Exception as f_err:
                    st.error(f"Feature selection error: {f_err}")

    # Tab 6: Mathematical Formulations
    with tech_tab6:
        st.markdown("#### Mathematical Inventory & Procurement Formulations")
        st.latex(r"I_{pos} = \text{On-Hand} + \text{On-Order} - \text{Backorders}")
        st.latex(r"H = R + L \quad (\text{Protection Horizon})")
        st.latex(r"D_H = \sum_{t=1}^{H} Q_{50,t} \quad (\text{Median Cumulative Demand})")
        st.latex(r"\sigma_t = \frac{Q_{90,t} - Q_{10,t}}{2 \times 1.2816} \approx \frac{Q_{90,t} - Q_{10,t}}{2.564}")
        st.latex(r"\sigma_H = \sqrt{\sum_{t=1}^{H} \sigma_t^2} \quad (\text{Protection Uncertainty})")
        st.latex(r"s = D_L + z_p \cdot \sigma_L \quad (\text{Dynamic Reorder Point})")
        st.latex(r"S = D_H + z_p \cdot \sigma_H \quad (\text{Target Order-Up-To Level})")
        st.latex(r"P(\text{Stockout}) = 1 - \Phi\left(\frac{I_{pos} - D_H}{\sigma_H}\right)")
        st.latex(r"E[\text{Excess}] = (I_{pos} - D_H)\Phi(z) + \sigma_H \phi(z)")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 10: Settings & REST API Microservice
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Settings & API Microservice":
    st.markdown('<div class="sw-page-title">Settings & REST API Integration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Configure application settings and integrate with external ERP/POS systems via FastAPI.</div>', unsafe_allow_html=True)

    with st.form("settings_form"):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            sel_st = st.selectbox(
                "Select Store Location",
                options=stores,
                index=stores.index(st.session_state.sel_store) if st.session_state.sel_store in stores else 0,
            )
        with col_c2:
            sel_fm = st.selectbox(
                "Select Product Category",
                options=families,
                index=families.index(st.session_state.sel_family) if st.session_state.sel_family in families else 0,
            )

        save_btn = st.form_submit_button("Apply Selection", type="primary")
        if save_btn:
            st.session_state.sel_store = sel_st
            st.session_state.sel_family = sel_fm
            st.success("Target store and category updated!")
            st.rerun()

    st.markdown("---")
    st.markdown("### Production REST API Microservice (`api.py`)")

    api_t1, api_t2, api_t3 = st.tabs(["Endpoints Summary", "Example cURL Commands", "Model Retrain"])
    with api_t1:
        st.markdown("""
        | HTTP Method | Endpoint | Description |
        | :--- | :--- | :--- |
        | `GET` | `/health` | Service health status and uptime |
        | `GET` | `/catalog` | Available stores and product family catalogs |
        | `GET` | `/segmentation` | ABC-XYZ product portfolio 9-box classification |
        | `POST` | `/forecast` | Probabilistic multi-day quantile demand predictions |
        | `POST` | `/ensemble-forecast` | Multi-model hybrid ensemble (LightGBM/XGB/CatBoost) |
        | `POST` | `/replenishment` | Replenishment recommendation ($S, Q$) and stockout risk |
        | `POST` | `/probabilistic-replenishment` | Tail stockout probability, expected fill rate & excess units |
        | `POST` | `/drift-audit` | Population Stability Index (PSI) and covariate drift audit |
        | `POST` | `/purchase-order` | Generates supplier PO with pack rounding & MOQ constraints |
        | `POST` | `/simulate` | Historical discrete-event policy financial backtesting |
        """)
    with api_t2:
        st.markdown("##### Query Probabilistic Replenishment via cURL:")
        st.code("""curl -X POST "http://localhost:8000/probabilistic-replenishment" \\
  -H "Content-Type: application/json" \\
  -d '{"store_nbr": 3, "family": "BEVERAGES", "on_hand": 85, "review_period": 7, "lead_time": 3, "target_service_level": 0.95}'""", language="bash")
        st.markdown("##### Run Statistical Drift Audit:")
        st.code("""curl -X POST "http://localhost:8000/drift-audit" """, language="bash")
    with api_t3:
        st.markdown("Retrain LightGBM forecasting models across all historical store categories:")
        if st.button("Retrain Models Now"):
            with st.spinner("Retraining LightGBM quantile models..."):
                run_training_pipeline(df)
                st.cache_resource.clear()
                st.success("Models retrained and refreshed successfully!")
                st.rerun()
