"""
app.py — AI-Based Retail Demand Forecasting & Replenishment Decision System.
"Enterprise-Grade Retail Demand Forecasting, Conformal Uncertainty Calibration, and Procurement Decision Microservice."

Commercial & Shopkeeper-friendly interface maintaining:
- Machine Learning Forecasting (LightGBM & XGBoost Quantile Regressors)
- Finite-Sample Conformalized Quantile Regression (CQR) Calibration
- Deterministic Multi-Echelon Inventory Replenishment Logic
- Supplier Purchase Order Generation (Case-pack rounding, MOQ, ReportLab PDF / CSV)
- Historical Inventory Policy Simulator & Financial Backtesting
- Custom Merchant POS Data Ingestion & Health Auditor
- Game-Theoretic TreeSHAP Explainability & Rationale Generation
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
        max-width: 1160px;
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
        gap: 0.2rem !important;
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

    # Fallback to demo sample
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
# Session State Initialization (Persistent across screens)
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

# Commercial PO Settings in Session State
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
        <div style="font-size: 0.8rem; color: #70796F; font-weight: 500; margin-top: 0.2rem;">Commercial Inventory Decision Engine</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    nav_options = [
        "Home",
        "Inventory",
        "Order Suggestions & PO",
        "Policy Simulator",
        "Custom Store Upload",
        "Sales Trends",
        "Advanced Analytics",
        "Settings & API Microservice",
    ]
    
    # Custom Radio navigation with icons
    def nav_format(opt):
        icons = {
            "Home": "🏠 Home",
            "Inventory": "📦 Inventory",
            "Order Suggestions & PO": "📋 Order Suggestions & PO",
            "Policy Simulator": "⚡ Policy Simulator",
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
    
    # Active Selection Summary
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
# Shared Computations (Forecast & Inventory Engine)
# ══════════════════════════════════════════════════════════════════════════════

sel_store = st.session_state.sel_store
sel_family = st.session_state.sel_family

# Calculate protection horizon & forecast
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

    inv_inputs = InventoryInputs(
        on_hand=st.session_state.on_hand,
        on_order=st.session_state.on_order,
        backorders=st.session_state.backorders,
        review_period=st.session_state.review_period,
        lead_time=st.session_state.lead_time,
        service_level=st.session_state.service_level,
    )
    inv_res = compute_inventory(inv_inputs, fc_df)
    
    # 7-day predicted volume estimate for summary cards
    seven_day_demand = float(fc_df["pred_q50"].iloc[:7].sum())

    # Days of inventory remaining
    daily_rate = max(0.1, inv_res.median_demand_H / max(1, inv_res.protection_horizon))
    days_remaining = round(st.session_state.on_hand / daily_rate, 1)
    
except Exception as err:
    st.error(f"Error computing store forecasts: {err}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1: Home — "Your Shop at a Glance"
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.nav_page == "Home":
    st.markdown('<div class="sw-page-title">Operational Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Real-time inventory posture, upcoming demand, and procurement action items.</div>', unsafe_allow_html=True)

    # Compact selection summary
    st.markdown(f"""
    <div style="background-color: #FFFFFF; border: 1px solid #E4E9E2; border-radius: 10px; padding: 0.75rem 1.1rem; margin-bottom: 1.25rem; font-size: 0.9rem; display: flex; align-items: center; justify-content: space-between;">
        <div>
            <span style="color: #70796F;">Active selection:</span> 
            <strong>Store {sel_store}</strong> — <strong>{sel_family}</strong>
        </div>
        <div style="color: #70796F; font-size: 0.82rem;">
            Protection Horizon: <strong>{inv_res.protection_horizon} Days</strong> (Review: {st.session_state.review_period}d + Lead: {st.session_state.lead_time}d)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Summary Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        needs_order = inv_res.order_quantity > 0
        status_badge = '<span class="sw-badge-orange">Needs Reorder</span>' if needs_order else '<span class="sw-badge-green">Stock Healthy</span>'
        review_text = "Order recommended" if needs_order else "Target level met"
        st.markdown(f"""
        <div class="sw-metric-box">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="sw-metric-label">Replenishment Status</div>
                {status_badge}
            </div>
            <div class="sw-metric-val">{inv_res.order_quantity:,.0f} <span style="font-size: 1rem; font-weight: 500; color: #70796F;">units</span></div>
            <div class="sw-metric-sub">{review_text} for Store {sel_store}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Current Stock on Hand</div>
            <div class="sw-metric-val">{st.session_state.on_hand:,.0f} <span style="font-size: 1rem; font-weight: 500; color: #70796F;">units</span></div>
            <div class="sw-metric-sub">Physical shelf inventory</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        # Days of Supply Remaining
        cov_badge = '<span class="sw-badge-orange">Low</span>' if days_remaining <= st.session_state.lead_time else '<span class="sw-badge-green">Adequate</span>'
        st.markdown(f"""
        <div class="sw-metric-box">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="sw-metric-label">Days of Supply Left</div>
                {cov_badge}
            </div>
            <div class="sw-metric-val">{days_remaining:.1f} <span style="font-size: 1rem; font-weight: 500; color: #70796F;">days</span></div>
            <div class="sw-metric-sub">Lead time is {st.session_state.lead_time} days</div>
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

    # Action Banner
    if inv_res.order_quantity > 0:
        action_card_class = "sw-card-highlight"
        action_title = "PROCUREMENT ACTION REQUIRED"
        action_headline = f"Order ~{inv_res.order_quantity:,.0f} units of {sel_family}"
        action_desc = f"Target stock position is {inv_res.target_level_S:,.0f} units (covers {inv_res.median_demand_H:,.0f} demand + {inv_res.safety_stock} safety buffer). Current position is {inv_res.net_inventory_position:,.0f} units."
    else:
        action_card_class = "sw-card-highlight"
        action_title = "INVENTORY HEALTHY"
        action_headline = "No additional stock order needed right now"
        action_desc = f"Your current net inventory ({inv_res.net_inventory_position:,.0f} units) already meets target level ({inv_res.target_level_S:,.0f} units). No reorder is suggested for this cycle."

    st.markdown(f"""
    <div class="{action_card_class}">
        <div class="sw-action-title">{action_title}</div>
        <div class="sw-action-main">{action_headline}</div>
        <div class="sw-action-desc">{action_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    btn_col1, btn_col2, _ = st.columns([2, 2, 3])
    with btn_col1:
        if st.button("Generate Purchase Order (PO) →", type="primary", use_container_width=True):
            st.session_state.nav_page = "Order Suggestions & PO"
            st.rerun()
    with btn_col2:
        if st.button("Run Policy Simulation →", use_container_width=True):
            st.session_state.nav_page = "Policy Simulator"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 7-Day Demand Preview Chart
    st.markdown("### Expected Demand Preview (7 Days)")
    st.caption("Green depicts recent observed sales; blue illustrates upcoming probabilistic quantile demand range (Q10 to Q90).")

    hist_tail = df[(df["store_nbr"] == sel_store) & (df["family"] == sel_family)].sort_values("date").tail(21)
    fc_preview = fc_df.iloc[:7]

    fig_prev = go.Figure()
    fig_prev.add_trace(go.Scatter(
        x=hist_tail["date"],
        y=hist_tail["sales"],
        mode="lines",
        name="Recent Observed Sales",
        line=dict(color="#287A50", width=2.2),
    ))
    fig_prev.add_trace(go.Scatter(
        x=pd.concat([fc_preview["date"], fc_preview["date"][::-1]]),
        y=pd.concat([fc_preview["pred_q90"], fc_preview["pred_q10"][::-1]]),
        fill="toself",
        fillcolor="rgba(79, 107, 237, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Uncertainty Band [Q10, Q90]",
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
# SCREEN 2: Inventory — Stock Register & Settings
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Inventory":
    st.markdown('<div class="sw-page-title">Inventory Register & Parameter Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Inspect inventory health, configure replenishment cycles, and adjust service levels.</div>', unsafe_allow_html=True)

    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">On Hand (Physical)</div>
            <div class="sw-metric-val">{st.session_state.on_hand:,.0f}</div>
            <div class="sw-metric-sub">Units available in warehouse</div>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">In Transit (On Order)</div>
            <div class="sw-metric-val">{st.session_state.on_order:,.0f}</div>
            <div class="sw-metric-sub">Arriving from supplier</div>
        </div>
        """, unsafe_allow_html=True)
    with i3:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Customer Backorders</div>
            <div class="sw-metric-val">{st.session_state.backorders:,.0f}</div>
            <div class="sw-metric-sub">Committed / unfulfilled units</div>
        </div>
        """, unsafe_allow_html=True)
    with i4:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Net Inventory Position</div>
            <div class="sw-metric-val" style="color: #287A50;">{inv_res.net_inventory_position:,.0f}</div>
            <div class="sw-metric-sub">On-hand + On-order − Backorders</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Update Inventory Parameters")
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
                "Outstanding backorders",
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
                options=[0.90, 0.95, 0.99],
                value=float(st.session_state.service_level),
                format_func=lambda x: f"{int(x*100)}% Service Level",
            )

        submit_btn = st.form_submit_button("Save & Recalculate", type="primary")
        if submit_btn:
            st.session_state.on_hand = new_on_hand
            st.session_state.on_order = new_on_order
            st.session_state.backorders = new_backorders
            st.session_state.review_period = new_review
            st.session_state.lead_time = new_lead
            st.session_state.service_level = new_svc
            st.success("Parameters updated successfully!")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3: Order Suggestions & Purchase Order Generator
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Order Suggestions & PO":
    st.markdown('<div class="sw-page-title">Order Suggestions & Supplier PO Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Algorithmic replenishment transformed into commercial, supplier-ready purchase orders.</div>', unsafe_allow_html=True)

    order_qty = inv_res.order_quantity
    
    # Mathematical summary card
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
            Target level <em>S</em> = <strong>{inv_res.target_level_S:,.0f} units</strong> (Expected demand <strong>{inv_res.median_demand_H:,.0f}</strong> + Safety stock <strong>{inv_res.safety_stock}</strong>). Net position is <strong>{inv_res.net_inventory_position:,.0f} units</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Commercial Constraints Setup
    st.markdown("### Commercial Procurement Constraints")
    st.caption("Suppliers distribute goods in master packs and require Minimum Order Quantities (MOQs).")

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

    # Build Official PO
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

    # Detailed PO Table
    po_df = export_po_to_dataframe(po)
    st.dataframe(po_df, hide_index=True, use_container_width=True)

    # Download Buttons
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
# SCREEN 4: Policy Simulator & Financial Backtesting
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Policy Simulator":
    st.markdown('<div class="sw-page-title">Historical Inventory Policy Simulator & Financial Backtesting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Backtest StockWise Dynamic Quantile against Static Min-Max (s, S) and Moving Average heuristics over historical timelines.</div>', unsafe_allow_html=True)

    # Subseries history
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

            # Daily On-Hand Inventory Trajectory Plot
            st.markdown("### Daily On-Hand Stock Trajectory (Side-by-Side)")
            fig_sim = go.Figure()

            policy_colors = {
                PolicyType.STOCKWISE_DYNAMIC: "#287A50", # Forest green
                PolicyType.STATIC_MIN_MAX: "#4F6BED",    # Blue
                PolicyType.MOVING_AVERAGE: "#D99024",    # Amber
            }

            for p_type, run_res in sim_results.items():
                fig_sim.add_trace(go.Scatter(
                    x=run_res.daily_log["date"],
                    y=run_res.daily_log["on_hand_end"],
                    mode="lines",
                    name=p_type.value,
                    line=dict(color=policy_colors.get(p_type, "#555555"), width=2.2),
                ))

            # Stockout line at 0
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
# SCREEN 5: Custom Store Upload (Merchant POS / ERP Ingestion)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Custom Store Upload":
    st.markdown('<div class="sw-page-title">Custom Store / Merchant POS Ingestion</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Upload your own Point-of-Sale (POS) or inventory CSV/Excel exports for automated forecasting and replenishment planning.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Sales / Transaction File (.csv, .xlsx)",
        type=["csv", "xlsx", "xls"],
        help="Upload sales transactions with date, item/family, and quantity/sales.",
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

            if report.is_valid and len(reg_df) > 0:
                cust_fam = st.selectbox("Select Item to Forecast from Upload", reg_df["family"].unique())
                cust_sub = reg_df[reg_df["family"] == cust_fam].sort_values("date")
                
                # Fast baseline forecast
                rolling_7_mean = float(cust_sub["sales"].tail(7).mean())
                cust_fc_qty = max(0, int(rolling_7_mean * 7))

                st.success(f"Calculated 7-day baseline demand for **{cust_fam}**: ~**{cust_fc_qty} units** based on your trailing 7-day average of {rolling_7_mean:.1f} units/day.")

        except Exception as e:
            st.error(f"Failed to process custom file: {e}")
    else:
        st.info("💡 **Ready for your store's data**: Simply export your recent transactions or daily sales ledger from Square, Shopify, Lightspeed, or SAP and upload it here.")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 6: Sales Trends
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

    avg_sales = series_df["sales"].mean()
    max_sales = series_df["sales"].max()
    recent_trend = "increasing" if series_df["sales"].tail(7).mean() > series_df["sales"].tail(14).mean() else "stable or decreasing"

    ob1, ob2, ob3 = st.columns(3)
    with ob1:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Average Daily Demand</div>
            <div class="sw-metric-val">{avg_sales:.1f} <span style="font-size: 0.9rem; font-weight: 500; color: #70796F;">units/day</span></div>
            <div class="sw-metric-sub">Calculated across recorded store history</div>
        </div>
        """, unsafe_allow_html=True)
    with ob2:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Peak Observed Sales</div>
            <div class="sw-metric-val">{max_sales:.1f} <span style="font-size: 0.9rem; font-weight: 500; color: #70796F;">units</span></div>
            <div class="sw-metric-sub">Highest daily volume recorded</div>
        </div>
        """, unsafe_allow_html=True)
    with ob3:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Recent Trend Direction</div>
            <div class="sw-metric-val" style="font-size: 1.35rem; text-transform: capitalize; color: #287A50;">{recent_trend}</div>
            <div class="sw-metric-sub">Past 7 days vs past 14 days</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 7: Advanced Analytics & Model Verification
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Advanced Analytics":
    st.markdown('<div class="sw-page-title">Advanced Analytics & Model Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Conformal prediction intervals, multi-model cross-validation, TreeSHAP attributions, and mathematical foundations.</div>', unsafe_allow_html=True)

    tech_tab1, tech_tab2, tech_tab3, tech_tab4, tech_tab5 = st.tabs([
        "🎯 Conformal Prediction (CQR)",
        "🏁 Model Benchmarks (Walk-Forward CV)",
        "🔬 Explainable AI (TreeSHAP)",
        "📊 Holdout Metrics",
        "📐 Mathematical Formulations",
    ])

    # Tab 1: Conformalized Quantile Regression
    with tech_tab1:
        st.markdown("#### Conformalized Quantile Regression (Romano et al., 2019)")
        st.markdown("""
        Standard quantile regression models often suffer from empirical coverage gaps (e.g., nominal 80% intervals covering only 30%–60% of real observations) due to unmodeled error skewness.
        **Conformal Prediction** calculates non-conformity residuals on held-out calibration data to guarantee exact finite-sample prediction coverage.
        """)

        # Compute calibration on validation holdout
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
                st.metric("Raw LightGBM Empirical Coverage", f"{cqr_res.raw_coverage*100:.1f}%")
            with cq3:
                st.metric("Conformal Calibrated Coverage", f"{cqr_res.calibrated_coverage*100:.1f}%", delta=f"{(cqr_res.calibrated_coverage - cqr_res.raw_coverage)*100:+.1f}%")

            st.info(f"Conformal non-conformity adjustment parameter: **Δ = {cqr_res.conformal_adjustment:.2f} units**. Prediction intervals are adaptively expanded to guarantee finite-sample coverage.")

            # Calibrate current forecast
            calibrated_fc = calibrate_forecast_dataframe(fc_df, cqr_res.conformal_adjustment)
            st.markdown("##### Calibrated Multi-Step Forecast (Current Store Category)")
            st.dataframe(
                calibrated_fc[["date", "pred_q10", "pred_q10_calib", "pred_q50", "pred_q90", "pred_q90_calib", "interval_width_calib"]].head(7),
                hide_index=True,
                use_container_width=True,
            )

    # Tab 2: Walk-Forward CV Benchmarking
    with tech_tab2:
        st.markdown("#### Expanding-Window Walk-Forward Cross Validation")
        st.caption("Out-of-sample benchmark comparing LightGBM vs XGBoost vs Seasonal Naive vs Rolling Mean over rolling time-series horizons.")

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
                    st.success(f"Benchmarking completed across {bench_report.n_folds} chronological folds of {bench_report.test_horizon_days} days each.")
                except Exception as b_err:
                    st.error(f"Benchmark error: {b_err}")
        else:
            st.markdown("""
            Click above to benchmark all model candidates on:
            - **WAPE (%)**: Weighted Absolute Percentage Error (sales-weighted accuracy)
            - **MAE & RMSE**: Error magnitudes
            - **Pinball Loss**: Quantile calibration loss
            - **80% Coverage**: Empirical interval hit rate
            - **Inference Latency**: Milliseconds per prediction
            """)

    # Tab 3: TreeSHAP
    with tech_tab3:
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
                st.markdown("##### Local Driver Breakdown (Next Day)")
                try:
                    from src.features import build_future_features
                    last_date = df[(df["store_nbr"] == sel_store) & (df["family"] == sel_family)]["date"].max()
                    next_day = pd.date_range(last_date + pd.Timedelta(days=1), periods=1)
                    feat_row = build_future_features(
                        history_df=df,
                        future_dates=next_day,
                        store_nbr=sel_store,
                        family=sel_family,
                        store_map=model_meta.get("store_map"),
                        family_map=model_meta.get("family_map"),
                    )
                    X_row = feat_row.reindex(columns=feature_cols)
                    loc_df = local_shap_explanation(q50_model, X_row, feature_cols)

                    if loc_df is not None and not loc_df.empty:
                        top_loc = loc_df.head(10).copy()
                        bar_colors = ["#D94F45" if v > 0 else "#4F6BED" for v in top_loc["shap_value"]]
                        fig_loc = go.Figure(go.Bar(
                            x=top_loc["shap_value"],
                            y=top_loc["feature"],
                            orientation="h",
                            marker_color=bar_colors,
                        ))
                        fig_loc.update_layout(
                            paper_bgcolor="#FFFFFF",
                            plot_bgcolor="#FFFFFF",
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=350,
                            yaxis=dict(autorange="reversed"),
                            xaxis=dict(title="SHAP Impact (Red = Increases Forecast)", showgrid=True, gridcolor="#F0F2ED"),
                        )
                        st.plotly_chart(fig_loc, use_container_width=True)
                except Exception as e:
                    st.info(f"Local SHAP status: {e}")

            st.markdown("---")
            st.markdown("##### Plain-Language Model Rationale")
            try:
                rat = generate_rationale(
                    forecast_df=fc_df,
                    shap_df=loc_df if "loc_df" in locals() and loc_df is not None else None,
                    inventory_result=inv_res,
                    data_source="real",
                )
                st.info(rat)
            except Exception:
                st.write(inv_res.explanation)

    # Tab 4: Holdout Metrics
    with tech_tab4:
        st.markdown("#### Validation Metrics (Chronological Holdout Split)")
        if "metrics" in model_meta:
            mdf = format_metrics_table(model_meta["metrics"])
            st.dataframe(mdf, hide_index=True, use_container_width=True)

    # Tab 5: Mathematical Logic
    with tech_tab5:
        st.markdown("#### Mathematical Inventory & Procurement Formulations")
        st.latex(r"I_{pos} = \text{On-Hand} + \text{On-Order} - \text{Backorders}")
        st.latex(r"H = R + L \quad (\text{Protection Horizon})")
        st.latex(r"D_H = \sum_{t=1}^{H} Q_{50,t} \quad (\text{Median Cumulative Demand})")
        st.latex(r"\sigma_t = \frac{Q_{90,t} - Q_{10,t}}{2 \times 1.2816} \approx \frac{Q_{90,t} - Q_{10,t}}{2.564}")
        st.latex(r"\sigma_H = \sqrt{\sum_{t=1}^{H} \sigma_t^2} \quad (\text{Protection Uncertainty})")
        st.latex(r"SS = \max\left(0, \lceil z_p \cdot \sigma_H \rceil\right) \quad (\text{Safety Stock})")
        st.latex(r"S = D_H + SS \quad (\text{Target Order-Up-To Level})")
        st.latex(r"Q_{algo} = \max\left(0, \lceil S - I_{pos} \rceil\right)")
        st.latex(r"Q_{cases} = \left\lceil \frac{\max(Q_{algo}, MOQ)}{\text{PackSize}} \right\rceil, \quad Q_{commercial} = Q_{cases} \times \text{PackSize}")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 8: Settings & REST API Microservice
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
    st.markdown("""
    The system includes an enterprise-ready FastAPI microservice to connect with external Warehouse Management Systems (WMS), Enterprise Resource Planning (ERP), and Point-of-Sale (POS) registers.
    """)

    api_t1, api_t2, api_t3 = st.tabs(["Endpoints Summary", "Example cURL Commands", "Model Retrain"])
    with api_t1:
        st.markdown("""
        | HTTP Method | Endpoint | Description |
        | :--- | :--- | :--- |
        | `GET` | `/health` | Service health status and uptime |
        | `GET` | `/catalog` | Available stores and product family catalogs |
        | `POST` | `/forecast` | Probabilistic multi-day quantile demand predictions |
        | `POST` | `/replenishment` | Replenishment recommendation ($S, Q$) and stockout risk |
        | `POST` | `/purchase-order` | Generates supplier PO with pack rounding & MOQ constraints |
        | `POST` | `/simulate` | Historical discrete-event policy financial backtesting |
        """)
    with api_t2:
        st.markdown("##### Run Server:")
        st.code("uvicorn api:app --host 0.0.0.0 --port 8000 --reload", language="bash")
        st.markdown("##### Query Replenishment via cURL:")
        st.code("""curl -X POST "http://localhost:8000/replenishment" \\
  -H "Content-Type: application/json" \\
  -d '{"store_nbr": 3, "family": "BEVERAGES", "on_hand": 120, "review_period": 7, "lead_time": 3}'""", language="bash")
    with api_t3:
        st.markdown("Retrain LightGBM forecasting models across all historical store categories:")
        if st.button("Retrain Models Now"):
            with st.spinner("Retraining LightGBM quantile models..."):
                run_training_pipeline(df)
                st.cache_resource.clear()
                st.success("Models retrained and refreshed successfully!")
                st.rerun()
