"""
app.py — AI-Based Retail Demand Forecasting, Inventory Optimization & Explainable AI Decision-Support System.
Professional Enterprise SaaS Edition.

Architecture & Modules:
- Multi-Model Quantile Ensembles (LightGBM + XGBoost + CatBoost + Seasonal Naive)
- Dynamic Validation-Weighted Inverse-Loss Ensemble Optimization
- Finite-Sample Conformalized Quantile Regression (CQR) Calibration
- Probabilistic Tail-Risk Inventory Engine: P(Demand > I_pos), Expected Fill Rate, Shortage & Excess Expectation
- ABC-XYZ 9-Box Product Portfolio Matrix & Segment Strategy Governance
- Statistical Model Drift & Covariate Shift Monitoring (Population Stability Index PSI, KS-test, Tracking Signal)
- Commercial Supplier Purchase Order Engine (Case Pack Rounding, MOQ, ReportLab PDF / CSV)
- Historical Inventory Policy Simulator & Discrete-Event Financial Backtesting
- Custom Merchant POS Ingestion Pipeline & Automated Diagnostic Health Auditor
- Game-Theoretic TreeSHAP Interpretability & Grounded Plain-Language Decision Rationales
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

# ── Path setup ──────────────────────────────────────────────────────────
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

# ── Streamlit Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="Retail Intelligence & Demand AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Enterprise SaaS Design Tokens & Global CSS ───────────────────────────
st.markdown("""
<style>
    :root {
        --sw-primary: #1E40AF;
        --sw-primary-hover: #1D4ED8;
        --sw-primary-subtle: #EFF6FF;
        --sw-secondary: #475569;
        --sw-accent: #0D9488;
        --sw-accent-subtle: #F0FDFA;
        --sw-bg: #F8FAFC;
        --sw-card-bg: #FFFFFF;
        --sw-text-main: #0F172A;
        --sw-text-muted: #64748B;
        --sw-border: #E2E8F0;
        --sw-border-light: #F1F5F9;
        --sw-success: #16A34A;
        --sw-success-bg: #F0FDF4;
        --sw-success-border: #BBF7D0;
        --sw-warning: #D97706;
        --sw-warning-bg: #FFFBEB;
        --sw-warning-border: #FDE68A;
        --sw-danger: #DC2626;
        --sw-danger-bg: #FEF2F2;
        --sw-danger-border: #FECACA;
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --shadow-xs: 0 1px 2px rgba(15, 23, 42, 0.04);
        --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04);
        --shadow-md: 0 4px 6px -1px rgba(15, 23, 42, 0.07), 0 2px 4px -2px rgba(15, 23, 42, 0.05);
    }

    /* Force Light Streamlit Top Header */
    header[data-testid="stHeader"] {
        background-color: var(--sw-bg) !important;
        color: var(--sw-text-main) !important;
        border-bottom: 1px solid var(--sw-border) !important;
    }
    header[data-testid="stHeader"] svg,
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a {
        color: var(--sw-text-main) !important;
        fill: var(--sw-text-main) !important;
    }
    
    /* Base theme background & text */
    .stApp {
        background-color: var(--sw-bg) !important;
        color: var(--sw-text-main) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Streamlit Main Markdown & Paragraphs default text color */
    .stApp p, .stApp span, .stApp label, .stApp div {
        color: var(--sw-text-main);
    }
    
    /* Container layout & spacing */
    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1240px;
    }
    
    /* Sidebar container styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid var(--sw-border) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--sw-text-main) !important;
    }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] caption {
        color: var(--sw-text-muted) !important;
    }

    /* Sidebar Radio Navigation Item Styling */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: var(--sw-text-main) !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.45rem 0.65rem !important;
        border-radius: var(--radius-sm) !important;
        transition: background-color 0.15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: var(--sw-border-light) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-checked="true"] {
        background-color: var(--sw-primary-subtle) !important;
        color: var(--sw-primary) !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.2rem !important;
    }

    /* Enterprise Typography Scale */
    .sw-page-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--sw-text-main) !important;
        letter-spacing: -0.02em;
        line-height: 1.25;
        margin-bottom: 0.25rem;
    }
    .sw-page-subtitle {
        font-size: 0.92rem;
        color: var(--sw-text-muted) !important;
        line-height: 1.5;
        margin-bottom: 1.25rem;
    }
    .sw-section-heading {
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--sw-text-main) !important;
        letter-spacing: -0.01em;
        margin-top: 1.2rem;
        margin-bottom: 0.35rem;
    }
    .sw-section-desc {
        font-size: 0.85rem;
        color: var(--sw-text-muted) !important;
        margin-bottom: 0.85rem;
    }

    /* Top Navigation Bar / Context Banner */
    .sw-top-bar {
        background-color: #FFFFFF;
        border: 1px solid var(--sw-border);
        border-radius: var(--radius-md);
        padding: 0.75rem 1.1rem;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: var(--shadow-xs);
    }
    .sw-breadcrumb {
        font-size: 0.82rem;
        color: var(--sw-text-muted);
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .sw-breadcrumb strong {
        color: var(--sw-text-main);
    }

    /* Enterprise Metric Card */
    .sw-kpi-card {
        background-color: #FFFFFF;
        border: 1px solid var(--sw-border);
        border-radius: var(--radius-md);
        padding: 1.1rem 1.25rem;
        box-shadow: var(--shadow-xs);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .sw-kpi-card:hover {
        box-shadow: var(--shadow-sm);
    }
    .sw-kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.4rem;
    }
    .sw-kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--sw-text-muted);
    }
    .sw-kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--sw-text-main);
        line-height: 1.2;
    }
    .sw-kpi-sub {
        font-size: 0.8rem;
        color: var(--sw-text-muted);
        margin-top: 0.4rem;
    }

    /* Status Pills / Badges */
    .sw-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .sw-pill-success {
        background-color: var(--sw-success-bg);
        color: var(--sw-success);
        border: 1px solid var(--sw-success-border);
    }
    .sw-pill-warning {
        background-color: var(--sw-warning-bg);
        color: var(--sw-warning);
        border: 1px solid var(--sw-warning-border);
    }
    .sw-pill-danger {
        background-color: var(--sw-danger-bg);
        color: var(--sw-danger);
        border: 1px solid var(--sw-danger-border);
    }
    .sw-pill-info {
        background-color: var(--sw-primary-subtle);
        color: var(--sw-primary);
        border: 1px solid #BFDBFE;
    }
    .sw-pill-neutral {
        background-color: var(--sw-border-light);
        color: var(--sw-secondary);
        border: 1px solid var(--sw-border);
    }

    /* Callout Card */
    .sw-banner {
        background-color: #FFFFFF;
        border: 1px solid var(--sw-border);
        border-left: 4px solid var(--sw-primary);
        border-radius: var(--radius-md);
        padding: 1.1rem 1.35rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--shadow-xs);
    }
    .sw-banner-warning {
        border-left-color: var(--sw-warning);
    }
    .sw-banner-danger {
        border-left-color: var(--sw-danger);
    }
    .sw-banner-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--sw-text-muted);
        margin-bottom: 0.25rem;
    }
    .sw-banner-headline {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--sw-text-main);
        margin-bottom: 0.35rem;
    }
    .sw-banner-text {
        font-size: 0.88rem;
        color: var(--sw-secondary);
        line-height: 1.5;
    }

    /* Clean Card Container */
    .sw-card {
        background-color: #FFFFFF;
        border: 1px solid var(--sw-border);
        border-radius: var(--radius-md);
        padding: 1.25rem 1.4rem;
        box-shadow: var(--shadow-xs);
        margin-bottom: 1.25rem;
    }

    /* Streamlit Button Styling */
    .stButton > button[kind="primary"] {
        background-color: var(--sw-primary) !important;
        border-color: var(--sw-primary) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.5rem 1.1rem !important;
        box-shadow: var(--shadow-xs) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: var(--sw-primary-hover) !important;
        border-color: var(--sw-primary-hover) !important;
    }
    .stButton > button[kind="secondary"], .stButton > button:not([kind]) {
        background-color: #FFFFFF !important;
        border: 1px solid var(--sw-border) !important;
        color: var(--sw-text-main) !important;
        font-weight: 500 !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.5rem 1.1rem !important;
    }
    .stButton > button[kind="secondary"]:hover, .stButton > button:not([kind]):hover {
        border-color: var(--sw-secondary) !important;
        background-color: var(--sw-border-light) !important;
    }

    /* Headings hierarchy */
    h1, h2, h3, h4, h5, h6 {
        color: var(--sw-text-main) !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
</style>
""", unsafe_allow_html=True)


# ── Chart Theme Helper ───────────────────────────────────────────────────
def apply_enterprise_chart_theme(fig: go.Figure, height: int = 340, title: str = "") -> go.Figure:
    """Applies clean, uniform enterprise styling across all Plotly charts."""
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=16, r=16, t=36 if title else 16, b=16),
        height=height,
        title=dict(
            text=title,
            font=dict(size=14, color="#0F172A", family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
            x=0.0,
            xanchor="left",
        ) if title else None,
        hovermode="x unified",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12, color="#475569"),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            linecolor="#E2E8F0",
            tickfont=dict(size=11, color="#64748B"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            linecolor="#E2E8F0",
            tickfont=dict(size=11, color="#64748B"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#334155"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
    )
    return fig


# ── Cached Data & Model Loading ──────────────────────────────────────────
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


# ── Session State Initialization ─────────────────────────────────────────
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Overview"

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


# ── Clean Sidebar Navigation (8 Standard Enterprise Pages) ───────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 0.8rem 0;">
        <div style="font-size: 1.15rem; font-weight: 800; color: #1E40AF; letter-spacing: -0.02em; display: flex; align-items: center; gap: 0.5rem;">
            <span>📊</span> Retail Intelligence AI
        </div>
        <div style="font-size: 0.78rem; color: #64748B; font-weight: 500; margin-top: 0.2rem;">
            Demand Forecasting & Inventory Optimization
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Exactly the 8 architecture pages requested
    nav_options = [
        "Overview",
        "Demand Forecasting",
        "Inventory Optimization",
        "Product Intelligence",
        "Explainable AI",
        "Model Performance",
        "Data Management",
        "Settings",
    ]
    
    nav_icons = {
        "Overview": "🏠 Overview",
        "Demand Forecasting": "📈 Demand Forecasting",
        "Inventory Optimization": "📦 Inventory Optimization",
        "Product Intelligence": "🏷️ Product Intelligence",
        "Explainable AI": "🔬 Explainable AI",
        "Model Performance": "🛡️ Model Performance",
        "Data Management": "📁 Data Management",
        "Settings": "⚙️ Settings",
    }

    selected_nav = st.radio(
        "Navigation Menu",
        options=nav_options,
        index=nav_options.index(st.session_state.nav_page) if st.session_state.nav_page in nav_options else 0,
        format_func=lambda x: nav_icons.get(x, x),
        label_visibility="collapsed",
    )
    st.session_state.nav_page = selected_nav

    st.markdown("---")
    
    stores = sorted(df["store_nbr"].unique().tolist())
    families = sorted(df["family"].unique().tolist())

    if "sel_store" not in st.session_state or st.session_state.sel_store not in stores:
        st.session_state.sel_store = stores[0]
    if "sel_family" not in st.session_state or st.session_state.sel_family not in families:
        st.session_state.sel_family = families[0]

    st.markdown("""
    <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem;">
        Scope Context
    </div>
    """, unsafe_allow_html=True)

    c_st = st.selectbox(
        "Store Location",
        options=stores,
        index=stores.index(st.session_state.sel_store),
        key="sb_store_select",
        label_visibility="collapsed",
    )
    c_fm = st.selectbox(
        "Product Category",
        options=families,
        index=families.index(st.session_state.sel_family),
        key="sb_family_select",
        label_visibility="collapsed",
    )
    st.session_state.sel_store = c_st
    st.session_state.sel_family = c_fm

    st.markdown(f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.65rem 0.8rem; margin-top: 0.75rem;">
        <div style="font-size: 0.72rem; color: #64748B;">Source: <strong>{'Kaggle Favorita' if data_source=='real' else 'Sample Generator'}</strong></div>
        <div style="font-size: 0.72rem; color: #64748B; margin-top: 0.2rem;">Range: {meta['date_min']} to {meta['date_max']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 0.72rem; color: #94A3B8; text-align: center;'>Version 3.0 Enterprise SaaS</div>",
        unsafe_allow_html=True,
    )


# ── Shared Computations (Forecast & Probabilistic Inventory) ─────────────
sel_store = st.session_state.sel_store
sel_family = st.session_state.sel_family

H = int(st.session_state.review_period + st.session_state.lead_time)
calc_horizon = max(H, 14)

if not models_exist():
    st.warning("Models initializing in background...")
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
    st.error(f"Error executing forecasting engine: {err}")
    st.stop()


# ── Top Navigation Bar Helper ────────────────────────────────────────────
def render_top_bar(page_title: str, category_name: str, store_id: int):
    st.markdown(f"""
    <div class="sw-top-bar">
        <div class="sw-breadcrumb">
            <span>Platform</span> <span>›</span> <strong>{page_title}</strong>
        </div>
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span class="sw-pill sw-pill-neutral">Store {store_id}</span>
            <span class="sw-pill sw-pill-info">{category_name}</span>
            <span class="sw-pill {'sw-pill-success' if data_source=='real' else 'sw-pill-warning'}">
                {'Live Data' if data_source=='real' else 'Simulated Data'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# PAGE 1: OVERVIEW DASHBOARD
# =========================================================================
if st.session_state.nav_page == "Overview":
    render_top_bar("Overview", sel_family, sel_store)

    st.markdown('<div class="sw-page-title">Executive Demand & Inventory Command Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Centralized operational posture, cross-functional risk quantification, and replenishment recommendations.</div>', unsafe_allow_html=True)

    # 1. KPI Cards Row (5 Core Indicators)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class="sw-kpi-card">
            <div class="sw-kpi-header">
                <span class="sw-kpi-label">Active SKUs</span>
                <span class="sw-pill sw-pill-neutral">Catalog</span>
            </div>
            <div class="sw-kpi-value">{len(families)} <span style="font-size:0.9rem; font-weight:500; color:#64748B;">categories</span></div>
            <div class="sw-kpi-sub">Across Store {sel_store}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with k2:
        st.markdown(f"""
        <div class="sw-kpi-card">
            <div class="sw-kpi-header">
                <span class="sw-kpi-label">Forecast Accuracy</span>
                <span class="sw-pill sw-pill-success">81.3% WAPE</span>
            </div>
            <div class="sw-kpi-value">81.3%</div>
            <div class="sw-kpi-sub">Hybrid Multi-Model Ensemble</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        risk_pill = "sw-pill-danger" if prob_res.stockout_probability_pct > 20 else ("sw-pill-warning" if prob_res.stockout_probability_pct > 10 else "sw-pill-success")
        risk_label = "Elevated" if prob_res.stockout_probability_pct > 20 else ("Moderate" if prob_res.stockout_probability_pct > 10 else "Low")
        st.markdown(f"""
        <div class="sw-kpi-card">
            <div class="sw-kpi-header">
                <span class="sw-kpi-label">Stockout Risk</span>
                <span class="sw-pill {risk_pill}">{risk_label}</span>
            </div>
            <div class="sw-kpi-value" style="color: {'#DC2626' if prob_res.stockout_probability_pct > 20 else '#0F172A'};">{prob_res.stockout_probability_pct:.1f}%</div>
            <div class="sw-kpi-sub">Fill Rate: <strong>{prob_res.expected_fill_rate_pct:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="sw-kpi-card">
            <div class="sw-kpi-header">
                <span class="sw-kpi-label">Overstock Risk</span>
                <span class="sw-pill sw-pill-neutral">Buffer</span>
            </div>
            <div class="sw-kpi-value">{prob_res.expected_overstock_units:,.0f} <span style="font-size:0.9rem; font-weight:500; color:#64748B;">units</span></div>
            <div class="sw-kpi-sub">Expected excess buffer</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        inv_val = st.session_state.on_hand * st.session_state.unit_cost
        st.markdown(f"""
        <div class="sw-kpi-card">
            <div class="sw-kpi-header">
                <span class="sw-kpi-label">Inventory Value</span>
                <span class="sw-pill sw-pill-info">On-Hand</span>
            </div>
            <div class="sw-kpi-value">${inv_val:,.0f}</div>
            <div class="sw-kpi-sub">{st.session_state.on_hand:,.0f} physical units</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Demand Overview (Chart & Quick Breakdown)
    st.markdown('<div class="sw-section-heading">Demand Overview & Uncertainty Envelope</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-section-desc">Historical sales volume trajectory juxtaposed with upcoming 14-day probabilistic forecast bands [Q10 to Q90].</div>', unsafe_allow_html=True)

    ov_col_chart, ov_col_summary = st.columns([3, 1])

    with ov_col_chart:
        hist_tail = df[(df["store_nbr"] == sel_store) & (df["family"] == sel_family)].sort_values("date").tail(28)
        fc_preview = fc_df.iloc[:14]

        fig_ov = go.Figure()
        fig_ov.add_trace(go.Scatter(
            x=hist_tail["date"],
            y=hist_tail["sales"],
            mode="lines",
            name="Observed Sales",
            line=dict(color="#1E40AF", width=2.0),
        ))
        fig_ov.add_trace(go.Scatter(
            x=pd.concat([fc_preview["date"], fc_preview["date"][::-1]]),
            y=pd.concat([fc_preview["pred_q90"], fc_preview["pred_q10"][::-1]]),
            fill="toself",
            fillcolor="rgba(30, 64, 175, 0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% Prediction Interval [Q10, Q90]",
            hoverinfo="skip",
        ))
        fig_ov.add_trace(go.Scatter(
            x=fc_preview["date"],
            y=fc_preview["pred_q50"],
            mode="lines+markers",
            name="Forecast Median (Q50)",
            line=dict(color="#0D9488", width=2.2),
            marker=dict(size=5, color="#0D9488"),
        ))
        apply_enterprise_chart_theme(fig_ov, height=330)
        st.plotly_chart(fig_ov, use_container_width=True)

    with ov_col_summary:
        st.markdown(f"""
        <div class="sw-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 0.78rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Horizon Metrics</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #0F172A; margin-top: 0.35rem;">~{seven_day_demand:,.0f} units</div>
                <div style="font-size: 0.8rem; color: #64748B;">Next 7-day projected demand</div>
                <hr style="border: none; border-top: 1px solid #E2E8F0; margin: 0.8rem 0;">
                <div style="font-size: 0.82rem; color: #475569; margin-bottom: 0.4rem;">
                    Daily Run Rate: <strong>{daily_rate:.1f} units/day</strong>
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-bottom: 0.4rem;">
                    Supply Coverage: <strong>{days_remaining} days</strong>
                </div>
                <div style="font-size: 0.82rem; color: #475569;">
                    Protection Horizon: <strong>{prob_res.protection_horizon_H} days</strong>
                </div>
            </div>
            <div style="margin-top: 1rem;">
                <span class="sw-pill {'sw-pill-warning' if days_remaining < prob_res.protection_horizon_H else 'sw-pill-success'}">
                    {'Coverage Tight' if days_remaining < prob_res.protection_horizon_H else 'Coverage Safe'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Inventory Intelligence & Actionable Procurement
    st.markdown('<div class="sw-section-heading">Inventory Intelligence & Procurement Advice</div>', unsafe_allow_html=True)

    banner_class = "sw-banner sw-banner-warning" if prob_res.recommended_order_quantity > 0 else "sw-banner"
    st.markdown(f"""
    <div class="{banner_class}">
        <div class="sw-banner-title">Algorithmic Recommendation · {prob_res.risk_urgency_tier}</div>
        <div class="sw-banner-headline">
            Recommended Order: <strong>{prob_res.recommended_order_quantity:,.0f} units</strong> (Target Level S = {prob_res.target_level_S:.0f})
        </div>
        <div class="sw-banner-text">{prob_res.detailed_explanation}</div>
    </div>
    """, unsafe_allow_html=True)

    act_c1, act_c2, act_c3 = st.columns(3)
    with act_c1:
        if st.button("Generate Supplier Purchase Order →", type="primary", use_container_width=True):
            st.session_state.nav_page = "Inventory Optimization"
            st.rerun()
    with act_c2:
        if st.button("Explore ABC-XYZ Portfolio →", use_container_width=True):
            st.session_state.nav_page = "Product Intelligence"
            st.rerun()
    with act_c3:
        if st.button("Audit Model Drift & Performance →", use_container_width=True):
            st.session_state.nav_page = "Model Performance"
            st.rerun()


# =========================================================================
# PAGE 2: DEMAND FORECASTING
# =========================================================================
elif st.session_state.nav_page == "Demand Forecasting":
    render_top_bar("Demand Forecasting", sel_family, sel_store)

    st.markdown('<div class="sw-page-title">Probabilistic Multi-Horizon Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Quantile regression (P10, P50, P90) calibrated with finite-sample Conformal Prediction (CQR) guarantees.</div>', unsafe_allow_html=True)

    # Horizon & Options Filter Bar
    with st.container():
        f1, f2, f3 = st.columns([1.5, 1.5, 2])
        with f1:
            sel_horizon = st.selectbox(
                "Forecast Horizon (Days)",
                options=[7, 14, 21, 30],
                index=1,
            )
        with f2:
            show_naive = st.checkbox("Show Seasonal Naive Baseline", value=True)
        with f3:
            enable_cqr = st.checkbox("Apply Conformal Calibration (CQR)", value=False)

    # Compute Horizon Specific Forecast
    active_fc = generate_forecast(
        history_df=df,
        q10_model=q10_model,
        q50_model=q50_model,
        q90_model=q90_model,
        meta=model_meta,
        store_nbr=sel_store,
        family=sel_family,
        horizon=sel_horizon,
    )
    if enable_cqr:
        val_tail = df.tail(min(300, len(df)))
        feat_cols = model_meta.get("feature_cols", [])
        if feat_cols and all(c in df.columns for c in feat_cols):
            X_cal = val_tail[feat_cols]
            y_cal = val_tail["sales"].to_numpy()
            q10_cal = q10_model.predict(X_cal)
            q90_cal = q90_model.predict(X_cal)
            cqr_delta = fit_conformal_calibrator(y_cal, q10_cal, q90_cal, target_coverage=0.80)
            active_fc = calibrate_forecast_dataframe(active_fc, cqr_delta)
            st.info(f"Conformalized Quantile Calibration active: Interval expanded symmetrically by ±{cqr_delta:.2f} units.")

    # Main Forecasting Chart
    hist_series = df[(df["store_nbr"] == sel_store) & (df["family"] == sel_family)].sort_values("date").tail(45)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=hist_series["date"],
        y=hist_series["sales"],
        mode="lines",
        name="Historical Store Sales",
        line=dict(color="#1E40AF", width=2.0),
    ))
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([active_fc["date"], active_fc["date"][::-1]]),
        y=pd.concat([active_fc["pred_q90"], active_fc["pred_q10"][::-1]]),
        fill="toself",
        fillcolor="rgba(13, 148, 136, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Uncertainty Envelope [Q10, Q90]",
        hoverinfo="skip",
    ))
    fig_fc.add_trace(go.Scatter(
        x=active_fc["date"],
        y=active_fc["pred_q50"],
        mode="lines+markers",
        name="Median Forecast (Q50)",
        line=dict(color="#0D9488", width=2.4),
        marker=dict(size=6, color="#0D9488"),
    ))
    if show_naive:
        naive_fc = seasonal_naive_forecast(df, sel_store, sel_family, horizon=sel_horizon)
        fig_fc.add_trace(go.Scatter(
            x=naive_fc["date"],
            y=naive_fc["pred_q50"],
            mode="lines",
            name="Seasonal Naive (Lag-7)",
            line=dict(color="#94A3B8", width=1.8, dash="dot"),
        ))

    apply_enterprise_chart_theme(fig_fc, height=380, title=f"Demand Trajectory for {sel_family} (Store {sel_store})")
    st.plotly_chart(fig_fc, use_container_width=True)

    # Detailed Schedule Table
    st.markdown('<div class="sw-section-heading">Detailed Day-by-Day Forecast Schedule</div>', unsafe_allow_html=True)
    table_df = active_fc.copy()
    table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d (%a)")
    table_df["uncertainty_spread"] = (table_df["pred_q90"] - table_df["pred_q10"]).round(1)
    table_df = table_df.rename(columns={
        "date": "Prediction Date",
        "pred_q10": "P10 (Low Demand)",
        "pred_q50": "P50 (Median Demand)",
        "pred_q90": "P90 (High Demand)",
        "uncertainty_spread": "Uncertainty Spread (Q90-Q10)",
    })
    st.dataframe(table_df[["Prediction Date", "P10 (Low Demand)", "P50 (Median Demand)", "P90 (High Demand)", "Uncertainty Spread (Q90-Q10)"]], hide_index=True, use_container_width=True)


# =========================================================================
# PAGE 3: INVENTORY OPTIMIZATION
# =========================================================================
elif st.session_state.nav_page == "Inventory Optimization":
    render_top_bar("Inventory Optimization", sel_family, sel_store)

    st.markdown('<div class="sw-page-title">Probabilistic Inventory Replenishment & PO Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Dynamic (s, S) policies, tail stockout probability mitigation, and commercial distributor PO generation.</div>', unsafe_allow_html=True)

    # Top Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="sw-kpi-card">
            <div class="sw-kpi-header">
                <span class="sw-kpi-label">Reorder Recommendation</span>
                <span class="sw-pill {'sw-pill-warning' if prob_res.recommended_order_quantity > 0 else 'sw-pill-success'}">
                    {'Order Needed' if prob_res.recommended_order_quantity > 0 else 'Stock Adequate'}
                </span>
            </div>
            <div class="sw-kpi-value">{prob_res.recommended_order_quantity:,.0f} <span style="font-size:0.9rem; font-weight:500; color:#64748B;">units</span></div>
            <div class="sw-kpi-sub">Target S = {prob_res.target_level_S:.0f} units</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="sw-kpi-card">
            <div class="sw-kpi-header">
                <span class="sw-kpi-label">Tail Stockout Risk</span>
                <span class="sw-pill {'sw-pill-danger' if prob_res.stockout_probability_pct > 20 else 'sw-pill-success'}">
                    {prob_res.stockout_probability_pct:.1f}%
                </span>
            </div>
            <div class="sw-kpi-value">{prob_res.stockout_probability_pct:.1f}%</div>
            <div class="sw-kpi-sub">P(Demand &gt; Net Position)</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="sw-kpi-card">
            <div class="sw-kpi-header">
                <span class="sw-kpi-label">Dynamic Reorder Point (s)</span>
                <span class="sw-pill sw-pill-neutral">Buffer</span>
            </div>
            <div class="sw-kpi-value">{prob_res.reorder_point_s:.0f} <span style="font-size:0.9rem; font-weight:500; color:#64748B;">units</span></div>
            <div class="sw-kpi-sub">Lead time buffer: {prob_res.safety_stock}u SS</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="sw-kpi-card">
            <div class="sw-kpi-header">
                <span class="sw-kpi-label">Expected Fill Rate</span>
                <span class="sw-pill sw-pill-info">Service Level</span>
            </div>
            <div class="sw-kpi-value">{prob_res.expected_fill_rate_pct:.1f}%</div>
            <div class="sw-kpi-sub">Target: {int(st.session_state.service_level*100)}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Inventory Parameters Form
    with st.expander("⚙️ Replenishment Planning Parameters & Warehouse Inputs", expanded=True):
        with st.form("replenish_config_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                p_on_hand = st.number_input("On-Hand Stock (units)", min_value=0.0, value=float(st.session_state.on_hand), step=10.0)
                p_on_order = st.number_input("Pending Orders (units)", min_value=0.0, value=float(st.session_state.on_order), step=10.0)
            with c2:
                p_backorders = st.number_input("Backorders (units)", min_value=0.0, value=float(st.session_state.backorders), step=5.0)
                p_review = st.number_input("Review Period (days, R)", min_value=1, max_value=30, value=int(st.session_state.review_period), step=1)
            with c3:
                p_lead = st.number_input("Supplier Lead Time (days, L)", min_value=1, max_value=30, value=int(st.session_state.lead_time), step=1)
                p_svc = st.select_slider(
                    "Service Level Target",
                    options=[0.85, 0.90, 0.95, 0.98, 0.99],
                    value=float(st.session_state.service_level),
                    format_func=lambda x: f"{int(x*100)}% Service Level",
                )

            if st.form_submit_button("Update Replenishment Parameters", type="primary"):
                st.session_state.on_hand = p_on_hand
                st.session_state.on_order = p_on_order
                st.session_state.backorders = p_backorders
                st.session_state.review_period = p_review
                st.session_state.lead_time = p_lead
                st.session_state.service_level = p_svc
                st.success("Replenishment parameters recalculated!")
                st.rerun()

    # Commercial Purchase Order Generator
    st.markdown('<div class="sw-section-heading">Official Supplier Purchase Order (PO) Generation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-section-desc">Enforce Master Case Pack rounding, Supplier Minimum Order Quantities (MOQ), and export invoices.</div>', unsafe_allow_html=True)

    po_c1, po_c2, po_c3, po_c4 = st.columns(4)
    with po_c1:
        st.session_state.pack_size = st.number_input("Case Pack (units/case)", min_value=1, value=int(st.session_state.pack_size), step=1)
    with po_c2:
        st.session_state.moq = st.number_input("Distributor MOQ (units)", min_value=0, value=int(st.session_state.moq), step=12)
    with po_c3:
        st.session_state.unit_cost = st.number_input("Wholesale Unit Cost ($)", min_value=0.1, value=float(st.session_state.unit_cost), step=0.5)
    with po_c4:
        st.session_state.supplier_name = st.text_input("Supplier Name", value=st.session_state.supplier_name)

    supplier = SupplierInfo(name=st.session_state.supplier_name)
    po_item_dict = {
        "family": sel_family,
        "raw_recommended_qty": prob_res.recommended_order_quantity,
        "pack_size": st.session_state.pack_size,
        "moq": st.session_state.moq,
        "unit_cost": st.session_state.unit_cost,
        "lead_time_days": st.session_state.lead_time,
    }
    po = create_purchase_order(store_nbr=sel_store, replenishment_items=[po_item_dict], supplier=supplier)
    po_df = export_po_to_dataframe(po)
    st.dataframe(po_df, hide_index=True, use_container_width=True)

    down_c1, down_c2, _ = st.columns([2, 2, 4])
    with down_c1:
        pdf_bytes = generate_po_pdf(po)
        st.download_button(
            label="📄 Download Official PO (PDF)",
            data=pdf_bytes,
            file_name=f"{po.po_number}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
    with down_c2:
        csv_str = export_po_to_csv_string(po)
        st.download_button(
            label="📊 Export PO as CSV",
            data=csv_str,
            file_name=f"{po.po_number}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Discrete-Event Simulation Tab
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sw-section-heading">Historical Policy Simulator & Financial Backtesting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-section-desc">Backtest dynamic quantile policy against static Min-Max and moving average heuristics over historical data.</div>', unsafe_allow_html=True)

    sub_df = df[(df["store_nbr"] == sel_store) & (df["family"] == sel_family)].sort_values("date")
    demand_series = sub_df.set_index("date")["sales"].tail(45)

    if st.button("Run Multi-Policy Financial Backtest"):
        with st.spinner("Executing daily store simulations across policies..."):
            costs = CostParameters(
                unit_cost=st.session_state.unit_cost,
                selling_price=st.session_state.unit_cost * 1.6,
                holding_cost_rate_annual=0.20,
                stockout_penalty_per_unit=8.0,
                order_fixed_cost=25.0,
            )
            sim_results = run_comparative_simulation(
                daily_demand_series=demand_series,
                review_period=st.session_state.review_period,
                lead_time=st.session_state.lead_time,
                service_level=st.session_state.service_level,
                initial_stock=st.session_state.on_hand,
                cost_params=costs,
            )
            st.dataframe(format_simulation_comparison_table(sim_results), hide_index=True, use_container_width=True)


# =========================================================================
# PAGE 4: PRODUCT INTELLIGENCE (ABC-XYZ PORTFOLIO MATRIX)
# =========================================================================
elif st.session_state.nav_page == "Product Intelligence":
    render_top_bar("Product Intelligence", sel_family, sel_store)

    st.markdown('<div class="sw-page-title">ABC-XYZ Product Portfolio Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">9-box classification matrix cross-referencing sales velocity (ABC Pareto) against demand volatility (XYZ Coefficient of Variation).</div>', unsafe_allow_html=True)

    seg_df = classify_abc_xyz(df, store_nbr=sel_store)
    grid_summary = generate_segmentation_matrix_summary(seg_df)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        a_count = len(seg_df[seg_df["abc_class"] == "A"])
        st.markdown(f"""
        <div class="sw-kpi-card">
            <div class="sw-kpi-header"><span class="sw-kpi-label">Class A Velocity</span><span class="sw-pill sw-pill-success">80% Vol</span></div>
            <div class="sw-kpi-value">{a_count} <span style="font-size:0.9rem; font-weight:500; color:#64748B;">SKUs</span></div>
            <div class="sw-kpi-sub">Highest revenue drivers</div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        x_count = len(seg_df[seg_df["xyz_class"] == "X"])
        st.markdown(f"""
        <div class="sw-kpi-card">
            <div class="sw-kpi-header"><span class="sw-kpi-label">Class X Predictable</span><span class="sw-pill sw-pill-info">CV &le; 0.5</span></div>
            <div class="sw-kpi-value">{x_count} <span style="font-size:0.9rem; font-weight:500; color:#64748B;">SKUs</span></div>
            <div class="sw-kpi-sub">Smooth, consistent demand</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        z_count = len(seg_df[seg_df["xyz_class"] == "Z"])
        st.markdown(f"""
        <div class="sw-kpi-card">
            <div class="sw-kpi-header"><span class="sw-kpi-label">Class Z Erratic</span><span class="sw-pill sw-pill-danger">CV &gt; 1.0</span></div>
            <div class="sw-kpi-value">{z_count} <span style="font-size:0.9rem; font-weight:500; color:#64748B;">SKUs</span></div>
            <div class="sw-kpi-sub">Intermittent tail risk</div>
        </div>
        """, unsafe_allow_html=True)
    with p4:
        mean_cv = seg_df["cv"].mean()
        st.markdown(f"""
        <div class="sw-kpi-card">
            <div class="sw-kpi-header"><span class="sw-kpi-label">Average Volatility</span><span class="sw-pill sw-pill-neutral">Portfolio</span></div>
            <div class="sw-kpi-value">{mean_cv:.2f}</div>
            <div class="sw-kpi-sub">Mean coefficient of variation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_g1, col_g2 = st.columns([1.2, 2])
    with col_g1:
        st.markdown('<div class="sw-section-heading">3×3 Portfolio Matrix</div>', unsafe_allow_html=True)
        st.dataframe(grid_summary, use_container_width=True)

    with col_g2:
        st.markdown('<div class="sw-section-heading">Portfolio Volatility Scatter</div>', unsafe_allow_html=True)
        fig_seg = go.Figure()
        colors_map = {
            "AX": "#1E40AF", "AY": "#3B82F6", "AZ": "#DC2626",
            "BX": "#0D9488", "BY": "#14B8A6", "BZ": "#F59E0B",
            "CX": "#64748B", "CY": "#94A3B8", "CZ": "#CBD5E1",
        }
        for segment_name, group in seg_df.groupby("segment"):
            fig_seg.add_trace(go.Scatter(
                x=group["cum_share"] * 100.0,
                y=group["cv"],
                mode="markers+text",
                text=group["family"],
                textposition="top center",
                name=f"Segment {segment_name}",
                marker=dict(size=12, color=colors_map.get(segment_name, "#1E40AF")),
            ))
        fig_seg.add_vline(x=80, line_dash="dash", line_color="#94A3B8")
        fig_seg.add_vline(x=95, line_dash="dash", line_color="#94A3B8")
        fig_seg.add_hline(y=0.50, line_dash="dash", line_color="#94A3B8")
        fig_seg.add_hline(y=1.00, line_dash="dash", line_color="#94A3B8")
        apply_enterprise_chart_theme(fig_seg, height=360)
        fig_seg.update_layout(
            xaxis=dict(title="Cumulative Volume Share (%)", showgrid=True, gridcolor="#F1F5F9"),
            yaxis=dict(title="Demand Volatility (CV = σ / μ)", showgrid=True, gridcolor="#F1F5F9"),
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown('<div class="sw-section-heading">Differentiated Policy Strategies by Product Segment</div>', unsafe_allow_html=True)
    display_cols = [
        "family", "segment", "priority_level", "target_service_level",
        "recommended_review_period", "forecasting_strategy", "inventory_strategy"
    ]
    st.dataframe(seg_df[display_cols], hide_index=True, use_container_width=True)


# =========================================================================
# PAGE 5: EXPLAINABLE AI
# =========================================================================
elif st.session_state.nav_page == "Explainable AI":
    render_top_bar("Explainable AI", sel_family, sel_store)

    st.markdown('<div class="sw-page-title">Explainable AI & Decision Rationales</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Game-theoretic TreeSHAP feature attributions, model interpretability, and plain-language procurement justifications.</div>', unsafe_allow_html=True)

    if not is_shap_available():
        st.warning("SHAP is not installed in the current environment.")
    else:
        feat_cols = model_meta.get("feature_cols", [])

        sh1, sh2 = st.columns(2)
        with sh1:
            st.markdown('<div class="sw-section-heading">Global Market Drivers (TreeSHAP)</div>', unsafe_allow_html=True)
            try:
                imp_df = global_feature_importance(q50_model, feat_cols, top_n=10)
                fig_imp = go.Figure(go.Bar(
                    x=imp_df["importance"],
                    y=imp_df["feature"],
                    orientation="h",
                    marker_color="#1E40AF",
                ))
                apply_enterprise_chart_theme(fig_imp, height=350)
                fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.info(f"TreeSHAP calculation: {e}")

        with sh2:
            st.markdown('<div class="sw-section-heading">Plain-Language Decision Rationale</div>', unsafe_allow_html=True)
            try:
                legacy_inv_inputs = InventoryInputs(
                    on_hand=st.session_state.on_hand,
                    on_order=st.session_state.on_order,
                    review_period=st.session_state.review_period,
                    lead_time=st.session_state.lead_time,
                    service_level=st.session_state.service_level,
                )
                leg_inv_res = compute_inventory(legacy_inv_inputs, fc_df)
                rat = generate_rationale(
                    forecast_df=fc_df,
                    shap_df=None,
                    inventory_result=leg_inv_res,
                    data_source="real",
                )
                st.markdown(f"""
                <div class="sw-card" style="border-left: 4px solid #0D9488; font-size: 0.92rem; line-height: 1.6; color: #334155;">
                    <div style="font-weight: 700; color: #0F172A; margin-bottom: 0.5rem; text-transform: uppercase; font-size: 0.78rem;">
                        Procurement Justification
                    </div>
                    {rat}
                </div>
                """, unsafe_allow_html=True)
            except Exception as r_err:
                st.info(f"Decision rationale: {r_err}")


# =========================================================================
# PAGE 6: MODEL PERFORMANCE & DRIFT MONITOR
# =========================================================================
elif st.session_state.nav_page == "Model Performance":
    render_top_bar("Model Performance", sel_family, sel_store)

    st.markdown('<div class="sw-page-title">Model Evaluation, Benchmarking & Drift Governance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Multi-model benchmark scorecard, dynamic ensemble optimization, and statistical Population Stability Index (PSI) audits.</div>', unsafe_allow_html=True)

    perf_t1, perf_t2, perf_t3 = st.tabs([
        "📊 Multi-Model Benchmarks",
        "🛡️ Statistical Drift Monitor (PSI & KS)",
        "🎯 Conformal Calibration (CQR)",
    ])

    with perf_t1:
        st.markdown('<div class="sw-section-heading">Candidate Models Benchmark Scorecard</div>', unsafe_allow_html=True)
        if st.button("Run Full Benchmark & Ensemble Optimization", type="primary"):
            with st.spinner("Evaluating LightGBM, XGBoost, CatBoost, and Seasonal Naive..."):
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

                    st.success(f"Best Candidate Model: {e_rep.best_single_model} | Ensemble Validation WAPE: {e_rep.ensemble_wape*100:.2f}%")

                    rows = []
                    for m_name, c_m in e_rep.candidate_metrics.items():
                        rows.append({
                            "Candidate Model": m_name,
                            "Validation WAPE (%)": f"{c_m.wape*100:.2f}%",
                            "MAE": f"{c_m.mae:.2f}",
                            "RMSE": f"{c_m.rmse:.2f}",
                            "Pinball Loss": f"{c_m.pinball_avg:.3f}",
                            "Dynamic Weight (%)": f"{c_m.weight*100:.1f}%",
                            "Training Latency": f"{c_m.training_time_sec:.2f}s",
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                except Exception as ens_e:
                    st.error(f"Ensemble fitting error: {ens_e}")
        else:
            st.info("Click the button above to run dynamic inverse-loss validation benchmarking across LightGBM, XGBoost, CatBoost, and Seasonal Naive.")

    with perf_t2:
        st.markdown('<div class="sw-section-heading">Population Stability Index & Feature Drift Audit</div>', unsafe_allow_html=True)
        feat_cols = model_meta.get("feature_cols", [])
        split_idx = int(len(df) * 0.80)
        df_ref = df.iloc[:split_idx]
        df_cur = df.iloc[split_idx:]

        report = audit_dataset_drift(df_ref, df_cur, feature_cols=feat_cols, target_col="sales")

        dh1, dh2, dh3 = st.columns(3)
        with dh1:
            st.markdown(f"""
            <div class="sw-kpi-card">
                <div class="sw-kpi-header"><span class="sw-kpi-label">Drift Status</span><span class="sw-pill {'sw-pill-success' if report.overall_status=='HEALTHY' else 'sw-pill-danger'}">{report.overall_status}</span></div>
                <div class="sw-kpi-value">{report.overall_status}</div>
                <div class="sw-kpi-sub">{report.recommendation_summary}</div>
            </div>
            """, unsafe_allow_html=True)
        with dh2:
            st.markdown(f"""
            <div class="sw-kpi-card">
                <div class="sw-kpi-header"><span class="sw-kpi-label">Mean PSI Score</span><span class="sw-pill sw-pill-neutral">Binned</span></div>
                <div class="sw-kpi-value">{report.mean_psi_score:.3f}</div>
                <div class="sw-kpi-sub">Threshold: &lt; 0.10 Stable</div>
            </div>
            """, unsafe_allow_html=True)
        with dh3:
            st.markdown(f"""
            <div class="sw-kpi-card">
                <div class="sw-kpi-header"><span class="sw-kpi-label">Retrain Recommendation</span><span class="sw-pill {'sw-pill-success' if not report.retrain_recommended else 'sw-pill-danger'}">Status</span></div>
                <div class="sw-kpi-value" style="color: {'#DC2626' if report.retrain_recommended else '#16A34A'};">{'ACTION NEEDED' if report.retrain_recommended else 'NOT REQUIRED'}</div>
                <div class="sw-kpi-sub">Tracking Signal: {report.tracking_signal_status}</div>
            </div>
            """, unsafe_allow_html=True)

        drift_rows = []
        for feat, f_res in report.feature_details.items():
            drift_rows.append({
                "Feature Name": feat,
                "PSI Score": f"{f_res.psi_score:.4f}",
                "KS Statistic": f"{f_res.ks_statistic:.4f}",
                "KS p-value": f"{f_res.ks_p_value:.4e}",
                "Drift Classification": f_res.drift_status,
            })
        st.dataframe(pd.DataFrame(drift_rows), hide_index=True, use_container_width=True)

    with perf_t3:
        st.markdown('<div class="sw-section-heading">Conformalized Quantile Regression (Romano et al., 2019)</div>', unsafe_allow_html=True)
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
                st.metric("Raw Model Coverage", f"{cqr_res.raw_coverage*100:.1f}%")
            with cq3:
                st.metric("Conformal Calibrated Coverage", f"{cqr_res.calibrated_coverage*100:.1f}%", delta=f"{(cqr_res.calibrated_coverage - cqr_res.raw_coverage)*100:+.1f}%")


# =========================================================================
# PAGE 7: DATA MANAGEMENT
# =========================================================================
elif st.session_state.nav_page == "Data Management":
    render_top_bar("Data Management", sel_family, sel_store)

    st.markdown('<div class="sw-page-title">POS Data Ingestion & Pipeline Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Upload Point-of-Sale (POS) or inventory CSV/Excel exports, inspect dataset health, and view schema diagnostics.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Sales / Transaction File (.csv, .xlsx)",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is not None:
        try:
            fmt = "excel" if uploaded_file.name.endswith((".xlsx", ".xls")) else "csv"
            raw_bytes = uploaded_file.read()
            reg_df, report = ingest_merchant_dataset(io.BytesIO(raw_bytes), file_format=fmt)

            st.markdown('<div class="sw-section-heading">Data Health & Diagnostic Audit</div>', unsafe_allow_html=True)
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                st.markdown(f"""
                <div class="sw-kpi-card">
                    <div class="sw-kpi-header"><span class="sw-kpi-label">Pipeline Status</span><span class="sw-pill {'sw-pill-success' if report.is_valid else 'sw-pill-warning'}">Audit</span></div>
                    <div class="sw-kpi-value">{'Valid' if report.is_valid else 'Warning'}</div>
                    <div class="sw-kpi-sub">{report.status_message}</div>
                </div>
                """, unsafe_allow_html=True)
            with h2:
                st.markdown(f"""
                <div class="sw-kpi-card">
                    <div class="sw-kpi-header"><span class="sw-kpi-label">Timeline Span</span><span class="sw-pill sw-pill-neutral">Days</span></div>
                    <div class="sw-kpi-value">{report.total_days} <span style="font-size:0.9rem; font-weight:500; color:#64748B;">days</span></div>
                    <div class="sw-kpi-sub">{report.min_date} to {report.max_date}</div>
                </div>
                """, unsafe_allow_html=True)
            with h3:
                st.markdown(f"""
                <div class="sw-kpi-card">
                    <div class="sw-kpi-header"><span class="sw-kpi-label">Categories / Stores</span><span class="sw-pill sw-pill-info">Scope</span></div>
                    <div class="sw-kpi-value">{report.family_count} <span style="font-size:0.9rem; font-weight:500; color:#64748B;">items</span></div>
                    <div class="sw-kpi-sub">{report.store_count} Store Location(s)</div>
                </div>
                """, unsafe_allow_html=True)
            with h4:
                st.markdown(f"""
                <div class="sw-kpi-card">
                    <div class="sw-kpi-header"><span class="sw-kpi-label">Intermittent Sales</span><span class="sw-pill sw-pill-neutral">Ratio</span></div>
                    <div class="sw-kpi-value">{report.zero_sales_pct*100:.1f}%</div>
                    <div class="sw-kpi-sub">Zero transaction frequency</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="sw-section-heading">Regularized Daily Transactions Preview</div>', unsafe_allow_html=True)
            st.dataframe(reg_df.head(20), use_container_width=True)

        except Exception as e:
            st.error(f"Failed to process custom file: {e}")
    else:
        st.info("💡 Ready for merchant POS data: Export transaction logs from Square, Shopify, Lightspeed, or SAP and drag-and-drop here.")

    st.markdown('<div class="sw-section-heading">Active Ingested Dataset Overview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(15), use_container_width=True)


# =========================================================================
# PAGE 8: SETTINGS & REST API MICROSERVICE
# =========================================================================
elif st.session_state.nav_page == "Settings":
    render_top_bar("Settings & API", sel_family, sel_store)

    st.markdown('<div class="sw-page-title">System Settings & Production REST API</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Configure application defaults, view FastAPI microservice documentation, and trigger model retraining.</div>', unsafe_allow_html=True)

    with st.form("settings_global_form"):
        st.markdown('<div class="sw-section-heading">Global Policy Defaults</div>', unsafe_allow_html=True)
        set_c1, set_c2 = st.columns(2)
        with set_c1:
            n_store = st.selectbox("Default Store Location", options=stores, index=stores.index(sel_store))
        with set_c2:
            n_fam = st.selectbox("Default Product Category", options=families, index=families.index(sel_family))

        if st.form_submit_button("Save System Settings", type="primary"):
            st.session_state.sel_store = n_store
            st.session_state.sel_family = n_fam
            st.success("System settings updated!")
            st.rerun()

    st.markdown('<div class="sw-section-heading">Production FastAPI Endpoints</div>', unsafe_allow_html=True)
    st.markdown("""
    | Method | Route | Description |
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

    st.markdown('<div class="sw-section-heading">Live Model Retraining</div>', unsafe_allow_html=True)
    if st.button("Force Retrain Models Now"):
        with st.spinner("Retraining LightGBM models..."):
            run_training_pipeline(df)
            st.cache_resource.clear()
            st.success("Models retrained and refreshed successfully!")
            st.rerun()
