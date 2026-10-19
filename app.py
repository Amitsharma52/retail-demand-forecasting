"""
app.py — StockWise: Retail Inventory & Demand Assistant
"Know what to stock. Know when to order."

Redesigned shopkeeper-friendly interface maintaining complete underlying
machine learning pipeline, deterministic inventory calculations, and explainability.
"""

from __future__ import annotations

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
        max-width: 1140px;
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

    /* Metric card small */
    .sw-metric-box {
        background-color: #FFFFFF;
        border: 1px solid #E4E9E2;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        height: 100%;
    }
    .sw-metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #70796F !important;
        margin-bottom: 0.35rem;
    }
    .sw-metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #202820 !important;
        line-height: 1.2;
    }
    .sw-metric-sub {
        font-size: 0.8rem;
        color: #70796F !important;
        margin-top: 0.25rem;
    }

    /* Action Banner */
    .sw-action-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #287A50 !important;
        margin-bottom: 0.3rem;
    }
    .sw-action-main {
        font-size: 1.55rem;
        font-weight: 700;
        color: #202820 !important;
        line-height: 1.3;
        margin-bottom: 0.35rem;
    }
    .sw-action-desc {
        font-size: 0.95rem;
        color: #485247 !important;
        line-height: 1.5;
    }
    
    /* Badges */
    .sw-badge-green {
        display: inline-block;
        background-color: #E8F3EB;
        color: #287A50 !important;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .sw-badge-orange {
        display: inline-block;
        background-color: #FDF3E5;
        color: #D99024 !important;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .sw-badge-blue {
        display: inline-block;
        background-color: #EEF2FD;
        color: #4F6BED !important;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #D1D9CF;
        color: #202820;
        background-color: #FFFFFF;
        transition: all 0.15s ease-in-out;
    }
    .stButton > button:hover {
        border-color: #287A50;
        color: #287A50;
    }
    .stButton > button[kind="primary"] {
        background-color: #287A50 !important;
        border-color: #287A50 !important;
        color: #FFFFFF !important;
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
# Session State Initialization (Persistent Inventory Inputs across screens)
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

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — StockWise Navigation & Store Setting
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.4rem 0 0.8rem 0;">
        <div style="font-size: 1.25rem; font-weight: 800; color: #287A50; letter-spacing: -0.02em; line-height: 1.25;">🛒 AI-Based Retail Demand Forecasting</div>
        <div style="font-size: 0.8rem; color: #70796F; font-weight: 500; margin-top: 0.2rem;">Inventory Optimization & Decision Support</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    nav_options = [
        "Home",
        "Inventory",
        "Order Suggestions",
        "Sales Trends",
        "Settings",
        "Advanced Analytics",
    ]
    
    # Custom Radio navigation with icons
    def nav_format(opt):
        icons = {
            "Home": "🏠 Home",
            "Inventory": "📦 Inventory",
            "Order Suggestions": "📝 Order Suggestions",
            "Sales Trends": "📈 Sales Trends",
            "Settings": "⚙️ Settings",
            "Advanced Analytics": "🔬 Advanced Analytics",
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
    
    # Active Selection Summary (Visible & clear on all pages)
    stores = sorted(df["store_nbr"].unique().tolist())
    families = sorted(df["family"].unique().tolist())

    if "sel_store" not in st.session_state:
        st.session_state.sel_store = stores[0]
    if "sel_family" not in st.session_state:
        st.session_state.sel_family = families[0]

    st.markdown(f"""
    <div style="background-color: #F7F8F5; border: 1px solid #E4E9E2; border-radius: 8px; padding: 0.7rem 0.85rem; margin-top: 0.5rem;">
        <div style="font-size: 0.72rem; font-weight: 700; color: #70796F; text-transform: uppercase; letter-spacing: 0.04em;">Current Category</div>
        <div style="font-size: 0.92rem; font-weight: 700; color: #202820; margin-top: 0.15rem;">Store {st.session_state.sel_store} · {st.session_state.sel_family}</div>
        <div style="font-size: 0.75rem; color: #70796F; margin-top: 0.1rem;">Window: {meta['date_min']} to {meta['date_max']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 0.75rem; color: #70796F; text-align: center; font-weight: 500;'>Retail Demand AI v2.0 · Local & Secure</div>",
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
    
except Exception as err:
    st.error(f"Error computing store forecasts: {err}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1: Home — "Your Shop at a Glance"
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.nav_page == "Home":
    st.markdown('<div class="sw-page-title">Good morning!</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Here is what needs your attention today.</div>', unsafe_allow_html=True)

    # Compact selection summary
    st.markdown(f"""
    <div style="background-color: #FFFFFF; border: 1px solid #E4E9E2; border-radius: 10px; padding: 0.75rem 1.1rem; margin-bottom: 1.25rem; font-size: 0.9rem; display: flex; align-items: center; justify-content: space-between;">
        <div>
            <span style="color: #70796F;">Active selection:</span> 
            <strong>Store {sel_store}</strong> — <strong>{sel_family}</strong>
        </div>
        <div style="color: #70796F; font-size: 0.82rem;">
            7-Day Forecast Window: Next 7 Days
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3 Summary Cards
    c1, c2, c3 = st.columns(3)
    
    with c1:
        # Stock to Review card
        needs_order = inv_res.order_quantity > 0
        status_badge = '<span class="sw-badge-orange">Needs Review</span>' if needs_order else '<span class="sw-badge-green">Well Stocked</span>'
        review_text = "Replenishment recommended" if needs_order else "Target level satisfied"
        
        st.markdown(f"""
        <div class="sw-metric-box">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="sw-metric-label">Stock to Review</div>
                {status_badge}
            </div>
            <div class="sw-metric-val">{1 if needs_order else 0} <span style="font-size: 1rem; font-weight: 500; color: #70796F;">category</span></div>
            <div class="sw-metric-sub">{review_text} for Store {sel_store} ({sel_family})</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Current physical stock (clearly labeled as user-entered planning input)
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Current Stock (Input)</div>
            <div class="sw-metric-val">{st.session_state.on_hand:,.0f} <span style="font-size: 1rem; font-weight: 500; color: #70796F;">units</span></div>
            <div class="sw-metric-sub">User-entered on-hand stock for planning</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        # Upcoming Demand estimate
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Upcoming Demand (7 Days)</div>
            <div class="sw-metric-val">~{seven_day_demand:,.0f} <span style="font-size: 1rem; font-weight: 500; color: #70796F;">units</span></div>
            <div class="sw-metric-sub">Estimated expected sales (next 7 days)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Next Action Section
    st.markdown("### Your Next Action")
    
    if inv_res.order_quantity > 0:
        action_card_class = "sw-card-highlight"
        action_title = "ORDER RECOMMENDATION"
        action_headline = f"Consider ordering ~{inv_res.order_quantity:,.0f} units of {sel_family}"
        action_desc = f"Based on your lead time of {st.session_state.lead_time} days and review period of {st.session_state.review_period} days, your shop needs {inv_res.target_level_S:,.0f} units to safely cover expected sales plus buffer. Your current position is {inv_res.net_inventory_position:,.0f} units."
    else:
        action_card_class = "sw-card-highlight"
        action_title = "INVENTORY HEALTHY"
        action_headline = "No additional stock order needed right now"
        action_desc = f"Your current net inventory ({inv_res.net_inventory_position:,.0f} units) already meets the calculated target level ({inv_res.target_level_S:,.0f} units). No reorder is suggested for this cycle."

    st.markdown(f"""
    <div class="{action_card_class}">
        <div class="sw-action-title">{action_title}</div>
        <div class="sw-action-main">{action_headline}</div>
        <div class="sw-action-desc">{action_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns([1.5, 4])
    with btn_col1:
        if st.button("View Order Suggestion →", type="primary", use_container_width=True):
            st.session_state.nav_page = "Order Suggestions"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Expected Demand Preview Chart
    st.markdown("### Expected Demand Preview (7 Days)")
    st.caption("Green shows recent observed sales; blue shows estimated upcoming daily demand.")

    # Tail 21 days for clean preview
    hist_tail = df[(df["store_nbr"] == sel_store) & (df["family"] == sel_family)].sort_values("date").tail(21)
    fc_preview = fc_df.iloc[:7]

    fig_prev = go.Figure()

    # Historical line (Green #287A50)
    fig_prev.add_trace(go.Scatter(
        x=hist_tail["date"],
        y=hist_tail["sales"],
        mode="lines",
        name="Recent Observed Sales",
        line=dict(color="#287A50", width=2.2),
    ))

    # Uncertainty range band (Shaded light blue)
    fig_prev.add_trace(go.Scatter(
        x=pd.concat([fc_preview["date"], fc_preview["date"][::-1]]),
        y=pd.concat([fc_preview["pred_q90"], fc_preview["pred_q10"][::-1]]),
        fill="toself",
        fillcolor="rgba(79, 107, 237, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Expected Demand Range",
        hoverinfo="skip",
    ))

    # Forecast line (Blue #4F6BED)
    fig_prev.add_trace(go.Scatter(
        x=fc_preview["date"],
        y=fc_preview["pred_q50"],
        mode="lines+markers",
        name="Predicted Daily Sales",
        line=dict(color="#4F6BED", width=2.5),
        marker=dict(size=5, color="#4F6BED"),
    ))

    fig_prev.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#F0F2ED", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#F0F2ED", zeroline=False, title="Units Sold"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_prev, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2: Inventory — Simple Stock Register & Planning
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Inventory":
    st.markdown('<div class="sw-page-title">Inventory Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Understand the stock you have and the demand you need to prepare for.</div>', unsafe_allow_html=True)

    # Notice about planning inputs
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #E4E9E2; border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 1.25rem; font-size: 0.88rem; color: #485247;">
        💡 <strong>Planning Register</strong>: The numbers below represent your current planning inventory inputs for <strong>Store """ + str(sel_store) + """ — """ + sel_family + """</strong>. Update them anytime to recalculate recommendations immediately.
    </div>
    """, unsafe_allow_html=True)

    # 4 Simple Inventory Status Cards
    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Current Stock</div>
            <div class="sw-metric-val">{st.session_state.on_hand:,.0f}</div>
            <div class="sw-metric-sub">Units physically available in shop</div>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Already on Order</div>
            <div class="sw-metric-val">{st.session_state.on_order:,.0f}</div>
            <div class="sw-metric-sub">Units expected from supplier</div>
        </div>
        """, unsafe_allow_html=True)
    with i3:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Backorders</div>
            <div class="sw-metric-val">{st.session_state.backorders:,.0f}</div>
            <div class="sw-metric-sub">Committed / unfulfilled demand</div>
        </div>
        """, unsafe_allow_html=True)
    with i4:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Net Inventory Position</div>
            <div class="sw-metric-val" style="color: #287A50;">{inv_res.net_inventory_position:,.0f}</div>
            <div class="sw-metric-sub">Available + Incoming − Backorders</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Inventory Settings Section
    st.markdown("### Inventory Planning Settings")
    st.markdown("Adjust these numbers to match your current shop status:")

    with st.form("inventory_form"):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            new_on_hand = st.number_input(
                "Current stock on hand (units)",
                min_value=0.0,
                value=float(st.session_state.on_hand),
                step=10.0,
                help="Units physically available in your shop.",
            )
            new_on_order = st.number_input(
                "Units already on order",
                min_value=0.0,
                value=float(st.session_state.on_order),
                step=10.0,
                help="Units expected to arrive from existing supplier orders.",
            )
        with col_s2:
            new_backorders = st.number_input(
                "Outstanding backorders",
                min_value=0.0,
                value=float(st.session_state.backorders),
                step=5.0,
                help="Units promised to customers that you currently cannot fulfill.",
            )
            new_review = st.number_input(
                "Review period (days)",
                min_value=1,
                max_value=30,
                value=int(st.session_state.review_period),
                step=1,
                help="How often you review stock and place orders (e.g., every 7 days).",
            )
        with col_s3:
            new_lead = st.number_input(
                "Supplier lead time (days)",
                min_value=1,
                max_value=30,
                value=int(st.session_state.lead_time),
                step=1,
                help="How many days your supplier takes to deliver after you order.",
            )
            new_svc = st.select_slider(
                "Desired availability level",
                options=[0.90, 0.95, 0.99],
                value=float(st.session_state.service_level),
                format_func=lambda x: f"{int(x*100)}% Availability",
                help="Higher availability provides greater safety stock buffer against stockouts, but holds more stock.",
            )

        submit_btn = st.form_submit_button("Update Inventory Settings", type="primary")
        if submit_btn:
            st.session_state.on_hand = new_on_hand
            st.session_state.on_order = new_on_order
            st.session_state.backorders = new_backorders
            st.session_state.review_period = new_review
            st.session_state.lead_time = new_lead
            st.session_state.service_level = new_svc
            st.success("Inventory settings updated!")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3: Order Suggestions — The Main Business Feature
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Order Suggestions":
    st.markdown('<div class="sw-page-title">Order Suggestions</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Plan your next order with confidence.</div>', unsafe_allow_html=True)

    # Prominent Recommendation Card
    order_qty = inv_res.order_quantity
    
    if order_qty > 0:
        st.markdown(f"""
        <div class="sw-card-highlight">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="sw-action-title">RECOMMENDED ORDER FOR {sel_family}</div>
                    <div style="font-size: 2.8rem; font-weight: 800; color: #287A50; line-height: 1.1;">
                        {order_qty:,.0f} <span style="font-size: 1.25rem; font-weight: 500; color: #70796F;">units</span>
                    </div>
                </div>
                <span class="sw-badge-green" style="font-size: 0.9rem; padding: 0.4rem 0.8rem;">Ready to Order</span>
            </div>
            <div style="font-size: 0.95rem; color: #485247; margin-top: 0.8rem; line-height: 1.5;">
                Your next order needs to cover approximately <strong>{inv_res.protection_horizon} days of demand</strong> (your {st.session_state.review_period}-day review cycle plus {st.session_state.lead_time} days supplier delivery time).
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="sw-card-highlight">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="sw-action-title">INVENTORY STATUS</div>
                    <div style="font-size: 2.6rem; font-weight: 800; color: #287A50; line-height: 1.1;">
                        0 <span style="font-size: 1.25rem; font-weight: 500; color: #70796F;">units needed</span>
                    </div>
                </div>
                <span class="sw-badge-green" style="font-size: 0.9rem; padding: 0.4rem 0.8rem;">Adequately Stocked</span>
            </div>
            <div style="font-size: 0.95rem; color: #485247; margin-top: 0.8rem; line-height: 1.5;">
                Your current stock position already covers expected sales and required safety buffer over the {inv_res.protection_horizon}-day protection period. No additional purchase order is recommended right now.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4 Key Calculation Drivers
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Expected Demand ({inv_res.protection_horizon}d)</div>
            <div class="sw-metric-val">{inv_res.median_demand_H:,.0f}</div>
            <div class="sw-metric-sub">Forecasted sales during cycle</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Safety Buffer</div>
            <div class="sw-metric-val">{inv_res.safety_stock:,.0f}</div>
            <div class="sw-metric-sub">Buffer for unexpected spikes</div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Target Inventory Level</div>
            <div class="sw-metric-val">{inv_res.target_level_S:,.0f}</div>
            <div class="sw-metric-sub">Expected sales + Safety buffer</div>
        </div>
        """, unsafe_allow_html=True)
    with d4:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Current Net Position</div>
            <div class="sw-metric-val">{inv_res.net_inventory_position:,.0f}</div>
            <div class="sw-metric-sub">On-hand + Incoming − Backorders</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Collapsible Explanation Section
    with st.expander("🔍 How we calculated this suggestion", expanded=False):
        st.markdown(f"""
        1. **What is currently available:** Your net stock position is **{inv_res.net_inventory_position:,.0f} units** ({st.session_state.on_hand:.0f} on hand + {st.session_state.on_order:.0f} ordered − {st.session_state.backorders:.0f} backorders).
        2. **What your shop is expected to sell:** Over your **{inv_res.protection_horizon}-day protection period** ({st.session_state.review_period} review days + {st.session_state.lead_time} delivery days), the model projects sales of **{inv_res.median_demand_H:,.0f} units**.
        3. **Extra buffer needed:** To safeguard an availability target of **{int(st.session_state.service_level*100)}%**, the system adds **{inv_res.safety_stock} units** of safety stock.
        4. **Target order amount:** Target level is **{inv_res.target_level_S:,.0f} units**. Subtracting your current position of {inv_res.net_inventory_position:,.0f} gives a recommended order of **{inv_res.order_quantity:,.0f} units**.
        """)
        st.caption("Note: Forecasts are data-driven estimates. Real replenishment orders should factor in supplier batch minimums and shelf life.")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 4: Sales Trends — Simple & Clean Historical Chart
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Sales Trends":
    st.markdown('<div class="sw-page-title">Sales Trends</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">See what has been selling and what may sell next.</div>', unsafe_allow_html=True)

    # Clean single interactive chart
    series_df = df[(df["store_nbr"] == sel_store) & (df["family"] == sel_family)].sort_values("date")
    hist_view = series_df.tail(60)

    fig_trends = go.Figure()

    # Historical sales in Green (#287A50)
    fig_trends.add_trace(go.Scatter(
        x=hist_view["date"],
        y=hist_view["sales"],
        mode="lines",
        name="Actual Historical Sales",
        line=dict(color="#287A50", width=2.2),
    ))

    # Prediction uncertainty band (#4F6BED light)
    fig_trends.add_trace(go.Scatter(
        x=pd.concat([fc_df["date"][:7], fc_df["date"][:7][::-1]]),
        y=pd.concat([fc_df["pred_q90"][:7], fc_df["pred_q10"][:7][::-1]]),
        fill="toself",
        fillcolor="rgba(79, 107, 237, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Predicted Demand Range",
        hoverinfo="skip",
    ))

    # Predicted sales in Blue (#4F6BED)
    fig_trends.add_trace(go.Scatter(
        x=fc_df["date"][:7],
        y=fc_df["pred_q50"][:7],
        mode="lines+markers",
        name="Predicted Demand (7 Days)",
        line=dict(color="#4F6BED", width=2.6),
        marker=dict(size=6, color="#4F6BED"),
    ))

    # Promotion markers if available
    if "onpromotion" in hist_view.columns:
        promo_data = hist_view[hist_view["onpromotion"] > 0]
        if not promo_data.empty:
            fig_trends.add_trace(go.Scatter(
                x=promo_data["date"],
                y=promo_data["sales"],
                mode="markers",
                name="Promotional Display Day",
                marker=dict(color="#D99024", size=6, symbol="diamond"),
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

    # 3 Simple Observations
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
            <div class="sw-metric-sub">Highest daily volume recorded for this category</div>
        </div>
        """, unsafe_allow_html=True)
    with ob3:
        st.markdown(f"""
        <div class="sw-metric-box">
            <div class="sw-metric-label">Recent Trend Pattern</div>
            <div class="sw-metric-val" style="font-size: 1.35rem; text-transform: capitalize; color: #287A50;">{recent_trend}</div>
            <div class="sw-metric-sub">Past 7 days compared to past 14 days</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Collapsible secondary trends
    with st.expander("📊 Explore More Trends (Category Volume Comparison)", expanded=False):
        fam_totals = df[df["store_nbr"] == sel_store].groupby("family")["sales"].sum().sort_values(ascending=True).reset_index()
        fig_cat = go.Figure(go.Bar(
            x=fam_totals["sales"],
            y=fam_totals["family"],
            orientation="h",
            marker_color="#287A50",
        ))
        fig_cat.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            xaxis=dict(showgrid=True, gridcolor="#F0F2ED", title="Total Units Sold"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_cat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 5: Settings — Store & Product Family Selectors
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Settings":
    st.markdown('<div class="sw-page-title">Store & Category Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Configure the retail store and product category you want to analyze.</div>', unsafe_allow_html=True)

    with st.form("settings_form"):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            sel_st = st.selectbox(
                "Select Store Location",
                options=stores,
                index=stores.index(st.session_state.sel_store) if st.session_state.sel_store in stores else 0,
                help="Chooses the retail store location from the Corporación Favorita dataset.",
            )
        with col_c2:
            sel_fm = st.selectbox(
                "Select Product Category",
                options=families,
                index=families.index(st.session_state.sel_family) if st.session_state.sel_family in families else 0,
                help="Chooses the product family series for demand forecasting and inventory replenishment.",
            )

        save_btn = st.form_submit_button("Apply Selection", type="primary")
        if save_btn:
            st.session_state.sel_store = sel_st
            st.session_state.sel_family = sel_fm
            st.success("Target store and category updated!")
            st.rerun()

    st.markdown("---")
    st.markdown("### Model & System Maintenance")
    st.markdown("Update or retrain the underlying LightGBM forecasting models on local data:")
    if st.button("Retrain Models Now"):
        with st.spinner("Retraining LightGBM quantile models..."):
            run_training_pipeline(df)
            st.cache_resource.clear()
            st.success("Models retrained and refreshed successfully!")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 6: Advanced Analytics — Academic & Technical Evaluation
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.nav_page == "Advanced Analytics":
    st.markdown('<div class="sw-page-title">Advanced Analytics & Model Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sw-page-subtitle">Academic evaluations, TreeSHAP feature importances, and mathematical specifications.</div>', unsafe_allow_html=True)

    tech_tab1, tech_tab2, tech_tab3, tech_tab4 = st.tabs([
        "🔬 Explainable AI (SHAP)",
        "📊 Model Verification & Metrics",
        "📐 Mathematical Inventory Logic",
        "ℹ️ Data Pipeline & Limitations",
    ])

    with tech_tab1:
        st.markdown("#### Feature Influence Analysis (TreeSHAP)")
        st.caption("Game-theoretic attributions for the LightGBM median (Q50) forecasting model.")

        if not is_shap_available():
            st.warning("SHAP package not installed. Run `pip install shap`.")
        else:
            feature_cols = model_meta.get("feature_cols", [])
            sh_col1, sh_col2 = st.columns(2)

            with sh_col1:
                st.markdown("##### Global Feature Drivers (Model-Wide)")
                try:
                    imp_df = global_feature_importance(q50_model, feature_cols, top_n=10)
                    
                    # Human-friendly feature labels mapping
                    friendly_names = {
                        "lag_1": "Sales 1 Day Ago (lag_1)",
                        "lag_7": "Sales 7 Days Ago (lag_7)",
                        "lag_14": "Sales 14 Days Ago (lag_14)",
                        "lag_30": "Sales 30 Days Ago (lag_30)",
                        "roll_mean_7": "7-Day Moving Average",
                        "roll_mean_14": "14-Day Moving Average",
                        "roll_mean_30": "30-Day Moving Average",
                        "roll_std_7": "7-Day Sales Volatility",
                        "roll_std_14": "14-Day Sales Volatility",
                        "roll_std_30": "30-Day Sales Volatility",
                        "ewma_7": "Exponential Moving Average (7d)",
                        "day_of_week": "Day of the Week",
                        "month": "Month of Year",
                        "is_on_promotion": "Promotion Active (Yes/No)",
                        "promo_roll_7": "7-Day Promotion Frequency",
                        "store_nbr": "Store Identifier",
                        "family": "Product Category",
                    }
                    imp_df["display_feature"] = imp_df["feature"].map(lambda x: friendly_names.get(x, x))

                    fig_imp = go.Figure(go.Bar(
                        x=imp_df["importance"],
                        y=imp_df["display_feature"],
                        orientation="h",
                        marker_color="#4F6BED",
                    ))
                    fig_imp.update_layout(
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=350,
                        yaxis=dict(autorange="reversed"),
                        xaxis=dict(title="Importance Score (Mean |SHAP|)", showgrid=True, gridcolor="#F0F2ED"),
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                except Exception as e:
                    st.info(f"Global importance status: {e}")

            with sh_col2:
                st.markdown("##### Local Driver Analysis (Next Day Forecast)")
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
                        top_loc["display_feature"] = top_loc["feature"].map(lambda x: friendly_names.get(x, x) if "friendly_names" in locals() else x)
                        bar_colors = ["#D94F45" if v > 0 else "#4F6BED" for v in top_loc["shap_value"]]
                        
                        fig_loc = go.Figure(go.Bar(
                            x=top_loc["shap_value"],
                            y=top_loc["display_feature"],
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
                    else:
                        st.info("Local SHAP values computed.")
                except Exception as e:
                    st.info(f"Local attribution status: {e}")

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

    with tech_tab2:
        st.markdown("#### Validation Metrics (Chronological Holdout Split)")
        if "metrics" in model_meta:
            mdf = format_metrics_table(model_meta["metrics"])
            st.dataframe(mdf, hide_index=True, use_container_width=True)

            wape_lgbm = model_meta["metrics"].get("lgbm_wape", 0)
            wape_base = model_meta["metrics"].get("baseline_wape", 0)
            if isinstance(wape_lgbm, float) and isinstance(wape_base, float):
                st.success(
                    f"LightGBM achieved **{wape_lgbm*100:.2f}% WAPE** vs. Seasonal Naive benchmark of **{wape_base*100:.2f}% WAPE** "
                    f"({(wape_base - wape_lgbm)*100:.1f} percentage point improvement on real held-out data)."
                )

    with tech_tab3:
        st.markdown("#### Technical Inventory Formulations")
        st.latex(r"I_{pos} = \text{On-Hand} + \text{On-Order} - \text{Backorders}")
        st.latex(r"H = R + L \quad (\text{Protection Horizon})")
        st.latex(r"D_H = \sum_{t=1}^{H} Q_{50,t} \quad (\text{Expected Cumulative Demand})")
        st.latex(r"\sigma_t = \frac{Q_{90,t} - Q_{10,t}}{2 \times 1.2816} \approx \frac{Q_{90,t} - Q_{10,t}}{2.564}")
        st.latex(r"\sigma_H = \sqrt{\sum_{t=1}^{H} \sigma_t^2} \quad (\text{Protection Uncertainty})")
        st.latex(r"SS = \max\left(0, \lceil z_p \cdot \sigma_H \rceil\right) \quad (\text{Safety Stock})")
        st.latex(r"S = D_H + SS \quad (\text{Target Order-Up-To Level})")
        st.latex(r"Q = \max\left(0, \lceil S - I_{pos} \rceil\right) \quad (\text{Recommended Order Quantity})")

    with tech_tab4:
        st.markdown("#### Dataset Specifications & Academic Limitations")
        st.markdown("""
        - **Source**: Corporación Favorita Grocery Retail Dataset (Ecuador).
        - **Aggregation Level**: Evaluated at store-product family series level (`store_nbr` × `family`).
        - **Leakage Prevention**: All rolling windows (`roll_mean`, `roll_std`, `ewma`) strictly utilize `.shift(1)` to ensure current-day target values never contaminate model inputs.
        - **Censored Demand Limitation**: The Favorita dataset logs recorded sales transactions. It lacks historical on-hand inventory levels and stockout records; genuine unobserved stockout demand reconstruction is preserved as a future research enhancement.
        - **Independence Assumption**: Multi-day protection-horizon uncertainty approximation ($\sigma_H$) assumes independent forecast errors across consecutive days.
        """)
