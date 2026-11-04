"""
api.py — Enterprise Retail Demand & Replenishment REST API Microservice.

Provides production HTTP endpoints for integration with external ERP, WMS, and POS systems:
- GET  /health          : Service health, model status, and uptime
- GET  /catalog         : Active store numbers and product families
- POST /forecast        : Multi-day quantile forecast (Q10, Q50, Q90) with optional Conformal Calibration
- POST /replenishment   : Algorithmic order-up-to (S) and recommended order quantity (Q)
- POST /purchase-order  : Commercial supplier PO generation with case-pack rounding and MOQ constraints
- POST /simulate        : Historical policy backtesting (StockWise Dynamic vs Static Min-Max vs Moving Avg)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

import config
from src.conformal import (
    apply_conformal_calibration,
    calibrate_forecast_dataframe,
    fit_conformal_calibrator,
)
from src.forecasting import generate_forecast
from src.inventory import InventoryInputs, compute_inventory
from src.po_generator import (
    PurchaseOrder,
    SupplierInfo,
    create_purchase_order,
    export_po_to_csv_string,
    export_po_to_dataframe,
)
from src.simulation import CostParameters, PolicyType, run_comparative_simulation

logger = logging.getLogger("retail_api")

app = FastAPI(
    title="AI-Based Retail Demand Forecasting API",
    description="Commercial-grade Demand Forecasting, Conformal Calibration, and Replenishment Decision Microservice",
    version="2.0.0",
)

# Enable CORS for frontend / web integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State / Model Loader ────────────────────────────────────────────────

_MODELS: Dict[str, Any] = {}
_META: Dict[str, Any] = {}
_DATA: Optional[pd.DataFrame] = None
_CONFORMAL_DELTA: float = 0.0


def load_artifacts():
    """Load trained models, metadata, and historical dataset on startup."""
    global _MODELS, _META, _DATA, _CONFORMAL_DELTA

    models_dir = config.MODELS_DIR
    q10_path = models_dir / "lgbm_q10.joblib"
    q50_path = models_dir / "lgbm_q50.joblib"
    q90_path = models_dir / "lgbm_q90.joblib"
    meta_path = models_dir / "model_meta.joblib"

    if q10_path.exists() and q50_path.exists() and q90_path.exists():
        _MODELS["q10"] = joblib.load(q10_path)
        _MODELS["q50"] = joblib.load(q50_path)
        _MODELS["q90"] = joblib.load(q90_path)
        if meta_path.exists():
            _META = joblib.load(meta_path)
        logger.info("Successfully loaded LGBM quantile models from %s", models_dir)
    else:
        logger.warning("Model files not found in %s. API running in degraded mode.", models_dir)

    data_path = config.PROCESSED_TRAIN
    if data_path.exists():
        _DATA = pd.read_parquet(data_path)
        logger.info("Loaded processed dataset with %d rows from %s", len(_DATA), data_path)
    else:
        logger.warning("Data file not found at %s.", data_path)

    # Pre-fit a default conformal calibration delta using the validation subset
    if _DATA is not None and "q50" in _MODELS:
        try:
            # Generate calibration residuals on a subset of the historical data
            val_df = _DATA.tail(min(300, len(_DATA))).copy()
            feature_cols = _META.get("feature_cols", [])
            if feature_cols and all(col in val_df.columns for col in feature_cols):
                X_val = val_df[feature_cols]
                y_val = val_df["sales"].to_numpy()
                q10_pred = _MODELS["q10"].predict(X_val)
                q90_pred = _MODELS["q90"].predict(X_val)
                _CONFORMAL_DELTA = fit_conformal_calibrator(y_val, q10_pred, q90_pred, target_coverage=0.80)
                logger.info("Conformal calibration fitted: delta = %.3f", _CONFORMAL_DELTA)
        except Exception as e:
            logger.warning("Could not precompute conformal calibration: %s", e)


@app.on_event("startup")
def startup_event():
    load_artifacts()


# ── Request / Response Schemas ────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    data_loaded: bool
    conformal_calibrated: bool
    timestamp: str


class CatalogResponse(BaseModel):
    stores: List[int]
    families: List[str]
    total_stores: int
    total_families: int


class ForecastRequest(BaseModel):
    store_nbr: int = Field(default=3, description="Retail Store ID")
    family: str = Field(default="BEVERAGES", description="Product Family / Category")
    horizon: int = Field(default=7, ge=1, le=30, description="Forecast horizon in days")
    apply_conformal: bool = Field(default=True, description="Whether to apply finite-sample Conformal Calibration")


class ForecastDayItem(BaseModel):
    date: str
    q10: float
    q50: float
    q90: float
    raw_q10: float
    raw_q90: float
    calibrated: bool


class ForecastResponse(BaseModel):
    store_nbr: int
    family: str
    horizon_days: int
    conformal_applied: bool
    conformal_delta: float
    forecasts: List[ForecastDayItem]


class ReplenishmentRequest(BaseModel):
    store_nbr: int = Field(default=3)
    family: str = Field(default="BEVERAGES")
    on_hand: float = Field(ge=0.0, description="Current stock physically on shelf")
    on_order: float = Field(default=0.0, ge=0.0, description="Units in transit from distributor")
    backorders: float = Field(default=0.0, ge=0.0, description="Unfulfilled customer backorders")
    review_period: int = Field(default=7, ge=1, le=30, description="Review cycle R in days")
    lead_time: int = Field(default=3, ge=1, le=30, description="Supplier lead time L in days")
    service_level: float = Field(default=0.95, ge=0.50, le=0.999, description="Target cycle service level")
    apply_conformal: bool = Field(default=True, description="Use Conformal uncertainty estimates")


class ReplenishmentResponse(BaseModel):
    store_nbr: int
    family: str
    on_hand: float
    on_order: float
    backorders: float
    net_inventory_position: float
    protection_horizon_days: int
    median_demand_during_horizon: float
    horizon_uncertainty_sigma: float
    safety_stock: int
    order_up_to_level_S: float
    recommended_order_qty: int
    days_of_inventory_remaining: float
    urgency_status: str
    explanation: str


class POItemInput(BaseModel):
    family: str
    on_hand: float = Field(ge=0.0)
    on_order: float = Field(default=0.0, ge=0.0)
    backorders: float = Field(default=0.0, ge=0.0)
    pack_size: int = Field(default=12, ge=1, description="Master case pack size")
    moq: int = Field(default=24, ge=0, description="Distributor Minimum Order Quantity")
    unit_cost: float = Field(default=10.0, ge=0.0, description="Wholesale cost per unit")
    lead_time_days: int = Field(default=3, ge=1)


class PurchaseOrderRequest(BaseModel):
    store_nbr: int = Field(default=3)
    supplier_name: Optional[str] = "Global Retail Logistics Ltd."
    payment_terms: Optional[str] = "Net 30 Days"
    currency: Optional[str] = "USD"
    items: List[POItemInput]


class POLineItemOutput(BaseModel):
    family: str
    algorithmic_order: float
    moq: int
    pack_size: int
    cases_ordered: int
    final_units: int
    unit_cost: float
    total_line_cost: float
    expected_delivery_date: str


class PurchaseOrderResponse(BaseModel):
    po_number: str
    po_date: str
    store_nbr: int
    supplier_name: str
    total_cases: int
    total_units: int
    total_cost: float
    qualifies_for_free_shipping: bool
    lines: List[POLineItemOutput]


class SimulationRequest(BaseModel):
    store_nbr: int = Field(default=3)
    family: str = Field(default="BEVERAGES")
    review_period: int = Field(default=7, ge=1, le=30)
    lead_time: int = Field(default=3, ge=1, le=30)
    service_level: float = Field(default=0.95, ge=0.50, le=0.99)
    unit_cost: float = Field(default=10.0, ge=0.01)
    selling_price: float = Field(default=16.0, ge=0.01)
    holding_cost_rate: float = Field(default=0.20, ge=0.0)
    stockout_penalty: float = Field(default=8.0, ge=0.0)
    order_fixed_cost: float = Field(default=25.0, ge=0.0)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "AI-Based Retail Demand Forecasting API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/catalog", "/forecast", "/replenishment", "/purchase-order", "/simulate"]
    }


@app.get("/health", response_model=HealthResponse)
def get_health():
    models_ready = ("q10" in _MODELS and "q50" in _MODELS and "q90" in _MODELS)
    data_ready = (_DATA is not None)
    return HealthResponse(
        status="healthy" if models_ready and data_ready else "degraded",
        version="2.0.0",
        models_loaded=models_ready,
        data_loaded=data_ready,
        conformal_calibrated=(_CONFORMAL_DELTA > 0),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/catalog", response_model=CatalogResponse)
def get_catalog():
    if _DATA is None:
        raise HTTPException(status_code=503, detail="Historical dataset not loaded.")
    stores = sorted(int(s) for s in _DATA["store_nbr"].unique())
    families = sorted(str(f) for f in _DATA["family"].unique())
    return CatalogResponse(
        stores=stores,
        families=families,
        total_stores=len(stores),
        total_families=len(families),
    )


@app.post("/forecast", response_model=ForecastResponse)
def post_forecast(req: ForecastRequest):
    if "q50" not in _MODELS or _DATA is None:
        raise HTTPException(status_code=503, detail="Forecasting engine or dataset not initialized.")

    # Filter history
    hist_sub = _DATA[(_DATA["store_nbr"] == req.store_nbr) & (_DATA["family"] == req.family)]
    if hist_sub.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No historical records found for Store #{req.store_nbr} and Family '{req.family}'."
        )

    # Generate raw multi-step forecasts
    raw_fc = generate_forecast(
        history_df=_DATA,
        q10_model=_MODELS["q10"],
        q50_model=_MODELS["q50"],
        q90_model=_MODELS["q90"],
        meta=_META,
        store_nbr=req.store_nbr,
        family=req.family,
        horizon=req.horizon,
    )

    delta = _CONFORMAL_DELTA if req.apply_conformal else 0.0

    items = []
    for _, row in raw_fc.iterrows():
        raw_q10 = float(row["pred_q10"])
        raw_q50 = float(row["pred_q50"])
        raw_q90 = float(row["pred_q90"])

        if req.apply_conformal and delta != 0.0:
            c_q10, c_q50, c_q90 = apply_conformal_calibration(
                np.array([raw_q10]), np.array([raw_q50]), np.array([raw_q90]), delta
            )
            fin_q10, fin_q50, fin_q90 = float(c_q10[0]), float(c_q50[0]), float(c_q90[0])
            is_calib = True
        else:
            fin_q10, fin_q50, fin_q90 = raw_q10, raw_q50, raw_q90
            is_calib = False

        items.append(
            ForecastDayItem(
                date=str(pd.Timestamp(row["date"]).date()),
                q10=round(fin_q10, 2),
                q50=round(fin_q50, 2),
                q90=round(fin_q90, 2),
                raw_q10=round(raw_q10, 2),
                raw_q90=round(raw_q90, 2),
                calibrated=is_calib,
            )
        )

    return ForecastResponse(
        store_nbr=req.store_nbr,
        family=req.family,
        horizon_days=req.horizon,
        conformal_applied=req.apply_conformal and (delta != 0.0),
        conformal_delta=round(delta, 3),
        forecasts=items,
    )


@app.post("/replenishment", response_model=ReplenishmentResponse)
def post_replenishment(req: ReplenishmentRequest):
    fc_res = post_forecast(
        ForecastRequest(
            store_nbr=req.store_nbr,
            family=req.family,
            horizon=req.review_period + req.lead_time,
            apply_conformal=req.apply_conformal,
        )
    )

    fc_df = pd.DataFrame([
        {
            "date": pd.to_datetime(f.date),
            "pred_q10": f.q10,
            "pred_q50": f.q50,
            "pred_q90": f.q90,
        }
        for f in fc_res.forecasts
    ])

    inv_inputs = InventoryInputs(
        on_hand=req.on_hand,
        on_order=req.on_order,
        backorders=req.backorders,
        review_period=req.review_period,
        lead_time=req.lead_time,
        service_level=req.service_level,
    )

    inv_res = compute_inventory(inv_inputs, fc_df)

    # Days of supply remaining
    daily_rate = inv_res.median_demand_H / max(1, inv_res.protection_horizon)
    days_remaining = round(req.on_hand / max(0.1, daily_rate), 1)

    if days_remaining <= req.lead_time:
        urgency = "CRITICAL: Imminent Stockout within Lead Time"
    elif days_remaining <= inv_res.protection_horizon:
        urgency = "ORDER NEEDED: Replenish to maintain Target S"
    else:
        urgency = "HEALTHY: Sufficient inventory coverage"

    return ReplenishmentResponse(
        store_nbr=req.store_nbr,
        family=req.family,
        on_hand=req.on_hand,
        on_order=req.on_order,
        backorders=req.backorders,
        net_inventory_position=inv_res.net_inventory_position,
        protection_horizon_days=inv_res.protection_horizon,
        median_demand_during_horizon=round(inv_res.median_demand_H, 2),
        horizon_uncertainty_sigma=round(inv_res.sigma_H, 2),
        safety_stock=inv_res.safety_stock,
        order_up_to_level_S=round(inv_res.target_level_S, 2),
        recommended_order_qty=inv_res.order_quantity,
        days_of_inventory_remaining=days_remaining,
        urgency_status=urgency,
        explanation=inv_res.explanation,
    )


@app.post("/purchase-order", response_model=PurchaseOrderResponse)
def post_purchase_order(req: PurchaseOrderRequest):
    supplier = SupplierInfo(
        name=req.supplier_name or "Global Retail Logistics Ltd.",
        payment_terms=req.payment_terms or "Net 30 Days",
        currency=req.currency or "USD",
    )

    replenishment_items = []
    for item in req.items:
        # Run replenishment calculation for each line
        rep = post_replenishment(
            ReplenishmentRequest(
                store_nbr=req.store_nbr,
                family=item.family,
                on_hand=item.on_hand,
                on_order=item.on_order,
                backorders=item.backorders,
                lead_time=item.lead_time_days,
            )
        )
        replenishment_items.append({
            "family": item.family,
            "raw_recommended_qty": rep.recommended_order_qty,
            "pack_size": item.pack_size,
            "moq": item.moq,
            "unit_cost": item.unit_cost,
            "lead_time_days": item.lead_time_days,
        })

    po = create_purchase_order(
        store_nbr=req.store_nbr,
        replenishment_items=replenishment_items,
        supplier=supplier,
    )

    lines = [
        POLineItemOutput(
            family=it.family,
            algorithmic_order=it.raw_recommended_qty,
            moq=it.moq,
            pack_size=it.pack_size,
            cases_ordered=it.cases_ordered,
            final_units=it.rounded_units,
            unit_cost=it.unit_cost,
            total_line_cost=it.total_line_cost,
            expected_delivery_date=it.expected_delivery_date,
        )
        for it in po.items
    ]

    return PurchaseOrderResponse(
        po_number=po.po_number,
        po_date=po.po_date,
        store_nbr=po.store_nbr,
        supplier_name=po.supplier.name,
        total_cases=po.total_cases,
        total_units=po.total_units,
        total_cost=po.total_cost,
        qualifies_for_free_shipping=po.qualifies_for_free_shipping,
        lines=lines,
    )


@app.post("/simulate")
def post_simulation(req: SimulationRequest):
    if _DATA is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded for simulation.")

    sub = _DATA[(_DATA["store_nbr"] == req.store_nbr) & (_DATA["family"] == req.family)].copy()
    if len(sub) < 25:
        raise HTTPException(status_code=400, detail="Insufficient history for meaningful simulation (need >= 25 days).")

    sub = sub.sort_values("date").set_index("date")
    demand_series = sub["sales"]

    # Compute quantile forecasts if features available
    q10_s, q50_s, q90_s = None, None, None
    if "q50" in _MODELS and _META:
        feat_cols = _META.get("feature_cols", [])
        if all(c in sub.columns for c in feat_cols):
            X = sub[feat_cols]
            q10_s = pd.Series(_MODELS["q10"].predict(X), index=demand_series.index)
            q50_s = pd.Series(_MODELS["q50"].predict(X), index=demand_series.index)
            q90_s = pd.Series(_MODELS["q90"].predict(X), index=demand_series.index)

    cost_params = CostParameters(
        unit_cost=req.unit_cost,
        selling_price=req.selling_price,
        holding_cost_rate_annual=req.holding_cost_rate,
        stockout_penalty_per_unit=req.stockout_penalty,
        order_fixed_cost=req.order_fixed_cost,
    )

    results = run_comparative_simulation(
        daily_demand_series=demand_series,
        review_period=req.review_period,
        lead_time=req.lead_time,
        service_level=req.service_level,
        cost_params=cost_params,
        q10_predictions=q10_s,
        q50_predictions=q50_s,
        q90_predictions=q90_s,
    )

    metrics_out = {}
    for p_type, sim_res in results.items():
        s = sim_res.summary
        metrics_out[getattr(p_type, "value", str(p_type))] = {
            "fill_rate_pct": s.fill_rate_pct,
            "stockout_days": s.stockout_days_count,
            "stockout_freq_pct": s.stockout_frequency_pct,
            "orders_placed": s.orders_placed_count,
            "total_units_ordered": s.total_units_ordered,
            "avg_on_hand_units": s.average_on_hand_units,
            "total_holding_cost": s.total_holding_cost,
            "total_stockout_cost": s.total_stockout_cost,
            "total_ordering_cost": s.total_ordering_cost,
            "gross_revenue": s.gross_revenue,
            "net_operating_profit": s.net_operating_profit,
        }

    return {
        "store_nbr": req.store_nbr,
        "family": req.family,
        "total_days_simulated": len(demand_series),
        "policies": metrics_out,
    }
