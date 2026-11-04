"""
tests/test_api.py — Unit tests for the FastAPI REST API microservice.
"""

import pytest
from fastapi.testclient import TestClient

from api import app, load_artifacts


@pytest.fixture(scope="module")
def client():
    load_artifacts()
    with TestClient(app) as test_client:
        yield test_client


def test_api_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "endpoints" in data


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "version" in data
    assert "timestamp" in data


def test_api_catalog(client):
    response = client.get("/catalog")
    assert response.status_code == 200
    data = response.json()
    assert "stores" in data
    assert "families" in data
    assert data["total_stores"] > 0
    assert data["total_families"] > 0


def test_api_forecast_endpoint(client):
    payload = {
        "store_nbr": 3,
        "family": "BEVERAGES",
        "horizon": 5,
        "apply_conformal": True,
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["store_nbr"] == 3
    assert data["family"] == "BEVERAGES"
    assert len(data["forecasts"]) == 5
    for item in data["forecasts"]:
        assert item["q10"] <= item["q50"] <= item["q90"]


def test_api_replenishment_endpoint(client):
    payload = {
        "store_nbr": 3,
        "family": "BEVERAGES",
        "on_hand": 20.0,
        "on_order": 10.0,
        "backorders": 0.0,
        "review_period": 7,
        "lead_time": 3,
        "service_level": 0.95,
        "apply_conformal": True,
    }
    response = client.post("/replenishment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["net_inventory_position"] == 30.0
    assert data["protection_horizon_days"] == 10
    assert data["order_up_to_level_S"] > 0
    assert data["recommended_order_qty"] >= 0
    assert "urgency_status" in data


def test_api_purchase_order_endpoint(client):
    payload = {
        "store_nbr": 3,
        "supplier_name": "Supplier Logistics S.A.",
        "items": [
            {
                "family": "BEVERAGES",
                "on_hand": 10.0,
                "on_order": 0.0,
                "backorders": 0.0,
                "pack_size": 12,
                "moq": 24,
                "unit_cost": 8.0,
                "lead_time_days": 3,
            }
        ]
    }
    response = client.post("/purchase-order", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "po_number" in data
    assert data["total_cases"] >= 1
    assert data["total_units"] >= 24
    assert data["total_cost"] > 0
    assert len(data["lines"]) == 1


def test_api_simulate_endpoint(client):
    payload = {
        "store_nbr": 3,
        "family": "BEVERAGES",
        "review_period": 7,
        "lead_time": 3,
        "service_level": 0.95,
        "unit_cost": 10.0,
        "selling_price": 16.0,
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "policies" in data
    assert "StockWise Dynamic Quantile" in data["policies"]
    assert "Static Min-Max (s, S)" in data["policies"]
    assert "7-Day Moving Average Heuristic" in data["policies"]
