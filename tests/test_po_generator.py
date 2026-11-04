"""
tests/test_po_generator.py — Unit tests for Purchase Order generation & ReportLab PDF.
"""

import pytest
from src.po_generator import (
    PurchaseOrderItem,
    SupplierInfo,
    create_purchase_order,
    export_po_to_csv_string,
    export_po_to_dataframe,
    generate_po_pdf,
)


def test_purchase_order_item_rounding_and_moq():
    # Test 1: Below MOQ (15 < 24) -> Should bump to 24 units (2 cases of 12)
    item1 = PurchaseOrderItem(
        store_nbr=1,
        family="DAIRY",
        raw_recommended_qty=15.0,
        pack_size=12,
        moq=24,
        unit_cost=5.0,
    )
    assert item1.cases_ordered == 2
    assert item1.rounded_units == 24
    assert item1.total_line_cost == 120.0  # 24 * 5.0

    # Test 2: Above MOQ (31 > 24) -> Should round up to 36 units (3 cases of 12)
    item2 = PurchaseOrderItem(
        store_nbr=1,
        family="BEVERAGES",
        raw_recommended_qty=31.0,
        pack_size=12,
        moq=24,
        unit_cost=10.0,
    )
    assert item2.cases_ordered == 3
    assert item2.rounded_units == 36
    assert item2.total_line_cost == 360.0

    # Test 3: Zero demand -> Should be 0
    item3 = PurchaseOrderItem(
        store_nbr=1,
        family="HARDWARE",
        raw_recommended_qty=0.0,
        pack_size=6,
        moq=12,
    )
    assert item3.cases_ordered == 0
    assert item3.rounded_units == 0
    assert item3.total_line_cost == 0.0


def test_create_purchase_order():
    supplier = SupplierInfo(name="Test Logistics", free_shipping_threshold=200.0)
    items = [
        {"family": "GROCERY I", "raw_recommended_qty": 50, "pack_size": 10, "moq": 20, "unit_cost": 4.0},
        {"family": "BEVERAGES", "raw_recommended_qty": 20, "pack_size": 6, "moq": 12, "unit_cost": 5.0},
    ]

    po = create_purchase_order(store_nbr=5, replenishment_items=items, supplier=supplier)

    assert po.store_nbr == 5
    assert po.supplier.name == "Test Logistics"
    assert len(po.items) == 2
    assert po.total_units == 50 + 24  # 50 (5 cases of 10) + 24 (4 cases of 6)
    assert po.total_cost == (50 * 4.0) + (24 * 5.0)  # 200 + 120 = 320.0
    assert po.qualifies_for_free_shipping is True


def test_export_po_dataframe_and_csv():
    items = [{"family": "BAKERY", "raw_recommended_qty": 30, "pack_size": 12, "moq": 12, "unit_cost": 2.5}]
    po = create_purchase_order(store_nbr=1, replenishment_items=items)

    df = export_po_to_dataframe(po)
    assert len(df) == 1
    assert "PO Number" in df.columns
    assert "Line Total ($)" in df.columns

    csv_text = export_po_to_csv_string(po)
    assert "PO Number" in csv_text
    assert "BAKERY" in csv_text


def test_generate_po_pdf():
    items = [
        {"family": "GROCERY I", "raw_recommended_qty": 45, "pack_size": 12, "moq": 24, "unit_cost": 8.0},
        {"family": "PRODUCE", "raw_recommended_qty": 30, "pack_size": 10, "moq": 20, "unit_cost": 6.5},
    ]
    po = create_purchase_order(store_nbr=2, replenishment_items=items)

    pdf_bytes = generate_po_pdf(po)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    # Standard PDF magic bytes start with %PDF
    assert pdf_bytes.startswith(b"%PDF")
