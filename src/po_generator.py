"""
src/po_generator.py — Purchase Order (PO) & Commercial Supplier Procurement Engine.

Converts algorithmic replenishment recommendations into commercial-grade,
supplier-ready Purchase Orders adhering to real-world procurement constraints:
1. Case/Pack Size Rounding (e.g., case packs of 6, 12, 24)
2. Minimum Order Quantity (MOQ) enforcement
3. Payment terms, Supplier profiles, and Expected Delivery ETA calculations
4. Multi-format export:
   - Pandas DataFrame
   - Standard CSV
   - Commercial-grade ReportLab PDF with corporate header, line-item table, and signature block
"""

from __future__ import annotations

import io
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SupplierInfo:
    """Supplier procurement profile and terms."""
    supplier_id: str = "SUP-001"
    name: str = "Andean Consumer Goods Logistics S.A."
    contact_email: str = "procurement@andeangoods.ec"
    payment_terms: str = "Net 30 Days"
    currency: str = "USD"
    default_lead_time_days: int = 3
    free_shipping_threshold: float = 500.0


@dataclass
class PurchaseOrderItem:
    """A single commercial line item on a purchase order."""
    store_nbr: int
    family: str
    raw_recommended_qty: float
    pack_size: int = 12
    moq: int = 24
    unit_cost: float = 8.50
    lead_time_days: int = 3
    
    # Computed fields
    cases_ordered: int = 0
    rounded_units: int = 0
    total_line_cost: float = 0.0
    expected_delivery_date: str = ""

    def __post_init__(self):
        # Enforce MOQ and pack-size rounding
        target_units = max(float(self.raw_recommended_qty), float(self.moq)) if self.raw_recommended_qty > 0 else 0
        if target_units > 0:
            cases = math.ceil(target_units / max(1, self.pack_size))
            self.cases_ordered = cases
            self.rounded_units = cases * self.pack_size
        else:
            self.cases_ordered = 0
            self.rounded_units = 0

        self.total_line_cost = round(self.rounded_units * self.unit_cost, 2)
        today = datetime.now()
        delivery = today + timedelta(days=self.lead_time_days)
        self.expected_delivery_date = delivery.strftime("%Y-%m-%d")


@dataclass
class PurchaseOrder:
    """Complete Purchase Order document."""
    po_number: str
    po_date: str
    store_nbr: int
    supplier: SupplierInfo
    items: List[PurchaseOrderItem] = field(default_factory=list)
    notes: str = "Automated replenishment order generated via AI Demand Forecasting Engine."
    
    @property
    def total_units(self) -> int:
        return sum(item.rounded_units for item in self.items)

    @property
    def total_cases(self) -> int:
        return sum(item.cases_ordered for item in self.items)

    @property
    def total_cost(self) -> float:
        return round(sum(item.total_line_cost for item in self.items), 2)

    @property
    def qualifies_for_free_shipping(self) -> bool:
        return self.total_cost >= self.supplier.free_shipping_threshold


def create_purchase_order(
    store_nbr: int,
    replenishment_items: List[dict],
    supplier: Optional[SupplierInfo] = None,
    po_number_prefix: str = "PO",
) -> PurchaseOrder:
    """
    Construct a PurchaseOrder from replenishment line dictionaries.
    
    Each item dict may contain:
      - 'family': str
      - 'raw_recommended_qty': float
      - 'pack_size': int (default 12)
      - 'moq': int (default 24)
      - 'unit_cost': float (default 10.0)
      - 'lead_time_days': int (default 3)
    """
    sup = supplier or SupplierInfo()
    now_str = datetime.now().strftime("%Y%m%d-%H%M")
    po_num = f"{po_number_prefix}-{store_nbr:03d}-{now_str}"
    today_str = datetime.now().strftime("%Y-%m-%d")

    po_items: List[PurchaseOrderItem] = []
    for item in replenishment_items:
        po_item = PurchaseOrderItem(
            store_nbr=store_nbr,
            family=item["family"],
            raw_recommended_qty=item.get("raw_recommended_qty", 0.0),
            pack_size=item.get("pack_size", 12),
            moq=item.get("moq", 24),
            unit_cost=item.get("unit_cost", 10.0),
            lead_time_days=item.get("lead_time_days", sup.default_lead_time_days),
        )
        po_items.append(po_item)

    return PurchaseOrder(
        po_number=po_num,
        po_date=today_str,
        store_nbr=store_nbr,
        supplier=sup,
        items=po_items,
    )


def export_po_to_dataframe(po: PurchaseOrder) -> pd.DataFrame:
    """Format Purchase Order lines into a user-facing pandas DataFrame."""
    rows = []
    for it in po.items:
        rows.append({
            "PO Number": po.po_number,
            "Store #": it.store_nbr,
            "Product Family": it.family,
            "Algorithmic Order": it.raw_recommended_qty,
            "MOQ": it.moq,
            "Pack Size": it.pack_size,
            "Cases Ordered": it.cases_ordered,
            "Final Units": it.rounded_units,
            "Unit Cost ($)": f"${it.unit_cost:.2f}",
            "Line Total ($)": f"${it.total_line_cost:.2f}",
            "Delivery ETA": it.expected_delivery_date,
        })
    return pd.DataFrame(rows)


def export_po_to_csv_string(po: PurchaseOrder) -> str:
    """Export the PO lines to standard RFC 4180 CSV string."""
    df = export_po_to_dataframe(po)
    return df.to_csv(index=False)


def generate_po_pdf(po: PurchaseOrder) -> bytes:
    """
    Generate an official commercial Purchase Order PDF using ReportLab.
    Returns bytes of the compiled PDF.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    primary_color = colors.HexColor("#287A50")  # Enterprise forest green
    dark_neutral = colors.HexColor("#202820")
    light_bg = colors.HexColor("#F7F8F5")
    border_color = colors.HexColor("#E0E3DE")

    title_style = ParagraphStyle(
        "POTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=primary_color,
        spaceAfter=4,
    )

    header_sub_style = ParagraphStyle(
        "POSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#555555"),
        spaceAfter=10,
    )

    meta_title_style = ParagraphStyle(
        "POMetaTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=dark_neutral,
    )

    meta_text_style = ParagraphStyle(
        "POMetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#333333"),
    )

    table_header_style = ParagraphStyle(
        "POTableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "POTableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=dark_neutral,
    )

    table_cell_bold = ParagraphStyle(
        "POTableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=dark_neutral,
    )

    elements = []

    # Title & Metadata Bar
    elements.append(Paragraph("OFFICIAL PURCHASE ORDER", title_style))
    elements.append(Paragraph("AI-Based Retail Demand Forecasting System — Procurement Microservice", header_sub_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=14))

    # Two-Column Header Info: PO Details vs Supplier Details
    po_info_col = [
        Paragraph(f"<b>PO Number:</b> {po.po_number}", meta_text_style),
        Paragraph(f"<b>Date:</b> {po.po_date}", meta_text_style),
        Paragraph(f"<b>Destination Store:</b> Store #{po.store_nbr:02d}", meta_text_style),
        Paragraph(f"<b>Status:</b> APPROVED / TRANSMITTED", meta_text_style),
    ]

    sup_info_col = [
        Paragraph(f"<b>Supplier:</b> {po.supplier.name}", meta_text_style),
        Paragraph(f"<b>Supplier ID:</b> {po.supplier.supplier_id}", meta_text_style),
        Paragraph(f"<b>Contact:</b> {po.supplier.contact_email}", meta_text_style),
        Paragraph(f"<b>Payment Terms:</b> {po.supplier.payment_terms}", meta_text_style),
    ]

    header_table = Table([[po_info_col, sup_info_col]], colWidths=[270, 270])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 14))

    # Table of Line Items
    col_widths = [140, 45, 45, 45, 50, 55, 65, 95]
    table_headers = [
        Paragraph("Product Family", table_header_style),
        Paragraph("Algo Qty", table_header_style),
        Paragraph("Pack", table_header_style),
        Paragraph("MOQ", table_header_style),
        Paragraph("Cases", table_header_style),
        Paragraph("Units", table_header_style),
        Paragraph("Unit Cost", table_header_style),
        Paragraph("Total Cost", table_header_style),
    ]

    data = [table_headers]

    for item in po.items:
        data.append([
            Paragraph(item.family, table_cell_style),
            Paragraph(f"{item.raw_recommended_qty:.0f}", table_cell_style),
            Paragraph(str(item.pack_size), table_cell_style),
            Paragraph(str(item.moq), table_cell_style),
            Paragraph(str(item.cases_ordered), table_cell_style),
            Paragraph(str(item.rounded_units), table_cell_bold),
            Paragraph(f"${item.unit_cost:.2f}", table_cell_style),
            Paragraph(f"${item.total_line_cost:,.2f}", table_cell_bold),
        ])

    items_table = Table(data, colWidths=col_widths, repeatRows=1)
    items_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    elements.append(items_table)
    elements.append(Spacer(1, 10))

    # Totals Summary Box
    totals_col = [
        Paragraph(f"<b>Total Ordered Cases:</b> {po.total_cases:,}", meta_text_style),
        Paragraph(f"<b>Total Ordered Units:</b> {po.total_units:,}", meta_text_style),
        Paragraph(f"<b>Net Order Valuation:</b> ${po.total_cost:,.2f} {po.supplier.currency}", meta_title_style),
        Paragraph(
            "<b>Shipping Terms:</b> " + ("Free Freight (Threshold Exceeded)" if po.qualifies_for_free_shipping else "Standard Freight FOB"),
            meta_text_style
        ),
    ]

    totals_table = Table([["", totals_col]], colWidths=[310, 230])
    totals_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (1, 0), (1, 0), light_bg),
        ('BOX', (1, 0), (1, 0), 1, primary_color),
        ('TOPPADDING', (1, 0), (1, 0), 8),
        ('BOTTOMPADDING', (1, 0), (1, 0), 8),
        ('LEFTPADDING', (1, 0), (1, 0), 10),
        ('RIGHTPADDING', (1, 0), (1, 0), 10),
    ]))
    elements.append(totals_table)
    elements.append(Spacer(1, 18))

    # Notes & Authorisation Signature Block
    notes_block = [
        Paragraph("<b>Commercial Terms & Conditions:</b>", meta_title_style),
        Paragraph(po.notes, meta_text_style),
        Paragraph("Deliveries accepted Mon-Fri between 06:00 and 14:00 at Receiving Dock B.", meta_text_style),
    ]

    sig_block = [
        Spacer(1, 20),
        HRFlowable(width="80%", thickness=0.8, color=colors.HexColor("#888888"), hAlign="LEFT"),
        Paragraph("Authorized Procurement Officer Signature", meta_text_style),
    ]

    footer_table = Table([[notes_block, sig_block]], colWidths=[320, 220])
    footer_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(footer_table)

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
