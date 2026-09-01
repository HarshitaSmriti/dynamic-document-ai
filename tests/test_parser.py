"""Unit tests for DynamicJSONParser, JSON repair engine, and multi-page consolidation."""

import pytest
from backend.extraction.parser import DynamicJSONParser
from backend.schemas.extraction import (
    DynamicFieldDefinition,
    ExtractedDocumentPayload,
    ExtractionSchema,
    FieldType,
)


def test_clean_markdown_fences():
    raw_input = """
Here is the extracted information:
```json
{
  "document_type": "grocery_receipt",
  "document": {
    "store_name": "SuperMart",
    "total": 45.50
  },
  "tables": [],
  "lists": [],
  "warnings": []
}
```
"""
    cleaned = DynamicJSONParser.clean_markdown_fences(raw_input)
    assert '"document_type": "grocery_receipt"' in cleaned


def test_repair_missing_commas_between_fields():
    # Model output with missing commas between key-value pairs
    broken_json = """
{
  "document_type": "purchase_order"
  "document": {
    "po_number": "PO-99120"
    "vendor": "Acme Corp"
    "total_amount": 5400.00
  }
}
"""
    success, payload, error = DynamicJSONParser.parse_and_validate(broken_json)
    assert success is True
    assert payload.document_type == "purchase_order"
    assert payload.document["po_number"] == "PO-99120"
    assert payload.document["vendor"] == "Acme Corp"


def test_repair_trailing_commas_and_truncated_braces():
    # Truncated JSON without closing braces and with trailing commas
    truncated_json = """
{
  "document_type": "agreement",
  "document": {
    "parties": "Alpha & Beta",
    "date": "2025-01-01",
"""
    success, payload, error = DynamicJSONParser.parse_and_validate(truncated_json)
    assert success is True
    assert payload.document["parties"] == "Alpha & Beta"
    assert payload.document["date"] == "2025-01-01"


def test_merge_document_payloads_field_deduplication():
    # Test TASK 2: Deduplication without "(Page X)" suffixes
    page1 = ExtractedDocumentPayload(
        document_type="stitching_invoice",
        document={
            "company_name": "DOLLAR INDUSTRIES LIMITED",
            "address": "8/624, AVINASHI GROUND PALAYAM",
            "bill_no": "8176000040",
            "bill_date": "13.04.2026",
        },
    )
    page2 = ExtractedDocumentPayload(
        document_type="invoice",
        document={
            "address": "8/624, AVINASHI GROUND PALAYAM BEHIND WEST COAST INDL AREA ANGERI PALAYAM",
            "BILL_NO": "8176000040",
            "total_qty": "12,672.00",
        },
    )

    merged = DynamicJSONParser.merge_document_payloads([page1, page2])

    assert merged.document_type == "stitching_invoice"
    # Merged address picked the more complete one without duplicate page keys
    assert "8/624, AVINASHI GROUND PALAYAM BEHIND WEST COAST" in merged.document["address"]
    # No page-suffixed keys created
    assert "address (Page 2)" not in merged.document
    assert "bill_no" in merged.document or "BILL_NO" in merged.document
    assert merged.document["total_qty"] == "12,672.00"


def test_merge_document_payloads_table_continuation():
    # Test TASK 3: Continuation table rows merged across pages
    page1 = ExtractedDocumentPayload(
        tables=[
            {
                "table_name": "Consumption_Details",
                "headers": ["Material Code", "Qty"],
                "rows": [["75000001", "7,200.00"], ["75000160", "7,200.00"]],
            }
        ]
    )
    page2 = ExtractedDocumentPayload(
        tables=[
            {
                "table_name": "Consumption_Details",
                "headers": ["Material Code", "Qty"],
                "rows": [["C-LHTR131000-65", "240.00"], ["C-XCTR083000-11-12", "228.00"]],
            }
        ]
    )

    merged = DynamicJSONParser.merge_document_payloads([page1, page2])

    # Should produce exactly 1 merged table rather than 2
    assert len(merged.tables) == 1
    table = merged.tables[0]
    assert table["table_name"] == "Consumption_Details"
    assert len(table["rows"]) == 4
    assert table["rows"][0][0] == "75000001"
    assert table["rows"][3][0] == "C-XCTR083000-11-12"
    assert table["pages"] == [1, 2]
