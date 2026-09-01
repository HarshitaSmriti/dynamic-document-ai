"""Real Document Dynamic Extraction Test Runner for 8176000040.pdf.

Uses the configured .env model (qwen/qwen2.5-vl-72b-instruct) without in-memory model overrides.
Demonstrates multi-page document consolidation, intelligent field deduplication, and continuation table merging.
"""

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import pymupdf
from PIL import Image

# Force UTF-8 encoding on standard output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings
from backend.extraction.parser import DynamicJSONParser
from backend.extraction.prompt import DynamicPromptBuilder
from backend.extraction.qwen_provider import HostedQwenProvider
from backend.schemas.extraction import ExtractedDocumentPayload


def render_pdf_to_images(pdf_path: str, dpi: int = 150) -> List[Image.Image]:
    """Render all pages of a PDF file to a list of PIL Images."""
    doc = pymupdf.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)
    return images


def run_real_document_test():
    pdf_path = r"C:\Users\HARSHITA\Downloads\8176000040.pdf"

    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    # Load configuration directly from .env (no in-memory model override)
    settings = get_settings()

    print("=" * 80)
    print("🚀 PRODUCTION MULTI-PAGE DYNAMIC EXTRACTION TEST")
    print("=" * 80)
    print(f"Document Path:       {pdf_path}")
    print(f"Configured Model:    {settings.QWEN_MODEL_NAME} (from .env)")
    print(f"API Endpoint:        {settings.QWEN_API_BASE}")
    print(f"API Key Configured:  {'Yes (Protected)' if settings.QWEN_API_KEY else 'No'}")
    print("-" * 80)

    # 1. Render PDF to images
    print("Rendering PDF pages to images...")
    t_render_start = time.perf_counter()
    page_images = render_pdf_to_images(pdf_path, dpi=150)
    t_render = time.perf_counter() - t_render_start
    total_pages = len(page_images)
    print(f"Rendered {total_pages} pages in {t_render:.2f}s.")

    # 2. Process all pages with Qwen-VL
    provider = HostedQwenProvider(settings=settings)

    page_payloads: List[ExtractedDocumentPayload] = []
    per_page_latencies: List[float] = []
    http_statuses: List[int] = []
    json_parsing_errors: List[str] = []

    overall_start_time = time.perf_counter()

    for idx, page_img in enumerate(page_images, 1):
        print(f"\n--- Processing Page {idx} of {total_pages} ---")
        prompt = DynamicPromptBuilder.build_extraction_prompt(
            page_num=idx,
            total_pages=total_pages,
        )

        page_t0 = time.perf_counter()
        try:
            raw_output, metadata = provider.generate_with_metadata(
                images=[page_img],
                prompt=prompt,
                system_prompt=DynamicPromptBuilder.SYSTEM_PROMPT,
            )
            page_lat = metadata.get("latency_seconds", time.perf_counter() - page_t0)
            status_code = metadata.get("http_status", 200)
            http_statuses.append(status_code)
            per_page_latencies.append(page_lat)
            print(f"Page {idx} received in {page_lat:.2f}s (HTTP {status_code})")

            # Parse with robust repair engine
            success, payload, parse_err = DynamicJSONParser.parse_and_validate(raw_output)
            if success:
                page_payloads.append(payload)
                print(f"Page {idx} parsed successfully ({len(payload.document)} fields, {len(payload.tables)} tables)")
                if parse_err:
                    print(f"  Note: {parse_err}")
            else:
                json_parsing_errors.append(f"Page {idx}: {parse_err}")
                print(f"Page {idx} parse failed: {parse_err}")

        except Exception as exc:
            page_lat = time.perf_counter() - page_t0
            per_page_latencies.append(page_lat)
            status_code = getattr(exc, "http_status", 500)
            http_statuses.append(status_code)
            err_msg = str(exc)
            json_parsing_errors.append(f"Page {idx} request error: {err_msg}")
            print(f"Page {idx} Error: {err_msg}")

    total_latency = time.perf_counter() - overall_start_time

    # 3. Intelligent multi-page consolidation (TASK 2 & TASK 3)
    consolidated_payload = DynamicJSONParser.merge_document_payloads(page_payloads)
    if json_parsing_errors:
        consolidated_payload.warnings.extend(json_parsing_errors)

    final_json = consolidated_payload.model_dump()

    # Save to disk
    output_json_path = PROJECT_ROOT / "tests" / "real_document_extraction_result.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    # 4. Report Results
    is_success = len(page_payloads) > 0 and len(json_parsing_errors) == 0

    print("\n" + "=" * 80)
    print("📊 MULTI-PAGE EXTRACTION EXECUTION REPORT")
    print("=" * 80)
    print(f"• Success/Failure:       {'✅ SUCCESS' if is_success else ('⚠️ PARTIAL SUCCESS' if page_payloads else '❌ FAILURE')}")
    print(f"• Total Latency:         {total_latency:.2f}s")
    print(f"• Pages Processed:       {total_pages}")
    print("• Per-Page Latencies:")
    for idx, lat in enumerate(per_page_latencies, 1):
        st = http_statuses[idx-1] if idx-1 < len(http_statuses) else "N/A"
        print(f"   - Page {idx}: {lat:.2f}s (HTTP {st})")
    print(f"• Detected Document Type:{consolidated_payload.document_type}")
    print(f"• Dynamic Fields Count:  {len(consolidated_payload.document)}")
    print(f"• Tables Count:          {len(consolidated_payload.tables)} (Continuation tables merged across pages)")
    for t_idx, tbl in enumerate(consolidated_payload.tables, 1):
        print(f"   - Table {t_idx} ('{tbl.get('table_name')}'): {len(tbl.get('rows', []))} rows, columns: {tbl.get('headers', [])}, spanning pages: {tbl.get('pages', [])}")
    print(f"• Lists Count:           {len(consolidated_payload.lists)}")
    print(f"• Warnings Count:        {len(consolidated_payload.warnings)}")
    print(f"• JSON Parsing Errors:   {len(json_parsing_errors)}")
    for err in json_parsing_errors:
        print(f"   * {err}")

    print("\n" + "-" * 80)
    print("🔍 FINAL CONSOLIDATED DYNAMIC JSON:")
    print("-" * 80)
    print(json.dumps(final_json, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    run_real_document_test()
