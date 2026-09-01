"""Minimal Standalone Test Script for Hosted Qwen2.5-VL Extraction.

Workflow:
1 Real Local Image
  ↓
HostedQwenProvider (OpenRouter / OpenAI-compatible API)
  ↓
Qwen2.5-VL-7B-Instruct
  ↓
Dynamic Extraction Prompt
  ↓
JSON Parser & Normalizer
  ↓
Final Dynamic JSON
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw

# Set UTF-8 encoding on standard streams for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings
from backend.extraction.parser import DynamicJSONParser
from backend.extraction.prompt import DynamicPromptBuilder
from backend.extraction.qwen_provider import HostedQwenProvider


def create_realistic_test_invoice_image(output_path: Path) -> Path:
    """Generate a high-clarity document image with real enterprise structure for testing."""
    img = Image.new("RGB", (900, 750), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Document Header & Border
    draw.rectangle([(25, 25), (875, 725)], outline=(180, 180, 180), width=2)
    draw.rectangle([(25, 25), (875, 95)], fill=(240, 244, 248))

    draw.text((45, 45), "ENTERPRISE CLOUD SERVICES - TAX INVOICE", fill=(20, 40, 80))
    draw.text((45, 70), "Invoice Number: INV-2025-9042 | Issue Date: 2025-01-15 | Due Date: 2025-02-15", fill=(80, 80, 80))

    # Vendor & Client Information
    draw.text((45, 115), "ISSUED BY (VENDOR):", fill=(20, 20, 20))
    draw.text((45, 135), "Cognitive Solutions Global Inc.", fill=(50, 50, 50))
    draw.text((45, 155), "100 Innovation Way, Suite 400, San Jose, CA 95134", fill=(70, 70, 70))
    draw.text((45, 175), "Tax ID / EIN: XX-XXXXXXX | support@cognitivesolutions.io", fill=(70, 70, 70))

    draw.text((480, 115), "BILLED TO (CLIENT):", fill=(20, 20, 20))
    draw.text((480, 135), "Nexus Supply Chain Dynamics Ltd.", fill=(50, 50, 50))
    draw.text((480, 155), "75 Park Avenue, 12th Floor, New York, NY 10017", fill=(70, 70, 70))
    draw.text((480, 175), "Client Account: ACC-NX-4412", fill=(70, 70, 70))

    # Table of Services
    draw.rectangle([(45, 220), (855, 250)], fill=(230, 235, 245))
    draw.text((55, 228), "Description", fill=(30, 30, 30))
    draw.text((420, 228), "Qty / Hrs", fill=(30, 30, 30))
    draw.text((560, 228), "Unit Rate", fill=(30, 30, 30))
    draw.text((740, 228), "Amount (USD)", fill=(30, 30, 30))

    # Row 1
    draw.text((55, 265), "Enterprise Document AI Model Pipeline Setup", fill=(40, 40, 40))
    draw.text((430, 265), "1.0", fill=(40, 40, 40))
    draw.text((560, 265), "$4,500.00", fill=(40, 40, 40))
    draw.text((740, 265), "$4,500.00", fill=(40, 40, 40))

    # Row 2
    draw.text((55, 305), "Multimodal Vision Inference API Integration", fill=(40, 40, 40))
    draw.text((430, 305), "40.0 hrs", fill=(40, 40, 40))
    draw.text((560, 305), "$150.00", fill=(40, 40, 40))
    draw.text((740, 305), "$6,000.00", fill=(40, 40, 40))

    # Row 3
    draw.text((55, 345), "Dedicated Production Support & SLA (Tier 1)", fill=(40, 40, 40))
    draw.text((430, 345), "1.0 mo", fill=(40, 40, 40))
    draw.text((560, 345), "$1,200.00", fill=(40, 40, 40))
    draw.text((740, 345), "$1,200.00", fill=(40, 40, 40))

    # Totals & Summary
    draw.line([(45, 390), (855, 390)], fill=(200, 200, 200), width=1)
    draw.text((560, 410), "Subtotal:", fill=(50, 50, 50))
    draw.text((740, 410), "$11,700.00", fill=(50, 50, 50))

    draw.text((560, 435), "Tax (0.00%):", fill=(50, 50, 50))
    draw.text((740, 435), "$0.00", fill=(50, 50, 50))

    draw.rectangle([(540, 465), (855, 505)], fill=(240, 248, 240))
    draw.text((560, 475), "TOTAL BALANCE DUE:", fill=(10, 80, 20))
    draw.text((740, 475), "$11,700.00 USD", fill=(10, 80, 20))

    # Terms & Notes
    draw.text((45, 530), "Payment Terms & Wire Instructions:", fill=(20, 20, 20))
    draw.text((45, 555), "- Bank Name: Silicon Valley Commercial Bank | Routing: 121000358", fill=(60, 60, 60))
    draw.text((45, 580), "- Account Beneficiary: Cognitive Solutions Global Inc. | Ref: INV-2025-9042", fill=(60, 60, 60))
    draw.text((45, 605), "- Late Fee: 1.5% per month applicable after due date (2025-02-15).", fill=(60, 60, 60))

    # Footer note
    draw.text((45, 680), "Authorized Officer Signature: [Digitally Verified]", fill=(120, 120, 120))

    img.save(output_path, format="PNG")
    return output_path


def run_test(image_path: str = None):
    """Run single document dynamic extraction test against configured Hosted Qwen API."""
    settings = get_settings()

    print("=" * 75)
    print("HOSTED QWEN2.5-VL EXTRACTION TEST")
    print("=" * 75)
    print(f"MODEL_BACKEND:       {settings.MODEL_BACKEND}")
    print(f"QWEN_API_BASE:       {settings.QWEN_API_BASE}")
    print(f"QWEN_MODEL_NAME:     {settings.QWEN_MODEL_NAME}")
    print(f"API_KEY_CONFIGURED:  {'Yes (Protected - not displayed)' if settings.QWEN_API_KEY else 'No'}")
    print("-" * 75)

    # Select document image
    if image_path and os.path.exists(image_path):
        target_path = Path(image_path)
        print(f"Using provided image: {target_path}")
    else:
        sample_path = PROJECT_ROOT / "tests" / "test_document.png"
        target_path = create_realistic_test_invoice_image(sample_path)
        print(f"Using local document image: {target_path}")

    image = Image.open(target_path)

    # Initialize provider & prompt
    provider = HostedQwenProvider(settings=settings)
    prompt = DynamicPromptBuilder.build_extraction_prompt()

    print("\nExecuting HostedQwenProvider extraction request against API...")

    http_status = None
    latency = None
    model_reported = settings.QWEN_MODEL_NAME
    raw_output = None
    error_msg = None
    success = False
    final_json = {}

    try:
        raw_output, metadata = provider.generate_with_metadata(
            images=[image],
            prompt=prompt,
            system_prompt=DynamicPromptBuilder.SYSTEM_PROMPT,
        )

        http_status = metadata.get("http_status")
        latency = metadata.get("latency_seconds")
        model_reported = metadata.get("model_used", settings.QWEN_MODEL_NAME)

        parse_success, payload, parse_error = DynamicJSONParser.parse_and_validate(raw_output)
        success = parse_success
        if parse_success:
            final_json = payload.model_dump()
        else:
            error_msg = f"JSON Parser error: {parse_error}"

    except Exception as exc:
        success = False
        error_msg = str(exc)
        http_status = getattr(exc, "http_status", None)
        latency = getattr(exc, "latency_seconds", None)

    # Report Results
    print("\n" + "=" * 75)
    print("EXTRACTION EXECUTION REPORT")
    print("=" * 75)
    print(f"HTTP Status:     {http_status if http_status else 'N/A'}")
    print(f"Model:           {model_reported}")
    print(f"Latency:         {f'{latency}s' if latency is not None else 'N/A'}")
    print(f"Success/Failure: {'SUCCESS' if success else 'FAILURE'}")
    if error_msg:
        print(f"Errors:          {error_msg}")
    else:
        print("Errors:          None")

    if raw_output:
        print("\n" + "-" * 75)
        print("RAW MODEL OUTPUT:")
        print("-" * 75)
        print(raw_output)

    print("\n" + "-" * 75)
    print("FINAL PARSED DYNAMIC JSON:")
    print("-" * 75)
    print(json.dumps(final_json, indent=2))
    print("=" * 75)


if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else None
    run_test(test_img)
