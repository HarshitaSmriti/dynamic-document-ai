"""Document Extraction Service.

Coordinates document pre-processing (PDF rendering, image decoding),
multi-page dynamic extraction, VLM inference, and intelligent document-level consolidation.
"""

import base64
import io
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import pymupdf  # PyMuPDF for fast PDF rendering
from PIL import Image

from backend.extraction.parser import DynamicJSONParser
from backend.extraction.prompt import DynamicPromptBuilder
from backend.extraction.qwen_provider import BaseVLMProvider, get_vlm_provider
from backend.schemas.extraction import (
    ExtractedDocumentPayload,
    ExtractionRequest,
    ExtractionResponse,
    ExtractionSchema,
)


class DocumentExtractionService:
    """Production-ready dynamic document extraction service supporting single and multi-page documents."""

    def __init__(self, provider: Optional[BaseVLMProvider] = None):
        self.provider = provider or get_vlm_provider()
        self.prompt_builder = DynamicPromptBuilder()
        self.parser = DynamicJSONParser()

    def decode_base64_image(self, base64_str: str) -> Image.Image:
        """Decode base64 string to PIL Image."""
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        image_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def render_pdf_bytes_to_images(self, pdf_bytes: bytes, dpi: int = 150) -> List[Image.Image]:
        """Convert PDF byte stream into a list of PIL Images."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(img)
        return images

    def extract_from_images(
        self,
        images: List[Union[Image.Image, str]],
        schema: Optional[ExtractionSchema] = None,
        custom_instructions: Optional[str] = None,
    ) -> ExtractionResponse:
        """Extract dynamic structured data from single or multiple document page images."""
        if not images:
            return ExtractionResponse(
                success=False,
                data=ExtractedDocumentPayload(
                    document_type="error",
                    warnings=["No document images provided for extraction."],
                ),
                raw_model_output=None,
                error_message="Image list is empty.",
            )

        total_pages = len(images)
        start_time = time.perf_counter()

        # Single-page document extraction
        if total_pages == 1:
            prompt = self.prompt_builder.build_extraction_prompt(
                schema=schema,
                custom_instructions=custom_instructions,
                page_num=1,
                total_pages=1,
            )
            try:
                raw_output = self.provider.generate(
                    images=[images[0]],
                    prompt=prompt,
                    system_prompt=self.prompt_builder.SYSTEM_PROMPT,
                )
                success, payload, error_msg = self.parser.parse_and_validate(
                    raw_output=raw_output,
                    schema=schema,
                )
                return ExtractionResponse(
                    success=success,
                    data=payload,
                    raw_model_output=raw_output,
                    error_message=error_msg,
                )
            except Exception as exc:
                return ExtractionResponse(
                    success=False,
                    data=ExtractedDocumentPayload(
                        document_type="error",
                        warnings=[f"Single-page extraction failed: {str(exc)}"],
                    ),
                    raw_model_output=None,
                    error_message=str(exc),
                )

        # Multi-page document extraction with document-level consolidation (TASK 1, 2, 3)
        page_payloads: List[ExtractedDocumentPayload] = []
        raw_outputs: List[str] = []
        all_warnings: List[str] = []

        for idx, img in enumerate(images, 1):
            prompt = self.prompt_builder.build_extraction_prompt(
                schema=schema,
                custom_instructions=custom_instructions,
                page_num=idx,
                total_pages=total_pages,
            )
            try:
                raw_out = self.provider.generate(
                    images=[img],
                    prompt=prompt,
                    system_prompt=self.prompt_builder.SYSTEM_PROMPT,
                )
                raw_outputs.append(f"--- PAGE {idx} RAW OUTPUT ---\n{raw_out}")

                success, page_payload, parse_err = self.parser.parse_and_validate(
                    raw_output=raw_out,
                    schema=schema,
                )
                if success:
                    page_payloads.append(page_payload)
                else:
                    all_warnings.append(f"Page {idx} parse issue: {parse_err}")
            except Exception as exc:
                all_warnings.append(f"Page {idx} extraction failed: {str(exc)}")

        if not page_payloads:
            return ExtractionResponse(
                success=False,
                data=ExtractedDocumentPayload(
                    document_type="error",
                    warnings=all_warnings or ["All pages failed extraction."],
                ),
                raw_model_output="\n\n".join(raw_outputs) if raw_outputs else None,
                error_message="Multi-page extraction failed for all pages.",
            )

        # Merge fields & continuation tables across pages into a unified document
        consolidated_payload = self.parser.merge_document_payloads(page_payloads)
        if all_warnings:
            consolidated_payload.warnings.extend(all_warnings)

        return ExtractionResponse(
            success=True,
            data=consolidated_payload,
            raw_model_output="\n\n".join(raw_outputs),
            error_message=None if not all_warnings else f"{len(all_warnings)} page warnings recorded.",
        )

    def process_file_bytes(
        self,
        file_bytes: bytes,
        filename: str = "document",
        schema: Optional[ExtractionSchema] = None,
        custom_instructions: Optional[str] = None,
    ) -> ExtractionResponse:
        """Process an uploaded document (PDF or image file bytes)."""
        is_pdf = filename.lower().endswith(".pdf") or file_bytes.startswith(b"%PDF")

        if is_pdf:
            try:
                images = self.render_pdf_bytes_to_images(file_bytes, dpi=150)
            except Exception as exc:
                return ExtractionResponse(
                    success=False,
                    data=ExtractedDocumentPayload(document_type="error"),
                    error_message=f"Failed to render PDF document: {str(exc)}",
                )
        else:
            try:
                img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                images = [img]
            except Exception as exc:
                return ExtractionResponse(
                    success=False,
                    data=ExtractedDocumentPayload(document_type="error"),
                    error_message=f"Failed to open image file: {str(exc)}",
                )

        return self.extract_from_images(
            images=images,
            schema=schema,
            custom_instructions=custom_instructions,
        )

    def process_request(self, request: ExtractionRequest) -> ExtractionResponse:
        """Process an ExtractionRequest containing a base64-encoded document."""
        if not request.document_base64:
            return ExtractionResponse(
                success=False,
                data=ExtractedDocumentPayload(
                    document_type="error",
                    warnings=["Missing document_base64 in request payload."],
                ),
                raw_model_output=None,
                error_message="No document data provided.",
            )

        raw_b64 = request.document_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        file_bytes = base64.b64decode(raw_b64)

        return self.process_file_bytes(
            file_bytes=file_bytes,
            filename="document.pdf" if file_bytes.startswith(b"%PDF") else "document.png",
            schema=request.schema_definition,
            custom_instructions=request.raw_prompt_override,
        )
