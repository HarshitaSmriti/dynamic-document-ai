"""Document Extraction Service.

Coordinates document pre-processing (PDF rendering, image decoding),
ultra-fast lightweight parallel multi-page dynamic extraction, VLM inference, and intelligent document-level consolidation.
Optimized for zero-timeout execution on Render Free Tier (< 20s total latency, < 60MB RAM).
"""

import base64
import concurrent.futures
import gc
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
    """Production-ready dynamic document extraction service supporting fast, lightweight multi-page extraction."""

    def __init__(self, provider: Optional[BaseVLMProvider] = None):
        self.provider = provider or get_vlm_provider()
        self.prompt_builder = DynamicPromptBuilder()
        self.parser = DynamicJSONParser()

    def decode_base64_image(self, base64_str: str) -> Image.Image:
        """Decode base64 string to PIL Image with memory optimization."""
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        image_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._optimize_image(img)

    def _optimize_image(self, img: Image.Image, max_dim: int = 1200) -> Image.Image:
        """Resize image if dimensions exceed max_dim to ensure fast tokenization and low memory."""
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.Resampling.BILINEAR)
        return img

    def render_pdf_bytes_to_images(self, pdf_bytes: bytes, dpi: int = 96) -> List[Image.Image]:
        """Convert PDF byte stream into highly optimized PIL Images for rapid inference."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("jpeg"))).convert("RGB")
            img = self._optimize_image(img)
            images.append(img)
        doc.close()
        gc.collect()
        return images

    def _extract_single_page(
        self,
        img: Union[Image.Image, str],
        page_idx: int,
        total_pages: int,
        schema: Optional[ExtractionSchema],
        custom_instructions: Optional[str],
    ) -> Tuple[int, bool, Optional[ExtractedDocumentPayload], Optional[str], Optional[str]]:
        """Worker function to process one page concurrently."""
        prompt = self.prompt_builder.build_extraction_prompt(
            schema=schema,
            custom_instructions=custom_instructions,
            page_num=page_idx,
            total_pages=total_pages,
        )
        try:
            raw_out = self.provider.generate(
                images=[img],
                prompt=prompt,
                system_prompt=self.prompt_builder.SYSTEM_PROMPT,
                max_new_tokens=2048,
            )
            success, page_payload, parse_err = self.parser.parse_and_validate(
                raw_output=raw_out,
                schema=schema,
            )
            return (page_idx, success, page_payload, raw_out, parse_err)
        except Exception as exc:
            return (page_idx, False, None, None, str(exc))

    def extract_from_images(
        self,
        images: List[Union[Image.Image, str]],
        schema: Optional[ExtractionSchema] = None,
        custom_instructions: Optional[str] = None,
    ) -> ExtractionResponse:
        """Extract dynamic structured data from single or multiple document page images in parallel."""
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
                    max_new_tokens=2048,
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

        # Ultra-fast parallel multi-page extraction (all pages extracted simultaneously)
        page_results: List[Tuple[int, bool, Optional[ExtractedDocumentPayload], Optional[str], Optional[str]]] = []
        max_workers = min(6, total_pages)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._extract_single_page,
                    img=img,
                    page_idx=idx,
                    total_pages=total_pages,
                    schema=schema,
                    custom_instructions=custom_instructions,
                )
                for idx, img in enumerate(images, 1)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    page_results.append(res)
                except Exception as exc:
                    page_results.append((0, False, None, None, str(exc)))

        # Sort results by page index to preserve strict sequential ordering
        page_results.sort(key=lambda x: x[0])

        page_payloads: List[ExtractedDocumentPayload] = []
        raw_outputs: List[str] = []
        all_warnings: List[str] = []

        for page_idx, success, payload, raw_out, err in page_results:
            if raw_out:
                raw_outputs.append(f"--- PAGE {page_idx} RAW OUTPUT ---\n{raw_out}")
            if success and payload:
                page_payloads.append(payload)
                if err:
                    all_warnings.append(f"Page {page_idx} note: {err}")
            else:
                all_warnings.append(f"Page {page_idx} error: {err}")

        # Clean memory immediately
        del images
        gc.collect()

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
                images = self.render_pdf_bytes_to_images(file_bytes, dpi=96)
            except Exception as exc:
                return ExtractionResponse(
                    success=False,
                    data=ExtractedDocumentPayload(document_type="error"),
                    error_message=f"Failed to render PDF document: {str(exc)}",
                )
        else:
            try:
                img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                img = self._optimize_image(img)
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
