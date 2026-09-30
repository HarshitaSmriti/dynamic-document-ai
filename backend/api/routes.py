"""FastAPI Router for Dynamic Extraction Endpoints."""

import io
import json
from typing import Optional
import anyio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.config import get_settings
from backend.extraction.service import DocumentExtractionService
from backend.schemas.extraction import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractionSchema,
)

router = APIRouter(tags=["Extraction"])
settings = get_settings()
service = DocumentExtractionService(settings=settings)


@router.api_route("/api/v1/health", methods=["GET", "HEAD"], status_code=status.HTTP_200_OK)
def api_health_check():
    """Health check endpoint reporting backend service and model status."""
    provider_status = service.provider.get_status()
    model_name = (
        provider_status.get("model_id")
        or provider_status.get("model_name")
        or settings.QWEN_MODEL_ID
    )
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "backend": settings.MODEL_BACKEND,
        "model": model_name,
        "device": provider_status.get("device", "auto"),
        "provider": provider_status,
    }


@router.post("/extract", response_model=ExtractionResponse)
@router.post("/api/v1/extract", response_model=ExtractionResponse)
async def extract_document(request: ExtractionRequest):
    """Dynamic document extraction endpoint receiving JSON payload with base64 image or PDF.
    
    Offloaded to worker thread via anyio.to_thread.run_sync to keep the event loop non-blocking.
    """
    try:
        response = await anyio.to_thread.run_sync(service.process_request, request)
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction processing error: {str(exc)}",
        )


@router.post("/extract/upload", response_model=ExtractionResponse)
@router.post("/api/v1/extract/upload", response_model=ExtractionResponse)
async def extract_document_upload(
    file: UploadFile = File(..., description="Document file (PDF, PNG, JPG, JPEG, WEBP)"),
    schema_json: Optional[str] = Form(
        None, description="Optional dynamic extraction schema as JSON string"
    ),
    custom_instructions: Optional[str] = Form(
        None, description="Optional custom extraction guidance"
    ),
):
    """Dynamic document extraction endpoint receiving multipart file upload (single image or multi-page PDF).
    
    Offloads heavy image processing and VLM execution to worker threads so the main event loop
    remains non-blocking and responsive.
    """
    try:
        content = await file.read()

        schema: Optional[ExtractionSchema] = None
        if schema_json and schema_json.strip():
            schema_dict = json.loads(schema_json)
            schema = ExtractionSchema(**schema_dict)

        # Offload synchronous extraction to thread pool
        response = await anyio.to_thread.run_sync(
            lambda: service.process_file_bytes(
                file_bytes=content,
                filename=file.filename or "document.pdf",
                schema=schema,
                custom_instructions=custom_instructions,
            )
        )
        return response

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid schema_json provided in multipart form data.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File extraction failed: {str(exc)}",
        )
