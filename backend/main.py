"""FastAPI Application Main Entrypoint for Production and Development."""

import os
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router as api_router
from backend.config import get_settings
from backend.extraction.service import DocumentExtractionService

settings = get_settings()
service = DocumentExtractionService()

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Dynamic Enterprise Document AI Extraction Platform with Vision-Language Models",
    debug=settings.DEBUG,
)

# Parse CORS origins
cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
if not cors_origins or "*" in cors_origins:
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints (e.g. /health, /extract, /extract/upload)
app.include_router(api_router)


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """Root status endpoint for uptime checks."""
    return {
        "app": settings.APP_NAME,
        "status": "online",
        "model": settings.QWEN_MODEL_NAME,
        "backend": settings.MODEL_BACKEND,
        "docs_url": "/docs",
    }


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint for Render / load balancer liveness probe."""
    provider_status = service.provider.get_status()
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "backend": settings.MODEL_BACKEND,
        "provider": provider_status,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", settings.PORT))
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=port,
        reload=settings.DEBUG,
    )
