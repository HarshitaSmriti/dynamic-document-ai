"""FastAPI Application Main Entrypoint for Production and Development."""

import os
from fastapi import FastAPI, Response, status
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

# Standard Fast, Native Starlette CORS Middleware (No BaseHTTPMiddleware overhead)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"^https?:\/\/.*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include API endpoints (e.g. /extract, /extract/upload)
app.include_router(api_router)


@app.api_route("/", methods=["GET", "HEAD"], status_code=status.HTTP_200_OK)
def root():
    """Root status endpoint for uptime and Render health checks."""
    return {
        "app": settings.APP_NAME,
        "status": "online",
        "model": settings.QWEN_MODEL_NAME,
        "backend": settings.MODEL_BACKEND,
        "docs_url": "/docs",
    }


@app.api_route("/health", methods=["GET", "HEAD"], status_code=status.HTTP_200_OK)
def health_check():
    """Instantaneous health check endpoint for Render load balancer liveness probes."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "backend": settings.MODEL_BACKEND,
        "model": settings.QWEN_MODEL_NAME,
        "provider": {
            "backend": settings.MODEL_BACKEND,
            "model_name": settings.QWEN_MODEL_NAME,
            "status": "online",
        },
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
