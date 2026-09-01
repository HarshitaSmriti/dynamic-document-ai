"""FastAPI Application Main Entrypoint for Production and Development."""

import os
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

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


class UniversalCORSMiddleware(BaseHTTPMiddleware):
    """Custom middleware guaranteeing CORS headers on every response, including error states."""

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            response = Response(status_code=status.HTTP_200_OK)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response

        try:
            response = await call_next(request)
        except Exception as exc:
            response = Response(
                content=f'{{"detail":"Internal error: {str(exc)}"}}',
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                media_type="application/json",
            )

        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response


# Add universal CORS middleware
app.add_middleware(UniversalCORSMiddleware)

# Standard Starlette CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"^https?:\/\/.*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
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
