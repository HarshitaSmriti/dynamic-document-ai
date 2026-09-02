"""FastAPI Application Main Entrypoint for Production and Development."""

import os
from typing import Optional
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


@app.api_route("/", methods=["GET", "HEAD"], status_code=status.HTTP_204_NO_CONTENT)
@app.api_route("/health", methods=["GET", "HEAD"], status_code=status.HTTP_204_NO_CONTENT)
@app.api_route("/cron", methods=["GET", "HEAD"], status_code=status.HTTP_204_NO_CONTENT)
@app.api_route("/ping", methods=["GET", "HEAD"], status_code=status.HTTP_204_NO_CONTENT)
def cron_health_ping(code: Optional[int] = None):
    """Ultra-lightweight endpoint for cron-job.org and Render health checks.

    Returns HTTP 204 No Content with Content-Length: 0 and no response body by default.
    If ?code=200 is passed, returns HTTP 200 OK with minimal plain-text 'OK'.
    Explicitly ensures zero JSON dumps, HTML pages, model output, or heavy payloads.
    """
    if code == 200:
        return Response(
            content=b"OK",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
            headers={"Content-Length": "2"},
        )
    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers={"Content-Length": "0"},
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", settings.PORT))
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=port,
        reload=settings.DEBUG,
    )
