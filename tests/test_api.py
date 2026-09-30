"""API endpoint tests using FastAPI TestClient."""

import io
import json
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "docs_url" in data


def test_ping_endpoint():
    response = client.get("/ping")
    assert response.status_code == 204
    assert response.content == b""

    response_200 = client.get("/ping?code=200")
    assert response_200.status_code == 200
    assert response_200.text == "OK"


def test_health_endpoints_all_aliases():
    for path in ["/health", "/healthz", "/api/health", "/api/v1/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "provider" in data
        assert "Qwen" in data["model"]


def test_extract_get_informational():
    for path in ["/extract", "/api/extract", "/api/v1/extract"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert "endpoint" in data

    for path in ["/extract/upload", "/api/extract/upload", "/api/v1/extract/upload"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert "endpoint" in data


def test_extract_endpoint_without_image():
    payload = {
        "raw_prompt_override": "Extract test fields",
    }
    response = client.post("/api/v1/extract", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "data" in data
    assert data["data"]["document_type"] == "error"
