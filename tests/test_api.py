"""API endpoint tests using FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "provider" in data
    assert data["provider"]["backend"] == "hosted_api"


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
