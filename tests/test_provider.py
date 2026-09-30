"""Unit tests for VLM Provider abstraction, LocalQwenVLProvider, and provider factory."""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from backend.config import Settings
from backend.extraction.qwen_provider import (
    BaseVLMProvider,
    LocalQwenVLProvider,
    OpenAICompatibleVLMProvider,
    get_vlm_provider,
)
from backend.extraction.service import DocumentExtractionService
from backend.schemas.extraction import ExtractionSchema, DynamicFieldDefinition, FieldType


def test_get_vlm_provider_factory():
    local_cfg = Settings(MODEL_BACKEND="local", QWEN_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct")
    provider = get_vlm_provider(local_cfg)
    assert isinstance(provider, LocalQwenVLProvider)
    assert provider.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"

    remote_cfg = Settings(MODEL_BACKEND="hosted_api", QWEN_API_BASE="http://gpu-server:8000/v1")
    remote_provider = get_vlm_provider(remote_cfg)
    assert isinstance(remote_provider, OpenAICompatibleVLMProvider)
    assert remote_provider.api_base == "http://gpu-server:8000/v1"


def test_hardware_detection_cpu_fallback():
    with patch("torch.cuda.is_available", return_value=False):
        cfg = Settings(MODEL_BACKEND="local", QWEN_DEVICE="auto", QWEN_TORCH_DTYPE="auto")
        provider = LocalQwenVLProvider(cfg)
        device, dtype, device_map = provider._detect_hardware()
        assert device == "cpu"
        assert device_map is None


def test_hardware_detection_cuda():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"), \
         patch("torch.cuda.is_bf16_supported", return_value=True):
        cfg = Settings(MODEL_BACKEND="local", QWEN_DEVICE="cuda", QWEN_TORCH_DTYPE="auto")
        provider = LocalQwenVLProvider(cfg)
        device, dtype, device_map = provider._detect_hardware()
        assert device == "cuda"
        assert device_map == "auto"


def test_service_extraction_with_mock_provider():
    class MockVLMProvider(BaseVLMProvider):
        def generate(self, images, prompt, system_prompt=None, max_new_tokens=None, temperature=None):
            return """```json
{
  "document_type": "tax_invoice",
  "document": {
    "invoice_number": "INV-2026-001",
    "total_amount": 1250.00,
    "issuer": "Global Tech Corp"
  },
  "tables": [
    {
      "table_name": "Line_Items",
      "headers": ["Description", "Amount"],
      "rows": [["Cloud Hosting", "1250.00"]]
    }
  ],
  "lists": [],
  "warnings": []
}
```"""

        def generate_with_metadata(self, images, prompt, system_prompt=None, max_new_tokens=None, temperature=None):
            text = self.generate(images, prompt, system_prompt, max_new_tokens, temperature)
            return text, {"model_used": "mock", "latency_seconds": 0.05}

        def get_status(self):
            return {"provider_type": "MockVLMProvider", "backend": "mock"}

    mock_provider = MockVLMProvider()
    service = DocumentExtractionService(provider=mock_provider)

    test_img = Image.new("RGB", (100, 100), color="white")
    response = service.extract_from_images([test_img])

    assert response.success is True
    assert response.data.document_type == "tax_invoice"
    assert response.data.document["invoice_number"] == "INV-2026-001"
    assert response.data.document["total_amount"] == 1250.00
    assert len(response.data.tables) == 1
    assert response.data.tables[0]["table_name"] == "Line_Items"


def test_service_empty_image_error():
    service = DocumentExtractionService()
    response = service.extract_from_images([])
    assert response.success is False
    assert "IMAGE_PROCESSING_ERROR" in response.error_message or "empty" in response.error_message.lower()
