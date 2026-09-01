"""Qwen Vision-Language Model Provider Implementation.

Supports:
1. HostedQwenProvider: OpenAI-compatible Vision API (OpenRouter, Alibaba Cloud Model Studio / DashScope, DeepInfra, vLLM)
2. LocalQwenProvider: Placeholder for local GPU execution with isolated model loading.

Isolated provider layer with replaceable backend architecture.
"""

import base64
import io
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from PIL import Image

from backend.config import Settings, get_settings


class BaseVLMProvider(ABC):
    """Abstract base class for Vision-Language Model providers."""

    @abstractmethod
    def generate(
        self,
        images: List[Union[Image.Image, str]],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from image and prompt inputs."""
        pass

    @abstractmethod
    def generate_with_metadata(
        self,
        images: List[Union[Image.Image, str]],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text and return execution metadata (status code, latency, model)."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return provider status and configuration details."""
        pass


class HostedQwenProvider(BaseVLMProvider):
    """Hosted Qwen2.5-VL Provider using OpenAI-Compatible Vision API.

    Compatible with OpenRouter, Alibaba Cloud Model Studio (DashScope), DeepInfra, and vLLM servers.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.api_base = (self.settings.QWEN_API_BASE or "https://openrouter.ai/api/v1").rstrip("/")
        self.api_key = self.settings.QWEN_API_KEY
        self.model_name = self.settings.QWEN_MODEL_NAME or "qwen/qwen-2.5-vl-7b-instruct:free"
        self.timeout = self.settings.REQUEST_TIMEOUT

    def _image_to_data_uri(self, image: Union[Image.Image, str]) -> str:
        """Convert a PIL Image or base64 string to a valid data URI."""
        if isinstance(image, str):
            if image.startswith("data:image"):
                return image
            return f"data:image/jpeg;base64,{image}"

        # PIL Image conversion
        buffered = io.BytesIO()
        image_format = image.format or "JPEG"
        if image_format.upper() not in ["JPEG", "PNG", "WEBP"]:
            image_format = "JPEG"
        image.save(buffered, format=image_format)
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        mime_type = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
        return f"data:{mime_type};base64,{b64_str}"

    def generate_with_metadata(
        self,
        images: List[Union[Image.Image, str]],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute multimodal inference call against OpenAI-compatible endpoint with metadata."""
        endpoint = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/enterprise-doc-ai",
            "X-Title": "Dynamic Document AI Extractor",
        }
        if self.api_key and self.api_key.strip():
            headers["Authorization"] = f"Bearer {self.api_key.strip()}"

        # Build user message content with text and image(s)
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        for img in images:
            data_uri = self._image_to_data_uri(img)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": data_uri}
            })

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens or self.settings.MAX_NEW_TOKENS,
            "temperature": temperature if temperature is not None else self.settings.TEMPERATURE,
        }

        start_time = time.perf_counter()
        response = None

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            latency = time.perf_counter() - start_time

            if response.status_code == 401:
                raise PermissionError(
                    f"HTTP 401 Unauthorized: Invalid or missing API key on '{self.api_base}'. Provider message: {response.text}"
                )
            elif response.status_code == 404:
                raise RuntimeError(
                    f"HTTP 404 Not Found: Model '{self.model_name}' or path at '{endpoint}' not found."
                )
            elif response.status_code == 429:
                raise RuntimeError(
                    f"HTTP 429 Rate Limit Exceeded on '{self.model_name}'. Provider message: {response.text}"
                )
            elif response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code} Error: {response.text}"
                )

            res_json = response.json()
            choices = res_json.get("choices", [])
            if not choices:
                raise ValueError(f"Provider returned an empty choices list: {res_json}")

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                raise ValueError("Provider returned empty message content.")

            metadata = {
                "http_status": response.status_code,
                "latency_seconds": round(latency, 3),
                "model_used": res_json.get("model", self.model_name),
                "usage": res_json.get("usage", {}),
            }

            return content, metadata

        except Exception as exc:
            latency = time.perf_counter() - start_time
            # Attach response status code to exception if available
            status_code = getattr(response, "status_code", None)
            setattr(exc, "http_status", status_code)
            setattr(exc, "latency_seconds", round(latency, 3))
            raise exc

    def generate(
        self,
        images: List[Union[Image.Image, str]],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Execute multimodal inference and return text output."""
        content, _ = self.generate_with_metadata(
            images=images,
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return content

    def get_status(self) -> Dict[str, Any]:
        """Return provider health status and configuration details."""
        return {
            "provider_type": "HostedQwenProvider",
            "backend": "hosted_api",
            "model_name": self.model_name,
            "api_base": self.api_base,
            "api_key_configured": bool(self.api_key and len(self.api_key.strip()) > 5),
            "timeout_seconds": self.timeout,
        }


class LocalQwenProvider(BaseVLMProvider):
    """Local Qwen2.5-VL Provider placeholder for future GPU-based inference."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.model_path = self.settings.QWEN_MODEL_PATH
        self.device = self.settings.DEVICE
        self._model = None
        self._processor = None

    def load_model(self) -> None:
        """Load weights once into GPU memory."""
        if self._model is not None:
            return
        raise NotImplementedError(
            "Local GPU inference is not enabled in this environment. "
            "Please use MODEL_BACKEND=hosted_api."
        )

    def generate_with_metadata(
        self,
        images: List[Union[Image.Image, str]],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        self.load_model()
        raise NotImplementedError("Local generation not active.")

    def generate(
        self,
        images: List[Union[Image.Image, str]],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        self.load_model()
        raise NotImplementedError("Local generation not active.")

    def get_status(self) -> Dict[str, Any]:
        return {
            "provider_type": "LocalQwenProvider",
            "backend": "local",
            "model_path": self.model_path,
            "device": self.device,
            "loaded": self._model is not None,
        }


def get_vlm_provider(settings: Optional[Settings] = None) -> BaseVLMProvider:
    """Factory function returning the configured VLM provider instance."""
    cfg = settings or get_settings()
    if cfg.MODEL_BACKEND == "hosted_api":
        return HostedQwenProvider(settings=cfg)
    elif cfg.MODEL_BACKEND == "local":
        return LocalQwenProvider(settings=cfg)
    else:
        raise ValueError(f"Unknown MODEL_BACKEND: '{cfg.MODEL_BACKEND}'. Must be 'hosted_api' or 'local'.")
