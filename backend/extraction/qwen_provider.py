"""Vision-Language Model Provider Abstraction and Implementation.

Provides:
1. BaseVLMProvider (VisionLanguageProvider): Abstract interface for multimodal model inference.
2. LocalQwenVLProvider (LocalQwenProvider): Direct Hugging Face Transformers inference using
   the official Apache-2.0 licensed Qwen/Qwen2.5-VL-7B-Instruct model with hardware-aware
   CUDA GPU / CPU fallback.
3. OpenAICompatibleVLMProvider (HostedQwenProvider): Optional generic OpenAI-compatible
   endpoint provider for self-hosted GPU inference servers (vLLM, Ollama, TGI).
4. get_vlm_provider: Factory function to retrieve the configured provider.
"""

import base64
import io
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import requests

from backend.config import Settings, get_settings

logger = logging.getLogger("document_ai.provider")


class BaseVLMProvider(ABC):
    """Abstract base interface for Vision-Language Model inference."""

    @abstractmethod
    def generate(
        self,
        images: List[Union[Image.Image, str]],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text output from images and textual prompt."""
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
        """Generate text and return execution metadata (latency, device, tokens)."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return provider configuration and operational status."""
        pass


# Conceptual alias for provider substitution
VisionLanguageProvider = BaseVLMProvider


class LocalQwenVLProvider(BaseVLMProvider):
    """Direct local Qwen2.5-VL inference using Hugging Face Transformers.

    Uses the official Apache-2.0 licensed Qwen/Qwen2.5-VL-7B-Instruct model.
    Features hardware-aware execution (CUDA GPU acceleration with automatic CPU fallback),
    thread-safe singleton model caching, and image token optimization.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.model_id = self.settings.QWEN_MODEL_ID or "Qwen/Qwen2.5-VL-7B-Instruct"
        self._model = None
        self._processor = None

        # Setup Hugging Face cache if configured
        if self.settings.HF_HOME:
            os.environ["HF_HOME"] = str(self.settings.HF_HOME)

        # Detect hardware and choose appropriate torch device/dtype
        self.device, self.torch_dtype, self.device_map = self._detect_hardware()

    def _detect_hardware(self) -> Tuple[str, Any, Optional[str]]:
        """Detect available compute hardware and select optimal device, dtype, and device mapping."""
        import torch

        req_device = (self.settings.QWEN_DEVICE or "auto").strip().lower()
        req_dtype = (self.settings.QWEN_TORCH_DTYPE or "auto").strip().lower()

        cuda_available = torch.cuda.is_available()

        if req_device in ["auto", "cuda", "gpu"]:
            if cuda_available:
                device = "cuda"
                device_map = "auto"
                logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}. GPU acceleration enabled.")
            else:
                device = "cpu"
                device_map = None
                logger.warning(
                    "CUDA is not available. Falling back to CPU inference. "
                    "Note: 7B model execution on CPU will have higher latency."
                )
        elif req_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            device_map = None
        else:
            device = "cpu"
            device_map = None

        # Select appropriate precision
        if req_dtype == "auto":
            if device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif device == "mps":
                dtype = torch.float16
            else:
                dtype = torch.float32
        elif req_dtype in ["bfloat16", "bf16"]:
            dtype = torch.bfloat16
        elif req_dtype in ["float16", "fp16"]:
            dtype = torch.float16
        elif req_dtype in ["float32", "fp32"]:
            dtype = torch.float32
        else:
            dtype = torch.float32

        return device, dtype, device_map

    def load_model(self) -> None:
        """Load Qwen2.5-VL model and processor once into memory (cached singleton)."""
        if self._model is not None and self._processor is not None:
            return

        with self._lock:
            if self._model is not None and self._processor is not None:
                return

            try:
                import torch
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

                logger.info(f"Loading processor and model '{self.model_id}' (device={self.device}, dtype={self.torch_dtype})...")

                # Configure bounded pixel dimensions to prevent excessive token count and high VRAM usage
                # Standard document optimal bounds: min 256*28*28, max 1280*28*28
                min_pixels = 256 * 28 * 28
                max_pixels = 1280 * 28 * 28

                self._processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    trust_remote_code=True,
                )

                if self.device_map:
                    self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_id,
                        torch_dtype=self.torch_dtype,
                        device_map=self.device_map,
                        trust_remote_code=True,
                    )
                else:
                    self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_id,
                        torch_dtype=self.torch_dtype,
                        trust_remote_code=True,
                    )
                    self._model.to(self.device)

                self._model.eval()
                logger.info(f"Model '{self.model_id}' loaded successfully on {self.device}.")

            except Exception as exc:
                logger.error(f"MODEL_LOAD_ERROR: Failed to load '{self.model_id}': {exc}", exc_info=True)
                raise RuntimeError(
                    f"MODEL_LOAD_ERROR: Failed to load local Qwen2.5-VL model '{self.model_id}'. "
                    f"Details: {str(exc)}"
                ) from exc

    def _prepare_images(self, images: List[Union[Image.Image, str]]) -> List[Image.Image]:
        """Convert input images (PIL Images or base64 strings) into normalized RGB PIL Images."""
        pil_images: List[Image.Image] = []
        for img in images:
            if isinstance(img, str):
                b64_str = img
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_str)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                pil_images.append(pil_img)
            elif isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB") if img.mode != "RGB" else img)
            else:
                raise TypeError(f"IMAGE_PROCESSING_ERROR: Expected PIL Image or base64 string, got {type(img)}")
        return pil_images

    def generate_with_metadata(
        self,
        images: List[Union[Image.Image, str]],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Run multimodal inference directly on local Qwen2.5-VL weights."""
        self.load_model()

        import torch

        pil_images = self._prepare_images(images)
        tokens_limit = max_new_tokens or self.settings.QWEN_MAX_NEW_TOKENS or 1024
        temp = temperature if temperature is not None else (self.settings.QWEN_TEMPERATURE or 0.0)

        # Build Qwen chat conversation
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        user_content: List[Dict[str, Any]] = []
        for img in pil_images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": user_content})

        start_time = time.perf_counter()

        try:
            # Extract vision inputs using qwen_vl_utils if available, or direct PIL image extraction
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                image_inputs = pil_images
                video_inputs = None

            # Apply chat template
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize & encode vision inputs
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Move inputs to target model device
            target_device = next(self._model.parameters()).device
            inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

            # Generate output tokens
            with torch.inference_mode():
                gen_kwargs = {
                    "max_new_tokens": tokens_limit,
                }
                if temp > 0.0:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = temp
                else:
                    gen_kwargs["do_sample"] = False

                generated_ids = self._model.generate(**inputs, **gen_kwargs)

                # Trim input prompt tokens from output
                input_len = inputs["input_ids"].shape[1]
                generated_ids_trimmed = [
                    out_ids[input_len:] for out_ids in generated_ids
                ]

                output_text = self._processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

            latency = time.perf_counter() - start_time
            tokens_count = len(generated_ids_trimmed[0]) if generated_ids_trimmed else 0

            metadata = {
                "provider": "LocalQwenVLProvider",
                "model_used": self.model_id,
                "device": self.device,
                "dtype": str(self.torch_dtype),
                "latency_seconds": round(latency, 3),
                "tokens_generated": tokens_count,
            }

            return output_text.strip(), metadata

        except Exception as exc:
            latency = time.perf_counter() - start_time
            logger.error(f"MODEL_INFERENCE_ERROR: {exc}", exc_info=True)
            raise RuntimeError(
                f"MODEL_INFERENCE_ERROR: Direct inference failed on '{self.model_id}'. "
                f"Details: {str(exc)}"
            ) from exc

    def generate(
        self,
        images: List[Union[Image.Image, str]],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Run multimodal inference and return generated text string."""
        text, _ = self.generate_with_metadata(
            images=images,
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return text

    def get_status(self) -> Dict[str, Any]:
        """Return provider status and configuration details."""
        return {
            "provider_type": "LocalQwenVLProvider",
            "backend": "local",
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "is_loaded": self._model is not None,
            "cuda_available": self.device == "cuda",
            "license": "Apache-2.0",
        }


# Alias for backward compatibility
LocalQwenProvider = LocalQwenVLProvider


class OpenAICompatibleVLMProvider(BaseVLMProvider):
    """Optional generic provider for self-hosted OpenAI-compatible Vision endpoints.

    Can be used to connect to self-hosted vLLM, Ollama, TGI, or custom GPU inference servers.
    Does not depend on paid proprietary APIs or hardcoded tokens.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.api_base = (self.settings.QWEN_API_BASE or "https://openrouter.ai/api/v1").rstrip("/")
        self.api_key = self.settings.QWEN_API_KEY
        raw_model = self.settings.QWEN_MODEL_ID or "qwen/qwen2.5-vl-72b-instruct"
        if "openrouter.ai" in self.api_base.lower():
            if "qwen-2.5-vl" in raw_model.lower():
                raw_model = raw_model.replace("qwen-2.5-vl", "qwen2.5-vl")
            if raw_model.endswith(":free"):
                raw_model = raw_model.replace(":free", "")
        self.model_name = raw_model
        self.timeout = self.settings.REQUEST_TIMEOUT

    def _image_to_data_uri(self, image: Union[Image.Image, str]) -> str:
        """Convert a PIL Image or base64 string to a compact data URI."""
        if isinstance(image, str):
            if image.startswith("data:image"):
                return image
            return f"data:image/jpeg;base64,{image}"

        buffered = io.BytesIO()
        rgb_img = image.convert("RGB") if image.mode != "RGB" else image
        if max(rgb_img.size) > 512:
            scale = 512 / float(max(rgb_img.size))
            rgb_img = rgb_img.resize(
                (int(rgb_img.width * scale), int(rgb_img.height * scale)),
                Image.Resampling.BILINEAR,
            )
        rgb_img.save(buffered, format="JPEG", quality=75, optimize=True)
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    def generate_with_metadata(
        self,
        images: List[Union[Image.Image, str]],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute multimodal inference call against OpenAI-compatible vision endpoint."""
        endpoint = f"{self.api_base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key.strip():
            headers["Authorization"] = f"Bearer {self.api_key.strip()}"

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            data_uri = self._image_to_data_uri(img)
            user_content.append({"type": "image_url", "image_url": {"url": data_uri}})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        req_max_tokens = min(max_new_tokens or self.settings.QWEN_MAX_NEW_TOKENS or 512, 512)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": req_max_tokens,
            "temperature": temperature if temperature is not None else (self.settings.QWEN_TEMPERATURE or 0.0),
        }

        start_time = time.perf_counter()
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
            
            # If 402 (low credits), retry once with a tighter token window
            if response.status_code == 402 and req_max_tokens > 512:
                payload["max_tokens"] = 512
                response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)

            latency = time.perf_counter() - start_time

            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code} from self-hosted endpoint '{endpoint}': {response.text}"
                )

            res_json = response.json()
            choices = res_json.get("choices", [])
            if not choices:
                raise ValueError(f"Provider returned an empty choices list: {res_json}")

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                raise ValueError("Provider returned empty message content.")

            metadata = {
                "provider": "OpenAICompatibleVLMProvider",
                "http_status": response.status_code,
                "latency_seconds": round(latency, 3),
                "model_used": res_json.get("model", self.model_name),
                "usage": res_json.get("usage", {}),
            }
            return content.strip(), metadata

        except Exception as exc:
            latency = time.perf_counter() - start_time
            raise RuntimeError(f"MODEL_INFERENCE_ERROR: Remote endpoint call failed: {exc}") from exc

    def generate(
        self,
        images: List[Union[Image.Image, str]],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        content, _ = self.generate_with_metadata(
            images=images,
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return content

    def get_status(self) -> Dict[str, Any]:
        return {
            "provider_type": "OpenAICompatibleVLMProvider",
            "backend": "hosted_api",
            "model_name": self.model_name,
            "api_base": self.api_base,
            "timeout_seconds": self.timeout,
        }


# Alias for backward compatibility
HostedQwenProvider = OpenAICompatibleVLMProvider


def get_vlm_provider(settings: Optional[Settings] = None) -> BaseVLMProvider:
    """Factory function returning the configured VLM provider instance.

    Defaults to LocalQwenVLProvider (direct Transformers inference on Apache-2.0 Qwen2.5-VL-7B).
    """
    cfg = settings or get_settings()
    backend = (cfg.MODEL_BACKEND or "local").lower().strip()

    if backend in ["local", "transformers", "qwen", "direct"]:
        return LocalQwenVLProvider(settings=cfg)
    elif backend in ["hosted_api", "remote", "vllm", "openai", "custom_server"]:
        return OpenAICompatibleVLMProvider(settings=cfg)
    else:
        logger.warning(f"Unrecognized MODEL_BACKEND='{backend}'. Defaulting to 'local' Transformers provider.")
        return LocalQwenVLProvider(settings=cfg)
