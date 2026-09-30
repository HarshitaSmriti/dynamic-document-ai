"""Application configuration settings for production and local environments."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE_PATH = Path(__file__).resolve().parent.parent / ".env"

# Explicitly load .env file into environment if it exists (for local development)
if ENV_FILE_PATH.exists():
    load_dotenv(dotenv_path=ENV_FILE_PATH, override=False)


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    # Server settings (Production / Local)
    HOST: str = Field(default="0.0.0.0", validation_alias=AliasChoices("HOST", "SERVER_HOST"))
    PORT: int = Field(default=8000, validation_alias=AliasChoices("PORT", "SERVER_PORT"))
    DEBUG: bool = Field(default=False, validation_alias=AliasChoices("DEBUG", "APP_DEBUG"))
    APP_NAME: str = "Dynamic Enterprise Document AI"

    # Backend and AI Service URLs
    BACKEND_API_URL: str = Field(
        default="http://localhost:8000",
        validation_alias=AliasChoices("BACKEND_API_URL", "BACKEND_URL"),
    )
    AI_SERVICE_URL: str = Field(
        default="http://localhost:8000",
        validation_alias=AliasChoices("AI_SERVICE_URL", "AI_URL"),
    )
    CORS_ORIGINS: str = Field(
        default="*",
        validation_alias=AliasChoices("CORS_ORIGINS", "ALLOWED_ORIGINS"),
    )

    # Model Backend Selector: 'local' (Hugging Face Transformers) | 'hosted_api' (Self-hosted OpenAI/vLLM endpoint)
    MODEL_BACKEND: str = Field(
        default="local",
        validation_alias=AliasChoices("MODEL_BACKEND", "BACKEND_TYPE"),
    )

    # Local Direct Qwen Transformers Model Settings (Apache-2.0 licensed)
    QWEN_MODEL_ID: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        validation_alias=AliasChoices("QWEN_MODEL_ID", "QWEN_MODEL_PATH", "QWEN_MODEL_NAME", "MODEL_NAME"),
    )
    QWEN_DEVICE: str = Field(
        default="auto",
        validation_alias=AliasChoices("QWEN_DEVICE", "DEVICE"),
    )
    QWEN_TORCH_DTYPE: str = Field(
        default="auto",
        validation_alias=AliasChoices("QWEN_TORCH_DTYPE", "TORCH_DTYPE"),
    )

    # Inference Hyperparameters
    QWEN_MAX_NEW_TOKENS: int = Field(
        default=1024,
        validation_alias=AliasChoices("QWEN_MAX_NEW_TOKENS", "MAX_NEW_TOKENS"),
    )
    QWEN_TEMPERATURE: float = Field(
        default=0.0,
        validation_alias=AliasChoices("QWEN_TEMPERATURE", "TEMPERATURE"),
    )
    REQUEST_TIMEOUT: int = Field(
        default=180,
        validation_alias=AliasChoices("REQUEST_TIMEOUT", "TIMEOUT"),
    )

    # Optional Hugging Face Cache Directory
    HF_HOME: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("HF_HOME", "HUGGINGFACE_HUB_CACHE"),
    )

    # Optional Self-Hosted API Provider Settings (for vLLM / Ollama / OpenAI-compatible local/remote GPU servers)
    QWEN_API_BASE: Optional[str] = Field(
        default="http://localhost:8000/v1",
        validation_alias=AliasChoices("QWEN_API_BASE", "OPENAI_API_BASE", "VLLM_API_BASE"),
    )
    QWEN_API_KEY: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("QWEN_API_KEY", "API_KEY"),
    )

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH) if ENV_FILE_PATH.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
