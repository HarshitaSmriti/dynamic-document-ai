"""Application configuration settings for production and local environments."""

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

    # Server settings (Production Render / Local)
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

    # Model Backend Selector: 'hosted_api' | 'local'
    MODEL_BACKEND: str = Field(
        default="hosted_api",
        validation_alias=AliasChoices("MODEL_BACKEND", "BACKEND_TYPE"),
    )

    # Hosted Qwen API Settings (OpenRouter / Alibaba Cloud DashScope / OpenAI-compatible)
    QWEN_API_BASE: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias=AliasChoices("QWEN_API_BASE", "OPENROUTER_API_BASE", "OPENAI_API_BASE"),
    )
    QWEN_MODEL_NAME: str = Field(
        default="qwen/qwen2.5-vl-72b-instruct",
        validation_alias=AliasChoices("QWEN_MODEL_NAME", "OPENROUTER_MODEL_NAME", "MODEL_NAME"),
    )
    QWEN_API_KEY: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("QWEN_API_KEY", "OPENROUTER_API_KEY", "DASHSCOPE_API_KEY", "API_KEY"),
    )

    # Inference Parameters
    MAX_NEW_TOKENS: int = 2048
    TEMPERATURE: float = 0.1
    REQUEST_TIMEOUT: int = 60

    # Local Model Settings (For GPU-based local execution)
    QWEN_MODEL_PATH: Optional[str] = None
    DEVICE: str = "cuda"
    TORCH_DTYPE: str = "bfloat16"

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH) if ENV_FILE_PATH.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
