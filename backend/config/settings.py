"""Type-Safe Settings and Configuration Management.

Loads configuration from environment variables (.env) and tenant YAML files
using Pydantic v2 BaseSettings with strict validation and sane local-first defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TENANTS_DIR = Path(__file__).resolve().parent / "tenants"


class VoiceSettings(BaseModel):
    """Acoustic voice settings for a persona."""

    provider: Literal["kokoro", "piper", "cartesia", "elevenlabs"] = "kokoro"
    voice_id: str = "af_sarah"
    fallback_voice_id: str = "af_bella"
    speed: float = Field(default=1.00, ge=0.70, le=1.40)
    pitch: float = Field(default=0.0, ge=-20.0, le=20.0)
    gender: Literal["female", "male", "neutral"] = "female"
    accent: str = "en-US"


class LLMSettings(BaseModel):
    """LLM model and generation settings."""

    provider: Literal["ollama", "gemini", "groq", "openai"] = "ollama"
    model: str = "llama3.2:3b"
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_tokens: int = Field(default=180, ge=50, le=2000)
    base_url: str | None = None
    api_key: str | None = None


class PersonaConfig(BaseModel):
    """Configuration for an autonomous specialist role."""

    key: str
    name: str = "Elena"
    role_description: str = "Front desk booking specialist and customer concierge"
    system_prompt: str = "You are Elena, a warm and efficient front desk booking assistant."
    tools: list[str] = Field(default_factory=lambda: ["hotel_check_availability", "hotel_book_room"])
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


class TenantConfig(BaseModel):
    """Full tenant configuration defining operations and active personas."""

    tenant_id: str = "grand-meridian"
    tenant_name: str = "Grand Meridian Hotel & Resort"
    timezone: str = "America/New_York"
    mode: Literal["single_role", "multi_service_router"] = "single_role"
    default_language: str = "en-US"
    enable_barge_in: bool = True
    active_persona: str = "hotel_booking_specialist"
    personas: dict[str, PersonaConfig] = Field(default_factory=dict)

    def get_active_persona(self) -> PersonaConfig:
        """Returns the currently active persona configuration."""
        if self.active_persona in self.personas:
            return self.personas[self.active_persona]
        # Fallback to first available or synthetic default
        if self.personas:
            return next(iter(self.personas.values()))
        return PersonaConfig(key="default")


class AppSettings(BaseSettings):
    """Root Application Configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Provider Toggles (Local-First Defaults)
    llm_provider: Literal["ollama", "gemini", "groq", "openai"] = "ollama"
    stt_provider: Literal["faster_whisper", "deepgram", "groq"] = "faster_whisper"
    tts_provider: Literal["kokoro", "piper", "cartesia", "elevenlabs"] = "kokoro"

    # Ollama Local Settings
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434/v1"

    # Local Speech Settings
    faster_whisper_model: str = "base.en"
    kokoro_voice_id: str = "af_sarah"
    kokoro_speed: float = 1.00
    kokoro_model_path: str = "./models/kokoro-v0_19.onnx"
    kokoro_voices_path: str = "./models/voices.bin"

    # Cloud AI Keys
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    deepgram_api_key: str | None = None
    cartesia_api_key: str | None = None
    cartesia_voice_id: str = "86e30c1d-714b-4074-a1f2-1cb6b552fb49"

    # Server Gateway
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    environment: Literal["development", "production", "testing"] = "development"
    admin_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    consumer_cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Security & Rate Limiting Controls (Phase 8A)
    admin_api_key: str | None = None
    voice_offer_rate_limit: int = 5
    voice_offer_rate_window_seconds: int = 60
    max_active_voice_sessions: int = 10
    voice_session_idle_timeout_seconds: int = 300

    # Storage Paths
    database_url: str = "sqlite:///./data/voice_agent.db"
    chroma_persist_dir: str = "./data/chroma_db"
    log_level: str = "INFO"

    # RAG Embeddings (local-first; nomic/fastembed are opt-in)
    embedding_provider: Literal["local_hash", "nomic_ollama", "fastembed"] = "local_hash"
    embedding_model: str = "nomic-embed-text"

    # Weather (stub stays offline; open_meteo is free and optional)
    weather_provider: Literal["stub", "open_meteo"] = "stub"

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_provider(cls, val: Any) -> str:
        if isinstance(val, str):
            return val.lower().strip()
        return val


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Returns singleton instance of AppSettings."""
    return AppSettings()


def load_tenant_config(tenant_file: str = "default_hotel.yaml") -> TenantConfig:
    """Loads and validates a tenant YAML configuration file."""
    path = TENANTS_DIR / tenant_file
    if not path.exists():
        return TenantConfig()

    raw_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return TenantConfig.model_validate(raw_data)
