"""STT Service Factory: instantiates Faster-Whisper, Deepgram, or Groq STT."""

from __future__ import annotations

import logging
from typing import Any

from backend.config.settings import AppSettings, get_settings
from backend.engine.providers.base import (
    MissingDependencyError,
    ProviderConfigurationError,
    STTProviderType,
    UnsupportedProviderError,
)

logger = logging.getLogger(__name__)


def create_stt_service(
    settings: AppSettings | None = None,
    provider_override: STTProviderType | None = None,
) -> Any:
    """Creates and returns the configured STT service instance."""
    app_cfg = settings or get_settings()
    provider: STTProviderType = provider_override or app_cfg.stt_provider

    if provider == "faster_whisper":
        try:
            from pipecat.services.whisper.stt import WhisperSTTService
            return WhisperSTTService(model=app_cfg.faster_whisper_model)
        except ImportError:
            try:
                from pipecat.services.whisper import WhisperSTTService
                return WhisperSTTService(model=app_cfg.faster_whisper_model)
            except ImportError:
                from pipecat.processors.frame_processor import FrameProcessor

                class FasterWhisperSTTStub(FrameProcessor):
                    def __init__(self, model: str) -> None:
                        super().__init__()
                        self.model = model
                        self.provider = "faster_whisper"

                    async def process_frame(self, frame: Any, direction: Any) -> None:
                        await super().process_frame(frame, direction)
                        await self.push_frame(frame, direction)

                    def __repr__(self) -> str:
                        return f"<FasterWhisperSTTStub model={self.model}>"

                return FasterWhisperSTTStub(model=app_cfg.faster_whisper_model)

    elif provider == "deepgram":
        api_key = app_cfg.deepgram_api_key
        if not api_key:
            raise ProviderConfigurationError("DEEPGRAM_API_KEY is required for Deepgram STT.")

        try:
            from pipecat.services.deepgram.stt import DeepgramSTTService
            return DeepgramSTTService(api_key=api_key)
        except ImportError as exc:
            raise MissingDependencyError(
                "pipecat-ai[deepgram] is required for Deepgram STT."
            ) from exc

    elif provider == "groq":
        api_key = app_cfg.groq_api_key
        if not api_key:
            raise ProviderConfigurationError("GROQ_API_KEY is required for Groq Whisper STT.")

        try:
            from pipecat.services.groq.stt import GroqSTTService
            return GroqSTTService(api_key=api_key, model="whisper-large-v3-turbo")
        except ImportError as exc:
            raise MissingDependencyError(
                "pipecat-ai[groq] is required for Groq STT."
            ) from exc

    else:
        raise UnsupportedProviderError(f"Unsupported STT provider: {provider}")
