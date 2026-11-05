"""TTS Service Factory: instantiates Kokoro, Piper, Cartesia, or ElevenLabs TTS."""

from __future__ import annotations

import logging
from typing import Any

from backend.config.settings import AppSettings, VoiceSettings, get_settings
from backend.engine.audio.voice_catalog import get_voice_profile
from backend.engine.providers.base import (
    MissingDependencyError,
    ProviderConfigurationError,
    TTSProviderType,
    UnsupportedProviderError,
)

logger = logging.getLogger(__name__)


def create_tts_service(
    settings: AppSettings | None = None,
    override: VoiceSettings | None = None,
) -> Any:
    """Creates and returns the configured TTS service instance."""
    app_cfg = settings or get_settings()
    provider: TTSProviderType = (override.provider if override else app_cfg.tts_provider)

    if provider == "kokoro":
        raw_voice_id = override.voice_id if override else app_cfg.kokoro_voice_id
        profile = get_voice_profile(raw_voice_id, provider="kokoro")
        voice_id = profile.voice_id
        speed = override.speed if override else app_cfg.kokoro_speed

        logger.info(
            "Initializing Kokoro TTS: voice_id=%s (%s), speed=%.2f",
            voice_id,
            profile.name,
            speed,
        )
        try:
            from pipecat.services.kokoro.tts import KokoroTTSService
        except ImportError as exc:
            raise MissingDependencyError(
                "pipecat-ai[kokoro] is required for Kokoro TTS."
            ) from exc

        return KokoroTTSService(voice_id=voice_id, speed=speed)

    elif provider == "cartesia":
        api_key = app_cfg.cartesia_api_key
        if not api_key:
            raise ProviderConfigurationError("CARTESIA_API_KEY is required for Cartesia TTS.")

        try:
            from pipecat.services.cartesia.tts import CartesiaTTSService
        except ImportError as exc:
            raise MissingDependencyError(
                "pipecat-ai[cartesia] is required for Cartesia TTS."
            ) from exc

        raw_voice_id = override.voice_id if override else app_cfg.cartesia_voice_id
        profile = get_voice_profile(raw_voice_id, provider="cartesia")
        voice_id = profile.voice_id

        logger.info("Initializing Cartesia TTS: voice_id=%s (%s)", voice_id, profile.name)
        return CartesiaTTSService(api_key=api_key, voice_id=voice_id)

    elif provider == "elevenlabs":
        api_key = getattr(app_cfg, "elevenlabs_api_key", None)
        if not api_key:
            raise ProviderConfigurationError("ELEVENLABS_API_KEY is required for ElevenLabs TTS.")

        try:
            from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
            raw_voice_id = override.voice_id if override else "21m00Tcm4TlvDq8ikWAM"
            profile = get_voice_profile(raw_voice_id, provider="elevenlabs")
            voice_id = profile.voice_id
            return ElevenLabsTTSService(api_key=api_key, voice_id=voice_id)
        except ImportError as exc:
            raise MissingDependencyError(
                "pipecat-ai[elevenlabs] is required for ElevenLabs TTS."
            ) from exc

    elif provider == "piper":
        try:
            from pipecat.services.piper.tts import PiperTTSService
            raw_voice_id = override.voice_id if override else "en_US-lessac-medium"
            profile = get_voice_profile(raw_voice_id, provider="piper")
            voice_id = profile.voice_id
            return PiperTTSService(voice_id=voice_id)
        except ImportError as exc:
            raise MissingDependencyError(
                "pipecat-ai[piper] is required for Piper TTS."
            ) from exc

    else:
        raise UnsupportedProviderError(f"Unsupported TTS provider: {provider}")
