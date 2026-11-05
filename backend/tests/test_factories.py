"""Unit tests for configuration and AI provider factories.

Written using standard Python `unittest` (0 external test-runner dependencies required).
Fully compatible with both `python -m unittest` and `pytest`.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config.settings import AppSettings, LLMSettings, VoiceSettings, load_tenant_config
from backend.engine.providers.base import (
    MissingDependencyError,
    ProviderConfigurationError,
    UnsupportedProviderError,
)
from backend.engine.providers.llm_factory import create_llm_service
from backend.engine.providers.stt_factory import create_stt_service
from backend.engine.providers.tts_factory import create_tts_service


class TestConfiguration(unittest.TestCase):
    """Tests for type-safe Pydantic v2 settings and tenant configs."""

    def test_default_settings_local_first(self):
        """Validates that default configuration is 100% local-first ($0 compute)."""
        settings = AppSettings(
            _env_file=None,  # Do not read host .env during clean unit test
            llm_provider="ollama",
            stt_provider="faster_whisper",
            tts_provider="kokoro",
        )
        self.assertEqual(settings.llm_provider, "ollama")
        self.assertEqual(settings.stt_provider, "faster_whisper")
        self.assertEqual(settings.tts_provider, "kokoro")
        self.assertEqual(settings.ollama_model, "llama3.2:3b")
        self.assertEqual(settings.ollama_base_url, "http://localhost:11434/v1")
        self.assertEqual(settings.port, 8000)

    def test_tenant_yaml_loading(self):
        """Validates loading the default hotel tenant YAML."""
        tenant = load_tenant_config("default_hotel.yaml")
        self.assertEqual(tenant.tenant_id, "grand-meridian")
        self.assertIn("Grand Meridian", tenant.tenant_name)

        persona = tenant.get_active_persona()
        self.assertEqual(persona.name, "Elena")
        self.assertEqual(persona.voice.provider, "cartesia")
        self.assertEqual(persona.voice.voice_id, "86e30c1d-714b-4074-a1f2-1cb6b552fb49")
        self.assertEqual(persona.voice.speed, 1.00)
        self.assertEqual(persona.llm.provider, "groq")
        self.assertEqual(persona.llm.model, "openai/gpt-oss-120b")
        self.assertIn("hotel_check_availability", persona.tools)


class TestLLMFactory(unittest.TestCase):
    """Tests for LLM Service Factory."""

    def test_missing_groq_api_key_raises_error(self):
        """Requesting Groq without an API key must raise ProviderConfigurationError."""
        settings = AppSettings(_env_file=None, groq_api_key=None)
        override = LLMSettings(provider="groq")
        with self.assertRaises(ProviderConfigurationError) as ctx:
            create_llm_service(settings, override=override)
        self.assertIn("GROQ_API_KEY is required", str(ctx.exception))

    def test_missing_openai_api_key_raises_error(self):
        """Requesting OpenAI without an API key must raise ProviderConfigurationError."""
        settings = AppSettings(_env_file=None, openai_api_key=None)
        override = LLMSettings(provider="openai")
        with self.assertRaises(ProviderConfigurationError) as ctx:
            create_llm_service(settings, override=override)
        self.assertIn("OPENAI_API_KEY is required", str(ctx.exception))

    def test_missing_gemini_api_key_raises_error(self):
        """Requesting Gemini without an API key must raise ProviderConfigurationError."""
        settings = AppSettings(_env_file=None, gemini_api_key=None)
        override = LLMSettings(provider="gemini")
        with self.assertRaises(ProviderConfigurationError) as ctx:
            create_llm_service(settings, override=override)
        self.assertIn("GEMINI_API_KEY is required", str(ctx.exception))

    def test_unsupported_provider_raises_error(self):
        """Requesting an invalid provider must raise UnsupportedProviderError."""
        settings = AppSettings(_env_file=None)
        invalid_override = LLMSettings.model_construct(provider="non_existent_llm")
        with self.assertRaises(UnsupportedProviderError):
            create_llm_service(settings, override=invalid_override)

    @patch("pipecat.services.openai.llm.OpenAILLMService", create=True)
    def test_ollama_llm_creation(self, mock_openai_service):
        """Ollama provider must initialize with base_url and model."""
        mock_instance = MagicMock()
        mock_openai_service.return_value = mock_instance

        with patch.dict(
            "sys.modules",
            {"pipecat.services.openai.llm": MagicMock(OpenAILLMService=mock_openai_service)},
        ):
            settings = AppSettings(
                _env_file=None,
                llm_provider="ollama",
                ollama_model="llama3.2:3b",
                ollama_base_url="http://localhost:11434/v1",
            )
            service = create_llm_service(settings)
            self.assertEqual(service, mock_instance)
            mock_openai_service.assert_called_once_with(
                api_key="ollama",
                base_url="http://localhost:11434/v1",
                model="llama3.2:3b",
            )


class TestSTTFactory(unittest.TestCase):
    """Tests for STT Service Factory."""

    def test_missing_deepgram_key_raises_error(self):
        """Deepgram requires DEEPGRAM_API_KEY."""
        settings = AppSettings(_env_file=None, deepgram_api_key=None)
        with self.assertRaises(ProviderConfigurationError) as ctx:
            create_stt_service(settings, provider_override="deepgram")
        self.assertIn("DEEPGRAM_API_KEY is required", str(ctx.exception))

    def test_missing_groq_stt_key_raises_error(self):
        """Groq STT requires GROQ_API_KEY."""
        settings = AppSettings(_env_file=None, groq_api_key=None)
        with self.assertRaises(ProviderConfigurationError) as ctx:
            create_stt_service(settings, provider_override="groq")
        self.assertIn("GROQ_API_KEY is required", str(ctx.exception))

    def test_faster_whisper_default_creation(self):
        """Default STT creates the configured Faster-Whisper service."""
        settings = AppSettings(_env_file=None, stt_provider="faster_whisper")
        mock_service = MagicMock()
        module = MagicMock(WhisperSTTService=mock_service)

        with patch.dict(sys.modules, {"pipecat.services.whisper.stt": module}):
            service = create_stt_service(settings)

        self.assertEqual(service, mock_service.return_value)
        mock_service.assert_called_once_with(model="base.en")

    def test_missing_faster_whisper_dependency_raises_error(self):
        """Missing speech dependencies must not silently produce a pass-through stub."""
        settings = AppSettings(_env_file=None, stt_provider="faster_whisper")

        with patch.dict(
            sys.modules,
            {
                "pipecat.services.whisper.stt": None,
                "pipecat.services.whisper": None,
            },
        ):
            with self.assertRaises(MissingDependencyError) as ctx:
                create_stt_service(settings)

        self.assertIn("pipecat-ai[whisper]", str(ctx.exception))


class TestTTSFactory(unittest.TestCase):
    """Tests for TTS Service Factory."""

    def test_missing_cartesia_key_raises_error(self):
        """Cartesia requires CARTESIA_API_KEY."""
        settings = AppSettings(_env_file=None, cartesia_api_key=None)
        override = VoiceSettings(provider="cartesia")
        with self.assertRaises(ProviderConfigurationError) as ctx:
            create_tts_service(settings, override=override)
        self.assertIn("CARTESIA_API_KEY is required", str(ctx.exception))

    def test_kokoro_default_creation(self):
        """Kokoro TTS instantiates with voice_id and speed."""
        settings = AppSettings(_env_file=None, tts_provider="kokoro")
        mock_service = MagicMock()
        module = MagicMock(KokoroTTSService=mock_service)

        with patch.dict(sys.modules, {"pipecat.services.kokoro.tts": module}):
            service = create_tts_service(settings)

        self.assertEqual(service, mock_service.return_value)
        mock_service.assert_called_once_with(voice_id="af_sarah", speed=1.00)

    def test_missing_kokoro_dependency_raises_error(self):
        """Missing Kokoro dependencies must not silently produce a pass-through stub."""
        settings = AppSettings(_env_file=None, tts_provider="kokoro")

        with patch.dict(sys.modules, {"pipecat.services.kokoro.tts": None}):
            with self.assertRaises(MissingDependencyError) as ctx:
                create_tts_service(settings)

        self.assertIn("pipecat-ai[kokoro]", str(ctx.exception))

    def test_unsupported_tts_provider_raises_error(self):
        """Requesting invalid TTS provider raises UnsupportedProviderError."""
        settings = AppSettings(_env_file=None)
        invalid_voice = VoiceSettings.model_construct(provider="unknown_tts")
        with self.assertRaises(UnsupportedProviderError):
            create_tts_service(settings, override=invalid_voice)


if __name__ == "__main__":
    unittest.main()
