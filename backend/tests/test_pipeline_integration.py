"""End-to-end integration tests for the live consumer voice pipeline (Phase 6B).

These tests exercise the AGENTS.md section 7 runtime verification contract:

1. A real ``SmallWebRTCConnection`` gets an actively running Pipecat
   ``PipelineWorker`` (VAD -> STT -> LLM -> TTS) attached to it.
2. Every DataChannel event ``consumer_ui`` renders has a live producer
   (agent/guest captions, room cards, steppers, checkout prompt).
3. Typed and spoken inputs actually enter the LLM context and produce an
   agent turn instead of a static acknowledgement.
4. The data path is proven at runtime with real pipeline objects; the
   local-Ollama case additionally runs against the live model daemon.
"""

from __future__ import annotations

import asyncio
import unittest
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMRunFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

from backend.api.schemas import ConsumerSessionRequest
from backend.api.voice_sessions import ConsumerVoiceSessionManager
from backend.config.settings import AppSettings, load_tenant_config
from backend.engine.pipeline import create_consumer_voice_pipeline


class RecordingConnection(SmallWebRTCConnection):
    """A real SmallWebRTC connection with an in-memory DataChannel spy.

    Overriding only ``send_app_message`` keeps the genuine transport binding,
    event-handler registry, and peer-connection lifecycle under test while
    capturing every outbound browser event for assertions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sent_messages: list[dict[str, Any]] = []

    def send_app_message(self, message: Any) -> None:
        """Record outbound DataChannel payloads instead of writing to SCTP."""
        self.sent_messages.append(message)


class ScriptedLLMService(FrameProcessor):
    """Deterministic LLM stand-in: answers each inference with one sentence.

    Pipecat 1.12 signals text LLM services with ``LLMContextFrame`` (the user
    aggregator converts queued ``LLMRunFrame`` triggers into a context frame),
    so this stand-in reacts to both shapes exactly where a real LLM service
    sits in the graph. Used where a network-free roundtrip is required; the
    real local model is covered by ``test_local_llm_produces_an_agent_caption``.
    """

    def __init__(self, reply: str) -> None:
        super().__init__()
        self._reply = reply
        self.run_requests = 0

    async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMContextFrame, LLMRunFrame)):
            self.run_requests += 1
            await self.push_frame(TextFrame(self._reply), direction)
            return
        await self.push_frame(frame, direction)


class PassThroughSpeechProcessor(FrameProcessor):
    """Keep pipeline tests independent from local speech model downloads."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class FakePeerConnection:
    """Minimal peer-connection double for session-manager failure-path tests."""

    def __init__(self, pc_id: str = "failed-pipeline-peer") -> None:
        self.pc_id = pc_id
        self.sent_messages: list[dict[str, Any]] = []
        self._handlers: dict[str, list[Callable]] = {}

    def event_handler(self, event_name: str) -> Callable:
        """Register a handler using the SmallWebRTC decorator protocol."""

        def decorator(handler: Callable) -> Callable:
            self._handlers.setdefault(event_name, []).append(handler)
            return handler

        return decorator

    def send_app_message(self, message: dict[str, Any]) -> None:
        self.sent_messages.append(message)

    async def connect(self) -> None:
        """No-op connect for the test double."""

    async def close(self) -> None:
        """No-op close for the test double."""


async def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll a predicate on the event loop until it holds or the deadline passes."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return predicate()


def ollama_reachable() -> bool:
    """Probe the local Ollama daemon without raising."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2):
            return True
    except Exception:
        return False


def captions_for(messages: list[dict[str, Any]], speaker: str) -> list[dict[str, Any]]:
    """Filter recorded DataChannel messages down to one speaker's captions."""
    return [
        m for m in messages if m.get("type") == "caption" and m.get("speaker") == speaker
    ]


class TestLivePipelineStartup(unittest.TestCase):
    """6B item 1: a real PipelineWorker attaches to a real WebRTC connection."""

    def test_pipeline_builds_starts_and_stops_on_a_real_connection(self) -> None:
        """Construction, worker start, and teardown on a genuine connection."""
        scripted = ScriptedLLMService("Welcome to the Grand Meridian. How may I help you?")

        async def scenario() -> None:
            connection = RecordingConnection()
            tenant = load_tenant_config()
            with (
                patch("backend.engine.pipeline.create_llm_service", return_value=scripted),
                patch(
                    "backend.engine.pipeline.create_stt_service",
                    return_value=PassThroughSpeechProcessor(),
                ),
                patch(
                    "backend.engine.pipeline.create_tts_service",
                    return_value=PassThroughSpeechProcessor(),
                ),
            ):
                session = await create_consumer_voice_pipeline(
                    connection=connection,
                    tenant=tenant,
                    persona_key="hotel_booking_specialist",
                    settings=AppSettings(),
                )

            # The transport registered its lifecycle handlers on THIS connection
            # (the keys exist at init, so assert the handler lists are filled).
            handlers = connection._event_handlers  # noqa: SLF001 - registry under test.
            for event_name, expected_handler in (
                ("connected", "on_connected"),
                ("disconnected", "on_disconnected"),
                ("closed", "on_closed"),
                ("app-message", "on_app_message"),
            ):
                registered = getattr(handlers[event_name], "handlers", [])
                self.assertTrue(
                    any(
                        getattr(handler, "__name__", "") == expected_handler
                        for handler in registered
                    ),
                    f"Transport never bound {expected_handler} to {event_name}",
                )

            # MCP tool schemas reached the LLM context (persona tool budget).
            tools = session.context.tools
            self.assertIsNotNone(tools)
            self.assertEqual(len(tools.standard_tools), 4)

            try:
                await session.start()
                await asyncio.sleep(0.5)
                self.assertIsNotNone(session.runner_task)
                self.assertFalse(
                    session.runner_task.done(),
                    "Pipeline worker died shortly after start",
                )
            finally:
                await session.stop()

            self.assertTrue(session._closed)
            self.assertIsNone(session.runner_task)

        asyncio.run(scenario())

    def test_typed_text_roundtrip_produces_an_agent_caption(self) -> None:
        """6B item 3: typed text enters context and returns as a live caption."""
        scripted = ScriptedLLMService("Our Executive Grand Suite is two hundred sixty dollars per night.")

        async def scenario() -> None:
            connection = RecordingConnection()
            tenant = load_tenant_config()
            with (
                patch("backend.engine.pipeline.create_llm_service", return_value=scripted),
                patch(
                    "backend.engine.pipeline.create_stt_service",
                    return_value=PassThroughSpeechProcessor(),
                ),
                patch(
                    "backend.engine.pipeline.create_tts_service",
                    return_value=PassThroughSpeechProcessor(),
                ),
            ):
                session = await create_consumer_voice_pipeline(
                    connection=connection,
                    tenant=tenant,
                    persona_key="hotel_booking_specialist",
                    settings=AppSettings(),
                )

            try:
                await session.start()
                await session.inject_user_text("What does the Executive Suite cost?")

                delivered = await wait_until(
                    lambda: bool(captions_for(connection.sent_messages, "agent")),
                    timeout=5.0,
                )
                self.assertTrue(
                    delivered,
                    f"No agent caption returned; captured: {connection.sent_messages}",
                )
                caption = captions_for(connection.sent_messages, "agent")[0]
                self.assertTrue(caption["text"].strip())
                self.assertTrue(caption["is_final"])

                # The payload really entered the LLM context (not just an ack).
                user_messages = [
                    m
                    for m in session.context.messages
                    if m.get("role") == "user"
                    and m.get("content") == "What does the Executive Suite cost?"
                ]
                self.assertEqual(len(user_messages), 1)
                self.assertEqual(scripted.run_requests, 1)
            finally:
                await session.stop()

        asyncio.run(scenario())

    def test_spoken_utterance_feeds_confirmation_gate_and_captions(self) -> None:
        """6B item 2/3: final STT text arms the spoken confirmation gate."""
        scripted = ScriptedLLMService("Of course, one moment please.")

        async def scenario() -> None:
            connection = RecordingConnection()
            tenant = load_tenant_config()
            with (
                patch("backend.engine.pipeline.create_llm_service", return_value=scripted),
                patch(
                    "backend.engine.pipeline.create_stt_service",
                    return_value=PassThroughSpeechProcessor(),
                ),
                patch(
                    "backend.engine.pipeline.create_tts_service",
                    return_value=PassThroughSpeechProcessor(),
                ),
            ):
                session = await create_consumer_voice_pipeline(
                    connection=connection,
                    tenant=tenant,
                    persona_key="hotel_booking_specialist",
                    settings=AppSettings(),
                )

            try:
                await session.start()
                self.assertEqual(session.mcp_hub.session.last_user_message, "")

                await session.worker.queue_frames(
                    [
                        TranscriptionFrame(
                            text="Yes, please confirm the booking",
                            user_id="guest-1",
                            timestamp=datetime.now(UTC).isoformat(),
                        )
                    ]
                )

                gate_updated = await wait_until(
                    lambda: session.mcp_hub.session.last_user_message
                    == "Yes, please confirm the booking",
                    timeout=5.0,
                )
                self.assertTrue(
                    gate_updated,
                    "Spoken utterance never reached the confirmation gate",
                )

                guest_captions = await wait_until(
                    lambda: bool(captions_for(connection.sent_messages, "guest")),
                    timeout=5.0,
                )
                self.assertTrue(
                    guest_captions,
                    f"No guest caption emitted; captured: {connection.sent_messages}",
                )
                guest_caption = captions_for(connection.sent_messages, "guest")[0]
                self.assertTrue(guest_caption["is_final"])
                self.assertIn("confirm", guest_caption["text"])
            finally:
                await session.stop()

        asyncio.run(scenario())

    def test_spoken_pii_is_masked_before_context_and_captions(self) -> None:
        """Spoken payment data is redacted before reaching context or the browser."""
        scripted = ScriptedLLMService("I can help with that payment.")

        async def scenario() -> None:
            connection = RecordingConnection()
            tenant = load_tenant_config()
            with (
                patch("backend.engine.pipeline.create_llm_service", return_value=scripted),
                patch(
                    "backend.engine.pipeline.create_stt_service",
                    return_value=PassThroughSpeechProcessor(),
                ),
                patch(
                    "backend.engine.pipeline.create_tts_service",
                    return_value=PassThroughSpeechProcessor(),
                ),
            ):
                session = await create_consumer_voice_pipeline(
                    connection=connection,
                    tenant=tenant,
                    persona_key="hotel_booking_specialist",
                    settings=AppSettings(),
                )

            raw_text = "My card number is 4532 0150 1234 5671 and the CVV is 842."
            try:
                await session.start()
                await session.worker.queue_frames(
                    [
                        TranscriptionFrame(
                            text=raw_text,
                            user_id="guest-1",
                            timestamp=datetime.now(UTC).isoformat(),
                        )
                    ]
                )

                redacted = await wait_until(
                    lambda: any(
                        message.get("role") == "user"
                        and "[CARD REDACTED]" in message.get("content", "")
                        for message in session.context.messages
                    ),
                    timeout=5.0,
                )
                self.assertTrue(redacted, "Spoken card data was not redacted in LLM context")
                user_context = next(
                    message["content"]
                    for message in session.context.messages
                    if message.get("role") == "user"
                    and "[CARD REDACTED]" in message.get("content", "")
                )
                self.assertIn("[CVV REDACTED]", user_context)
                self.assertNotIn("4532", user_context)
                self.assertNotIn("842", user_context)

                guest_caption_ready = await wait_until(
                    lambda: bool(captions_for(connection.sent_messages, "guest")),
                    timeout=5.0,
                )
                self.assertTrue(guest_caption_ready, "No guest caption was emitted")
                guest_caption = captions_for(connection.sent_messages, "guest")[0]["text"]
                self.assertIn("[CARD REDACTED]", guest_caption)
                self.assertIn("[CVV REDACTED]", guest_caption)
                self.assertNotIn("4532", guest_caption)
                self.assertNotIn("842", guest_caption)
                self.assertNotIn("4532", session.mcp_hub.session.last_user_message)
                self.assertNotIn("842", session.mcp_hub.session.last_user_message)

                typed_text = "My tax ID is 12-3456789 and my card is 4532015012345671."
                await session.inject_user_text(typed_text)
                self.assertEqual(
                    session.mcp_hub.session.last_user_message,
                    "My tax ID is [TAX ID REDACTED] and my card is [CARD REDACTED].",
                )

                await session.inject_touch_override(
                    "R-101",
                    "My card number is 4532015012345671",
                )
                touch_context = next(
                    message["content"]
                    for message in reversed(session.context.messages)
                    if message.get("role") == "developer"
                    and "Guest tapped" in message.get("content", "")
                )
                self.assertIn("[CARD REDACTED]", touch_context)
                self.assertNotIn("4532015012345671", touch_context)
            finally:
                await session.stop()

        asyncio.run(scenario())

    @unittest.skipUnless(ollama_reachable(), "Local Ollama daemon not reachable")
    def test_local_llm_produces_an_agent_caption(self) -> None:
        """6B item 4: a real local LLM roundtrip over the running pipeline."""
        timeout = 45.0 if ollama_reachable() else 0.0

        async def scenario() -> None:
            connection = RecordingConnection()
            tenant = load_tenant_config()
            # No factory patch: this uses the real configured local provider.
            with (
                patch(
                    "backend.engine.pipeline.create_stt_service",
                    return_value=PassThroughSpeechProcessor(),
                ),
                patch(
                    "backend.engine.pipeline.create_tts_service",
                    return_value=PassThroughSpeechProcessor(),
                ),
            ):
                session = await create_consumer_voice_pipeline(
                    connection=connection,
                    tenant=tenant,
                    persona_key="hotel_booking_specialist",
                    settings=AppSettings(),
                )
            try:
                await session.start()
                await session.inject_user_text(
                    "Reply with exactly one short sentence: welcome a hotel guest."
                )
                delivered = await wait_until(
                    lambda: bool(captions_for(connection.sent_messages, "agent")),
                    timeout=timeout,
                )
                self.assertTrue(
                    delivered,
                    "Local LLM never produced a captioned agent turn",
                )
            finally:
                await session.stop()

        asyncio.run(scenario())


class TestHonestFailurePath(unittest.TestCase):
    """AGENTS.md section 7.3: no fake acknowledgements when the pipeline is down."""

    def test_pipeline_start_failure_reports_ui_error_not_a_fake_ack(self) -> None:
        """A dead pipeline must reject text/touch input honestly."""

        async def scenario() -> None:
            manager = ConsumerVoiceSessionManager(AppSettings())
            connection = FakePeerConnection()
            tenant = load_tenant_config()

            with patch(
                "backend.api.voice_sessions.create_consumer_voice_pipeline",
                side_effect=RuntimeError("simulated pipeline failure"),
            ):
                await manager._register_connection(
                    connection,  # type: ignore[arg-type] - focused transport double.
                    ConsumerSessionRequest(),
                    tenant,
                )

            session = manager._sessions[connection.pc_id]
            self.assertTrue(session.pipeline_failed)
            self.assertIsNone(session.pipeline)

            # The browser is told the agent is not live right after connect.
            message_types = [m["type"] for m in connection.sent_messages]
            self.assertIn("session_ready", message_types)
            self.assertIn("ui_error", message_types)

            await manager._handle_app_message(
                connection,  # type: ignore[arg-type] - focused transport double.
                {"type": "text_message", "text": "Is the suite available?"},
            )
            await manager._handle_app_message(
                connection,  # type: ignore[arg-type] - focused transport double.
                {"type": "client_touch_override", "room_id": "R-301", "label": "Executive Grand Suite"},
            )

            replies = [m["type"] for m in connection.sent_messages[-2:]]
            self.assertEqual(replies, ["ui_error", "ui_error"])
            self.assertNotIn("text_acknowledged", replies)
            self.assertNotIn("selection_acknowledged", replies)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
