"""Regression and integration tests for bounded consumer WebRTC, pipeline, and DataChannel contracts."""

from __future__ import annotations

import asyncio
import unittest
from collections import defaultdict
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pydantic import TypeAdapter, ValidationError

from backend.api.app import create_app
from backend.api.schemas import ClientUiEvent, ConsumerSessionRequest, VoiceOfferResponse
from backend.api.voice_sessions import ConsumerVoiceSession, ConsumerVoiceSessionManager
from backend.config.settings import AppSettings
from backend.engine.pipeline import (
    ConsumerVoicePipeline,
    DataChannelBridgeProcessor,
    SpeechNormalizerProcessor,
    emit_generative_ui_events,
)
from backend.rag.retriever import RAGContextFrameProcessor, RAGContextRetriever
from backend.rag.vector_store import MemoryVectorStore


class FakeConnection:
    """Minimal connection double that captures outbound browser messages and supports event_handler protocol."""

    def __init__(self, pc_id: str = "consumer-test") -> None:
        self.pc_id = pc_id
        self.sent_messages: list[dict[str, Any]] = []
        self._event_handlers: dict[str, list[Callable]] = defaultdict(list)

    def send_app_message(self, message: dict[str, Any]) -> None:
        self.sent_messages.append(message)

    def event_handler(self, event_name: str) -> Callable:
        """Decorator for registering event handlers (mirrors SmallWebRTCConnection API)."""
        def decorator(handler: Callable) -> Callable:
            self._event_handlers[event_name].append(handler)
            return handler
        return decorator

    def add_event_handler(self, event_name: str, handler: Callable) -> None:
        self._event_handlers[event_name].append(handler)

    async def connect(self) -> None:
        """No-op connect for test double."""

    async def close(self) -> None:
        """No-op close for test double."""


class TestVoiceSessions(unittest.TestCase):
    """Test suite for voice session manager and browser contracts."""

    def test_consumer_config_is_safe_and_browser_cors_is_explicit(self) -> None:
        """The consumer bootstrap exposes only non-secret metadata to an approved origin."""
        with TestClient(create_app()) as client:
            response = client.get(
                "/api/v1/voice/config?tenant_id=grand-meridian",
                headers={"Origin": "http://localhost:3000"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["agent_name"], "Elena")
        self.assertFalse(response.json()["payment_collection_available"])
        self.assertNotIn("api_key", response.text)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:3000")

    def test_data_channel_events_cannot_bypass_spoken_confirmation(self) -> None:
        """Only presentation/context events are accepted from the consumer browser."""
        adapter = TypeAdapter(ClientUiEvent)
        with self.assertRaises(ValidationError):
            adapter.validate_python({"type": "confirm_booking", "booking_id": "BK-101"})

    def test_touch_selection_and_confirmation_request_remain_non_committing(self) -> None:
        """Screen interaction is acknowledged but never routed to an MCP booking write."""
        manager = ConsumerVoiceSessionManager(AppSettings())
        connection = FakeConnection()
        session = ConsumerVoiceSession(
            connection=connection,  # type: ignore[arg-type] - focused transport double.
            request_context=ConsumerSessionRequest(),
        )
        manager._sessions[connection.pc_id] = session

        async def exercise_events() -> None:
            await manager._handle_app_message(
                connection,  # type: ignore[arg-type] - focused transport double.
                {"type": "client_touch_override", "room_id": "R-301", "label": "Executive Suite"},
            )
            await manager._handle_app_message(
                connection,  # type: ignore[arg-type] - focused transport double.
                {"type": "request_out_loud_confirmation", "subject": "Executive Suite reservation"},
            )

        asyncio.run(exercise_events())

        self.assertEqual(session.selected_room_id, "R-301")
        self.assertEqual(connection.sent_messages[0]["type"], "selection_acknowledged")
        self.assertEqual(connection.sent_messages[1]["type"], "confirmation_required")
        self.assertIn("aloud", str(connection.sent_messages[1]["message"]).lower())

    def test_text_message_event_injects_into_pipeline_context(self) -> None:
        """A text_message event updates session state and calls pipeline.inject_user_text."""
        manager = ConsumerVoiceSessionManager(AppSettings())
        connection = FakeConnection(pc_id="text-test-peer")

        # Create a mock pipeline with async inject methods
        mock_pipeline = MagicMock()
        mock_pipeline.inject_user_text = AsyncMock()
        mock_pipeline.inject_touch_override = AsyncMock()

        session = ConsumerVoiceSession(
            connection=connection,  # type: ignore[arg-type]
            request_context=ConsumerSessionRequest(),
            pipeline=mock_pipeline,
        )
        manager._sessions[connection.pc_id] = session

        async def exercise() -> None:
            await manager._handle_app_message(
                connection,  # type: ignore[arg-type]
                {"type": "text_message", "text": "What is the price for Executive Suite?"},
            )

        asyncio.run(exercise())

        self.assertEqual(session.last_text_message, "What is the price for Executive Suite?")
        mock_pipeline.inject_user_text.assert_called_once_with("What is the price for Executive Suite?")
        self.assertEqual(connection.sent_messages[0]["type"], "text_acknowledged")
        self.assertIn("concierge", connection.sent_messages[0]["message"].lower())

    def test_touch_override_calls_pipeline_inject(self) -> None:
        """A client_touch_override event calls pipeline.inject_touch_override (non-committing)."""
        manager = ConsumerVoiceSessionManager(AppSettings())
        connection = FakeConnection(pc_id="touch-test-peer")

        mock_pipeline = MagicMock()
        mock_pipeline.inject_user_text = AsyncMock()
        mock_pipeline.inject_touch_override = AsyncMock()

        session = ConsumerVoiceSession(
            connection=connection,  # type: ignore[arg-type]
            request_context=ConsumerSessionRequest(),
            pipeline=mock_pipeline,
        )
        manager._sessions[connection.pc_id] = session

        async def exercise() -> None:
            await manager._handle_app_message(
                connection,  # type: ignore[arg-type]
                {"type": "client_touch_override", "room_id": "R-201", "label": "Deluxe City View"},
            )

        asyncio.run(exercise())

        self.assertEqual(session.selected_room_id, "R-201")
        mock_pipeline.inject_touch_override.assert_called_once_with("R-201", "Deluxe City View")
        self.assertEqual(connection.sent_messages[0]["type"], "selection_acknowledged")
        # CRITICAL: booking writes never happen from touch events
        mock_pipeline.inject_user_text.assert_not_called()


class TestPipelineEmittersAndProcessors(unittest.TestCase):
    """Test suite for DataChannel emitters, pipeline processors, and text injection."""

    def test_speech_normalizer_processor(self) -> None:
        """Verify SpeechNormalizerProcessor expands currency and codes in TextFrame."""
        processor = SpeechNormalizerProcessor()
        frame = TextFrame("Room R-101 is $180/night for booking BK-101.")

        pushed_frames: list[Any] = []

        async def fake_push(f: Any, d: Any) -> None:
            pushed_frames.append(f)

        processor.push_frame = fake_push  # type: ignore[assignment]
        asyncio.run(processor.process_frame(frame, FrameDirection.DOWNSTREAM))

        self.assertEqual(len(pushed_frames), 1)
        normalized_text = pushed_frames[0].text
        self.assertIn("one hundred eighty dollars", normalized_text)
        self.assertIn("room 1 0 1", normalized_text)

    def test_datachannel_bridge_guest_captions(self) -> None:
        """Verify DataChannelBridgeProcessor emits captions for guest speech."""
        connection = FakeConnection()
        bridge = DataChannelBridgeProcessor(connection=connection, role="guest")

        async def run_frames() -> None:
            bridge.push_frame = lambda f, d: asyncio.sleep(0)  # type: ignore[assignment]
            await bridge.process_frame(
                InterimTranscriptionFrame(text="Hello Elena", user_id="u1", timestamp=1.0),
                FrameDirection.DOWNSTREAM,
            )
            await bridge.process_frame(
                TranscriptionFrame(text="Are standard rooms available?", user_id="u1", timestamp=2.0),
                FrameDirection.DOWNSTREAM,
            )

        asyncio.run(run_frames())

        self.assertEqual(len(connection.sent_messages), 2)
        self.assertEqual(connection.sent_messages[0]["type"], "caption")
        self.assertEqual(connection.sent_messages[0]["speaker"], "guest")
        self.assertEqual(connection.sent_messages[0]["is_final"], False)
        self.assertEqual(connection.sent_messages[0]["text"], "Hello Elena")

        self.assertEqual(connection.sent_messages[1]["type"], "caption")
        self.assertEqual(connection.sent_messages[1]["speaker"], "guest")
        self.assertEqual(connection.sent_messages[1]["is_final"], True)
        self.assertEqual(connection.sent_messages[1]["text"], "Are standard rooms available?")

    def test_datachannel_bridge_agent_captions(self) -> None:
        """Verify DataChannelBridgeProcessor emits captions for agent voice output."""
        connection = FakeConnection()
        bridge = DataChannelBridgeProcessor(connection=connection, role="agent")

        async def run_frames() -> None:
            bridge.push_frame = lambda f, d: asyncio.sleep(0)  # type: ignore[assignment]
            await bridge.process_frame(
                TextFrame("Welcome to the Grand Meridian Hotel. How can I help you today?"),
                FrameDirection.DOWNSTREAM,
            )

        asyncio.run(run_frames())

        self.assertEqual(len(connection.sent_messages), 1)
        self.assertEqual(connection.sent_messages[0]["type"], "caption")
        self.assertEqual(connection.sent_messages[0]["speaker"], "agent")
        self.assertEqual(connection.sent_messages[0]["is_final"], True)
        self.assertIn("Grand Meridian", connection.sent_messages[0]["text"])

    def test_datachannel_bridge_role_isolation(self) -> None:
        """A guest-role bridge must not emit agent captions (and vice versa)."""
        connection = FakeConnection()
        bridge = DataChannelBridgeProcessor(connection=connection, role="guest")

        async def run_frames() -> None:
            bridge.push_frame = lambda f, d: asyncio.sleep(0)  # type: ignore[assignment]
            # Agent TextFrame should NOT produce caption when role=guest
            await bridge.process_frame(
                TextFrame("Agent sentence should not appear."),
                FrameDirection.DOWNSTREAM,
            )

        asyncio.run(run_frames())
        self.assertEqual(len(connection.sent_messages), 0)

    def test_generative_ui_availability_events(self) -> None:
        """Verify hotel_check_availability emits room_card and booking_stepper events."""
        connection = FakeConnection()
        tool_args = {"date_in": "2026-11-14", "date_out": "2026-11-16"}
        tool_result = {
            "status": "ok",
            "rooms": [
                {"room_id": "R-101", "category": "standard", "nightly_rate": 120.0},
                {"room_id": "R-201", "category": "deluxe", "nightly_rate": 180.0},
            ],
            "message": "2 rooms are available.",
        }

        emit_generative_ui_events(connection, "hotel_check_availability", tool_args, tool_result)

        # Expect 2 room_card events + 1 booking_stepper event
        self.assertEqual(len(connection.sent_messages), 3)

        room_card_1 = connection.sent_messages[0]
        self.assertEqual(room_card_1["type"], "room_card")
        self.assertEqual(room_card_1["room"]["id"], "R-101")
        self.assertEqual(room_card_1["room"]["price_per_night"], 120.0)
        self.assertGreater(len(room_card_1["room"]["amenities"]), 0)

        room_card_2 = connection.sent_messages[1]
        self.assertEqual(room_card_2["type"], "room_card")
        self.assertEqual(room_card_2["room"]["id"], "R-201")
        self.assertEqual(room_card_2["room"]["price_per_night"], 180.0)

        stepper = connection.sent_messages[2]
        self.assertEqual(stepper["type"], "booking_stepper")
        self.assertEqual(stepper["booking"]["current_step"], "room")
        self.assertEqual(stepper["booking"]["check_in"], "2026-11-14")
        self.assertEqual(stepper["booking"]["check_out"], "2026-11-16")

    def test_generative_ui_book_room_needs_confirmation(self) -> None:
        """Verify hotel_book_room needs_confirmation emits confirmation_required and stepper."""
        connection = FakeConnection()
        tool_args = {
            "room_id": "R-101",
            "date_in": "2026-11-14",
            "date_out": "2026-11-16",
            "guest_name": "Alexander Hayes",
        }
        tool_result = {
            "status": "needs_confirmation",
            "message": "Room R-101 requires confirmation.",
            "total_price": 240.0,
        }

        emit_generative_ui_events(connection, "hotel_book_room", tool_args, tool_result)

        self.assertEqual(len(connection.sent_messages), 2)
        stepper = connection.sent_messages[0]
        self.assertEqual(stepper["type"], "booking_stepper")
        self.assertEqual(stepper["booking"]["current_step"], "confirmation")
        self.assertEqual(stepper["booking"]["total"], 240.0)

        confirmation = connection.sent_messages[1]
        self.assertEqual(confirmation["type"], "confirmation_required")
        self.assertIn("aloud", confirmation["message"].lower())
        self.assertIn("R-101", confirmation["message"])

    def test_generative_ui_confirmed_booking_emits_checkout_prompt(self) -> None:
        """A confirmed booking must actively produce the frontend checkout_prompt event."""
        connection = FakeConnection()
        tool_args = {
            "room_id": "R-301",
            "date_in": "2026-11-14",
            "date_out": "2026-11-16",
            "guest_name": "Alexander Hayes",
        }
        tool_result = {
            "status": "confirmed",
            "booking_id": "BK-SEED-001",
            "total_price": 520.0,
        }

        emit_generative_ui_events(connection, "hotel_book_room", tool_args, tool_result)

        message_types = [m["type"] for m in connection.sent_messages]
        self.assertEqual(message_types, ["booking_stepper", "checkout_prompt"])

        checkout = connection.sent_messages[1]
        self.assertEqual(checkout["amount"], 520.0)
        self.assertEqual(checkout["currency"], "USD")
        self.assertIn("Executive Grand Suite", checkout["description"])

        # Frontend type guard requires a finite numeric amount.
        self.assertIsInstance(checkout["amount"], float)

    def test_generative_ui_confirmed_booking_without_total_skips_checkout(self) -> None:
        """Malformed tool results must never emit a checkout_prompt with no amount."""
        connection = FakeConnection()
        emit_generative_ui_events(
            connection,
            "hotel_book_room",
            {"room_id": "R-101", "date_in": "2026-11-14", "date_out": "2026-11-16"},
            {"status": "confirmed"},
        )
        message_types = [m["type"] for m in connection.sent_messages]
        self.assertNotIn("checkout_prompt", message_types)

    def test_generative_ui_no_events_for_unknown_tool(self) -> None:
        """Unknown tool names must not emit any DataChannel events."""
        connection = FakeConnection()
        emit_generative_ui_events(connection, "weather_get_forecast", {}, {"status": "ok", "message": "Sunny."})
        self.assertEqual(len(connection.sent_messages), 0)

    def test_consumer_voice_pipeline_context_injection(self) -> None:
        """Verify ConsumerVoicePipeline.inject_user_text adds message to context and queues LLMRunFrame."""
        # Build a minimal pipeline object without a real transport or worker
        from backend.mcp.client import create_mcp_hub

        connection = FakeConnection(pc_id="unit-pipeline-test")

        async def test_injection() -> None:
            # Build context without transport
            mcp_hub = create_mcp_hub()
            context = LLMContext(
                messages=[{"role": "system", "content": "You are Elena, Grand Meridian concierge."}],
            )

            # Mock the worker's queue_frames
            mock_worker = MagicMock()
            mock_worker.queue_frames = AsyncMock()

            # Mock runner
            mock_runner = MagicMock()
            mock_runner.cancel = AsyncMock()

            pipeline_obj = ConsumerVoicePipeline(
                connection=connection,
                transport=MagicMock(),
                pipeline=MagicMock(),
                worker=mock_worker,
                runner=mock_runner,
                context=context,
                mcp_hub=mcp_hub,
                knowledge_processor=RAGContextFrameProcessor(
                    retriever=RAGContextRetriever(vector_store=MemoryVectorStore())
                ),
            )

            # Test text injection
            await pipeline_obj.inject_user_text("Can you tell me about the Grand Suite?")
            messages = pipeline_obj.context.messages
            user_messages = [m for m in messages if m.get("role") == "user"]
            self.assertEqual(len(user_messages), 1)
            self.assertEqual(user_messages[0]["content"], "Can you tell me about the Grand Suite?")
            mock_worker.queue_frames.assert_called_once()

            # Test touch override injection
            mock_worker.queue_frames.reset_mock()
            await pipeline_obj.inject_touch_override("R-301", "Executive Grand Suite")
            dev_messages = [m for m in pipeline_obj.context.messages if m.get("role") == "developer"]
            self.assertIn("Executive Grand Suite", dev_messages[-1]["content"])
            self.assertIn("verbal confirmation aloud", dev_messages[-1]["content"])
            mock_worker.queue_frames.assert_called_once()

        asyncio.run(test_injection())

    def test_pipeline_stop_is_idempotent(self) -> None:
        """ConsumerVoicePipeline.stop() must be safe to call multiple times without error."""
        from backend.mcp.client import create_mcp_hub

        connection = FakeConnection()
        mcp_hub = create_mcp_hub()
        context = LLMContext()

        mock_worker = MagicMock()
        mock_runner = MagicMock()
        mock_runner.cancel = AsyncMock()

        pipeline_obj = ConsumerVoicePipeline(
            connection=connection,
            transport=MagicMock(),
            pipeline=MagicMock(),
            worker=mock_worker,
            runner=mock_runner,
            context=context,
            mcp_hub=mcp_hub,
        )

        async def run_stop_twice() -> None:
            await pipeline_obj.stop()
            await pipeline_obj.stop()  # Second call must be a safe no-op

        asyncio.run(run_stop_twice())
        # cancel should only be called once (idempotent guard via _closed flag)
        mock_runner.cancel.assert_called_once()


class TestVoiceOfferResponseContract(unittest.TestCase):
    """The browser refuses an SDP answer whose final line is unterminated.

    Regression for the live-connect failure: ``StrictSchema`` enables
    ``str_strip_whitespace``, which removed the trailing CRLF aiortc emits.
    Chromium then rejected the answer (``a=setup:active Invalid SDP line``),
    so the peer connection never established and every session timed out.
    """

    def test_answer_sdp_keeps_its_line_terminator(self) -> None:
        """Whitespace stripping must not drop the trailing CRLF of an answer."""
        response = VoiceOfferResponse.model_validate(
            {
                "sdp": "v=0\r\no=- 1 1 IN IP4 0.0.0.0\r\na=setup:active\r\n",
                "type": "answer",
                "pc_id": "peer-1",
            }
        )
        self.assertTrue(
            response.sdp.endswith("\r\n"),
            f"Answer lost its line terminator: {response.sdp[-40:]!r}",
        )
        self.assertFalse(response.sdp.endswith("\r\n\r\n"))

    def test_answer_sdp_gains_a_terminator_when_missing(self) -> None:
        """An unterminated answer from upstream is repaired, not passed through."""
        response = VoiceOfferResponse.model_validate(
            {"sdp": "v=0\r\na=setup:active", "type": "answer", "pc_id": "peer-2"}
        )
        self.assertTrue(response.sdp.endswith("\r\n"))
        self.assertFalse(response.sdp.endswith("\r\n\r\n"))


if __name__ == "__main__":
    unittest.main()
