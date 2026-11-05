"""Live multimodal Pipecat voice pipeline and generative DataChannel emitters.

Connects SmallWebRTCTransport, Silero VAD, STT, PII Sanitizer, LLM context + MCP tools,
Sentence Aggregator, DataChannel Bridge, Speech Normalizer, and TTS into an
active, real-time voice pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMRunFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from backend.config.settings import AppSettings, TenantConfig, get_settings
from backend.engine.audio.normalizer import normalize_speech_text
from backend.engine.audio.pii_sanitizer import PIISanitizerProcessor, sanitize_pii_text
from backend.engine.providers.llm_factory import create_llm_service
from backend.engine.providers.stt_factory import create_stt_service
from backend.engine.providers.tts_factory import create_tts_service
from backend.mcp.client import MCPHub, create_mcp_hub
from backend.mcp.session import AgentSession

logger = logging.getLogger(__name__)

# Curated room presentation metadata for consumer generative UI
ROOM_PRESENTATION: dict[str, dict[str, Any]] = {
    "R-101": {
        "name": "Classic King Room",
        "description": "Refined king room with tailored linens, executive work desk, and city views.",
        "amenities": ["Complimentary Wi-Fi", "Rainfall Shower", "Smart TV", "Work Desk"],
        "bed_label": "1 King Bed",
    },
    "R-201": {
        "name": "Deluxe City View Room",
        "description": "Expanded layout with floor-to-ceiling skyline views, deep soaking tub, and espresso bar.",
        "amenities": ["Skyline View", "Espresso Bar", "Soaking Tub", "High-speed Wi-Fi"],
        "bed_label": "1 King or 2 Queens",
    },
    "R-301": {
        "name": "Executive Grand Suite",
        "description": "Top-floor panorama with separate living lounge, private dining nook, and concierge lounge access.",
        "amenities": ["Panoramic Skyline", "Living Lounge", "Lounge Access", "Breakfast Included"],
        "bed_label": "1 California King",
    },
}


def _get_room_presentation(room_id: str, category: str = "") -> dict[str, Any]:
    """Return polished UI presentation data for a room ID or category."""
    if room_id in ROOM_PRESENTATION:
        return ROOM_PRESENTATION[room_id]
    label = category.replace("_", " ").title() if category else room_id
    return {
        "name": f"{label} Room ({room_id})",
        "description": f"Elegant {label.lower()} accommodation at the Grand Meridian Hotel.",
        "amenities": ["Complimentary Wi-Fi", "Room Service", "En-suite Marble Bath"],
        "bed_label": "1 King Bed",
    }


class SpeechNormalizerProcessor(FrameProcessor):
    """Normalizes speech text for natural, phonetic TTS pronunciation.

    Expands currencies ($180 -> 'one hundred eighty dollars'), room codes,
    booking IDs, and abbreviations so the synthesized voice sounds human.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and frame.text:
            normalized = normalize_speech_text(frame.text)
            frame.text = normalized
        await self.push_frame(frame, direction)


class DataChannelBridgeProcessor(FrameProcessor):
    """Bridges pipeline audio and LLM frames to browser DataChannel events.

    Emits live speaker captions for both guest speech (transcriptions from STT)
    and agent voice (sentences emitted by the LLM).
    """

    def __init__(
        self,
        connection: SmallWebRTCConnection | Any,
        role: str = "both",
        on_activity: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._connection = connection
        self._role = role  # "guest", "agent", or "both"
        self.on_activity = on_activity

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if self.on_activity is not None and isinstance(
            frame, (TranscriptionFrame, InterimTranscriptionFrame, TextFrame)
        ):
            try:
                self.on_activity()
            except Exception:
                pass

        # 1. Guest speech captions from STT
        if self._role in ("guest", "both"):
            if isinstance(frame, TranscriptionFrame) and frame.text:
                self._send_safe(
                    {
                        "type": "caption",
                        "speaker": "guest",
                        "text": frame.text.strip(),
                        "is_final": True,
                    }
                )
            elif isinstance(frame, InterimTranscriptionFrame) and frame.text:
                self._send_safe(
                    {
                        "type": "caption",
                        "speaker": "guest",
                        "text": frame.text.strip(),
                        "is_final": False,
                    }
                )

        # 2. Agent voice captions from LLM / SentenceAggregator
        if self._role in ("agent", "both"):
            if isinstance(frame, TextFrame) and frame.text and frame.text.strip():
                self._send_safe(
                    {
                        "type": "caption",
                        "speaker": "agent",
                        "text": frame.text.strip(),
                        "is_final": True,
                    }
                )

        await self.push_frame(frame, direction)

    def _send_safe(self, message: dict[str, Any]) -> None:
        try:
            if hasattr(self._connection, "send_app_message"):
                self._connection.send_app_message(message)
        except Exception as exc:
            logger.debug("DataChannel message drop: %s", exc)


class ConfirmationGateContextProcessor(FrameProcessor):
    """Feeds the guest's final spoken utterance into the out-loud confirmation gate.

    ``backend.mcp`` write tools compare the caller's latest utterance against
    ``AgentSession.last_user_message`` before committing a staged write. Typed
    text sets that value in ``inject_user_text``; this processor provides the
    spoken equivalent so a guest can actually say "yes" out loud to confirm a
    reservation. Only final ``TranscriptionFrame`` entries are recorded so
    partial dictation never reaches the gate.
    """

    def __init__(self, session: AgentSession) -> None:
        super().__init__()
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text and frame.text.strip():
            self._session.set_last_user_message(frame.text)
        await self.push_frame(frame, direction)


def emit_generative_ui_events(
    connection: SmallWebRTCConnection | Any,
    tool_name: str,
    args: Mapping[str, Any],
    result: dict[str, Any],
) -> None:
    """Translate MCP tool outcomes into typed generative UI events for consumer_ui."""
    if not hasattr(connection, "send_app_message"):
        return

    try:
        if tool_name == "hotel_check_availability":
            rooms = result.get("rooms", [])
            date_in = str(args.get("date_in", ""))
            date_out = str(args.get("date_out", ""))

            # Emit room cards for available rooms
            for room in rooms:
                room_id = room.get("room_id", "")
                category = room.get("category", "")
                rate = float(room.get("nightly_rate", 0.0))
                presentation = _get_room_presentation(room_id, category)

                connection.send_app_message(
                    {
                        "type": "room_card",
                        "room": {
                            "id": room_id,
                            "name": presentation["name"],
                            "description": presentation["description"],
                            "price_per_night": rate,
                            "currency": "USD",
                            "amenities": presentation["amenities"],
                            "bed_label": presentation["bed_label"],
                            "availability_label": f"Available {date_in} to {date_out}" if date_in else "Available",
                        },
                    }
                )

            # Advance booking stepper to room selection
            connection.send_app_message(
                {
                    "type": "booking_stepper",
                    "booking": {
                        "current_step": "room",
                        "check_in": date_in or None,
                        "check_out": date_out or None,
                        "currency": "USD",
                    },
                }
            )

        elif tool_name == "hotel_book_room":
            room_id = str(args.get("room_id", ""))
            presentation = _get_room_presentation(room_id)
            date_in = str(args.get("date_in", ""))
            date_out = str(args.get("date_out", ""))
            guest_name = str(args.get("guest_name", ""))
            status_val = result.get("status")

            if status_val == "needs_confirmation":
                total = result.get("total_price")
                connection.send_app_message(
                    {
                        "type": "booking_stepper",
                        "booking": {
                            "current_step": "confirmation",
                            "check_in": date_in or None,
                            "check_out": date_out or None,
                            "room_name": presentation["name"],
                            "guest_count": 1,
                            "total": total,
                            "currency": "USD",
                        },
                    }
                )
                connection.send_app_message(
                    {
                        "type": "confirmation_required",
                        "subject": f"{presentation['name']} reservation",
                        "message": (
                            f"Room {room_id} from {date_in} to {date_out} for {guest_name}. "
                            "Please say yes or confirm aloud to finalize your stay."
                        ),
                    }
                )

            elif status_val == "confirmed":
                total = result.get("total_price")
                connection.send_app_message(
                    {
                        "type": "booking_stepper",
                        "booking": {
                            "current_step": "confirmation",
                            "check_in": date_in or None,
                            "check_out": date_out or None,
                            "room_name": presentation["name"],
                            "guest_count": 1,
                            "total": total,
                            "currency": "USD",
                        },
                    }
                )
                # Active producer for consumer_ui's checkout surface: review of
                # the confirmed total. The modal still refuses card collection
                # because payment handoff is not configured in this build.
                if isinstance(total, (int, float)) and not isinstance(total, bool):
                    connection.send_app_message(
                        {
                            "type": "checkout_prompt",
                            "amount": float(total),
                            "currency": "USD",
                            "description": (
                                f"{presentation['name']} stay from {date_in} to {date_out}"
                                if date_in
                                else f"{presentation['name']} reservation"
                            ),
                        }
                    )

    except Exception as exc:
        logger.warning("Failed to emit generative UI events for tool %s: %s", tool_name, exc)


def build_mcp_function_schemas(
    mcp_hub: MCPHub,
    persona_key: str,
    connection: SmallWebRTCConnection | Any,
) -> list[FunctionSchema]:
    """Convert MCP catalog entries to Pipecat FunctionSchema with generative UI dispatch."""
    catalog = mcp_hub.list_tools(persona_key)
    schemas: list[FunctionSchema] = []

    for item in catalog:
        tool_name = item["name"]
        description = item["description"]
        schema = item.get("inputSchema", {})
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Bind each handler to its tool name so closures do not collapse onto
        # the loop variable (every schema must dispatch its own tool).
        def _bind(name: str):
            async def _handler(params: FunctionCallParams) -> None:
                call_args = dict(params.arguments)
                result = await mcp_hub.call_tool(name, call_args, persona=persona_key)
                emit_generative_ui_events(connection, name, call_args, result)
                await params.result_callback(result)

            return _handler

        schemas.append(
            FunctionSchema(
                name=tool_name,
                description=description,
                properties=properties,
                required=required,
                handler=_bind(tool_name),
            )
        )

    return schemas


@dataclass
class ConsumerVoicePipeline:
    """Manages the lifecycle of a real-time multimodal voice session."""

    connection: SmallWebRTCConnection | Any
    transport: SmallWebRTCTransport | Any
    pipeline: Pipeline
    worker: PipelineWorker
    runner: WorkerRunner
    context: LLMContext
    mcp_hub: MCPHub
    runner_task: asyncio.Task[None] | None = None
    idle_timeout_seconds: float = 300.0
    last_activity_time: float = field(default_factory=time.monotonic)
    watchdog_task: asyncio.Task[None] | None = None
    _closed: bool = False

    def touch_activity(self) -> None:
        """Reset the inactivity watchdog timer when audio or interaction occurs."""
        self.last_activity_time = time.monotonic()

    async def start(self) -> None:
        """Launch the worker runner and watchdog background tasks."""
        if self.runner_task is not None and not self.runner_task.done():
            return
        self._closed = False
        self.touch_activity()
        self.runner_task = asyncio.create_task(self.runner.run())
        if self.watchdog_task is None or self.watchdog_task.done():
            self.watchdog_task = asyncio.create_task(self._inactivity_watchdog())

    async def stop(self) -> None:
        """Cancel and clean up the pipeline worker runner and watchdog."""
        if self._closed:
            return
        self._closed = True

        watchdog_task = self.watchdog_task
        self.watchdog_task = None
        if watchdog_task is not None and watchdog_task is not asyncio.current_task():
            if not watchdog_task.done():
                watchdog_task.cancel()
                try:
                    await watchdog_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Inactivity watchdog failed during pipeline shutdown")

        try:
            await self.runner.cancel()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Failed to cancel the voice pipeline worker runner")

        if self.runner_task is not None:
            if not self.runner_task.done():
                self.runner_task.cancel()
                try:
                    await self.runner_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Voice pipeline worker runner failed during shutdown")
            self.runner_task = None

    async def _close_connection(self) -> None:
        """Close the peer connection and log failures instead of hiding them."""
        close = getattr(self.connection, "close", None)
        if close is None:
            return
        try:
            close_result = close()
            if asyncio.iscoroutine(close_result):
                await close_result
        except Exception:
            logger.exception("Failed to close voice connection")

    async def _inactivity_watchdog(self) -> None:
        """Reap zombie or abandoned pipelines after idle timeout expires."""
        try:
            while not self._closed:
                interval = min(15.0, max(0.05, self.idle_timeout_seconds / 4))
                await asyncio.sleep(interval)
                if self._closed:
                    break
                idle_duration = time.monotonic() - self.last_activity_time
                if idle_duration >= self.idle_timeout_seconds:
                    logger.warning(
                        "Voice session reached idle timeout (%.1fs >= %.1fs). Reaping pipeline.",
                        idle_duration,
                        self.idle_timeout_seconds,
                    )
                    if hasattr(self.connection, "send_app_message"):
                        try:
                            self.connection.send_app_message(
                                {
                                    "type": "ui_error",
                                    "message": "Session closed due to inactivity.",
                                }
                            )
                        except Exception:
                            logger.warning("Failed to notify client about voice session timeout")
                    await self.stop()
                    await self._close_connection()
                    break
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Inactivity watchdog failed")
            try:
                await self.stop()
            except Exception:
                logger.exception("Failed to stop voice pipeline after watchdog failure")
            await self._close_connection()

    async def inject_user_text(self, text: str) -> None:
        """Inject typed text from DualInputBar directly into LLM turn context."""
        cleaned = sanitize_pii_text(text.strip())
        if not cleaned:
            return
        self.touch_activity()
        self.context.add_message({"role": "user", "content": cleaned})
        self.mcp_hub.session.set_last_user_message(cleaned)
        await self.worker.queue_frames([LLMRunFrame()])

    async def inject_touch_override(self, room_id: str, label: str) -> None:
        """Inject client touch selection as non-committing conversational context."""
        self.touch_activity()
        cleaned_room_id = sanitize_pii_text(room_id.strip())
        cleaned_label = sanitize_pii_text(label.strip())
        hint = (
            f"[Guest tapped '{cleaned_label}' (Room ID: {cleaned_room_id}) on screen. "
            "Acknowledge that you see their selection, describe its highlights, "
            "and remind them that you need their verbal confirmation aloud to proceed with booking.]"
        )
        self.context.add_message({"role": "developer", "content": hint})
        await self.worker.queue_frames([LLMRunFrame()])


async def create_consumer_voice_pipeline(
    connection: SmallWebRTCConnection | Any,
    tenant: TenantConfig,
    persona_key: str,
    settings: AppSettings | None = None,
) -> ConsumerVoicePipeline:
    """Assemble and configure the full real-time Pipecat multimodal voice pipeline."""
    app_settings = settings or get_settings()
    persona = tenant.personas.get(persona_key) or tenant.get_active_persona()

    # 1. MCP Hub and Tool Schemas
    agent_session = AgentSession(
        session_id=getattr(connection, "pc_id", "session-local"),
        tenant_id=tenant.tenant_id,
        persona=persona.key,
    )
    mcp_hub = create_mcp_hub(session=agent_session)
    function_schemas = build_mcp_function_schemas(mcp_hub, persona.key, connection)
    tools_schema = ToolsSchema(standard_tools=function_schemas)

    # 2. LLM Context seeded with persona system prompt and MCP tools
    context = LLMContext(
        messages=[{"role": "system", "content": persona.system_prompt}],
        tools=tools_schema,
    )

    # 3. Context Aggregator Pair with Silero VAD
    vad_analyzer = SileroVADAnalyzer()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad_analyzer),
    )

    # 4. Service Factories
    stt_service = create_stt_service(settings=app_settings)
    llm_service = create_llm_service(settings=app_settings, override=persona.llm)
    tts_service = create_tts_service(settings=app_settings, override=persona.voice)

    # 5. Pipeline Frame Processors
    pii_sanitizer = PIISanitizerProcessor()
    confirmation_context = ConfirmationGateContextProcessor(session=agent_session)
    guest_caption_bridge = DataChannelBridgeProcessor(connection=connection, role="guest")
    sentence_aggregator = SentenceAggregator()
    agent_caption_bridge = DataChannelBridgeProcessor(connection=connection, role="agent")
    speech_normalizer = SpeechNormalizerProcessor()

    # 6. Transport
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    # 7. Assembled Pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt_service,
            pii_sanitizer,
            confirmation_context,
            guest_caption_bridge,
            user_aggregator,
            llm_service,
            sentence_aggregator,
            agent_caption_bridge,
            speech_normalizer,
            tts_service,
            transport.output(),
            assistant_aggregator,
        ]
    )

    # Idle cancellation in worker is false; the pipeline watchdog task manages reclamation
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        cancel_on_idle_timeout=False,
        cancel_runner_on_idle_timeout=False,
    )

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    # 8. Proactive greeting: the agent opens the conversation as soon as the
    # peer connection is established, which also proves the LLM -> TTS ->
    # transport audio loop is live for every session.
    greeting_state = {"sent": False}

    @transport.event_handler("on_client_connected")
    async def on_client_connected(
        active_transport: SmallWebRTCTransport,
        active_connection: Any,
    ) -> None:
        if greeting_state["sent"]:
            return
        greeting_state["sent"] = True
        context.add_message(
            {
                "role": "developer",
                "content": (
                    f"Greet the caller in one short sentence as {persona.name}, the "
                    f"{tenant.tenant_name} concierge, and ask how you can help. "
                    "One sentence only, no lists, no emojis. "
                    "Speak in short sentences separated by clear punctuation."
                ),
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    pipeline_instance = ConsumerVoicePipeline(
        connection=connection,
        transport=transport,
        pipeline=pipeline,
        worker=worker,
        runner=runner,
        context=context,
        mcp_hub=mcp_hub,
        idle_timeout_seconds=float(app_settings.voice_session_idle_timeout_seconds),
    )
    guest_caption_bridge.on_activity = pipeline_instance.touch_activity
    agent_caption_bridge.on_activity = pipeline_instance.touch_activity
    return pipeline_instance
