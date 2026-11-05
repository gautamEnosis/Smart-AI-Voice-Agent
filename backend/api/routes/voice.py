"""Browser WebRTC signalling endpoints for the consumer multimodal portal."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from backend.api.schemas import (
    ConsumerConfigurationResponse,
    VoiceIcePatchRequest,
    VoiceOfferRequest,
    VoiceOfferResponse,
)
from backend.api.voice_sessions import get_voice_session_manager

router = APIRouter(prefix="/api/v1/voice", tags=["Consumer Voice"])


@router.get("/config", response_model=ConsumerConfigurationResponse)
def get_consumer_configuration(tenant_id: str = "grand-meridian") -> ConsumerConfigurationResponse:
    """Return browser-safe tenant and concierge metadata for the consumer portal."""
    return get_voice_session_manager().configuration(tenant_id)


@router.post("/offer", response_model=VoiceOfferResponse)
async def create_voice_offer(payload: VoiceOfferRequest, request: Request) -> VoiceOfferResponse:
    """Negotiate a local browser-to-server SmallWebRTC peer connection."""
    # Do not trust caller-controlled forwarding headers for abuse controls.
    client_ip = request.client.host if request.client else "unknown"

    return await get_voice_session_manager().create_offer(payload, client_ip=client_ip)


@router.patch("/offer", status_code=status.HTTP_204_NO_CONTENT)
async def add_voice_ice_candidates(payload: VoiceIcePatchRequest) -> Response:
    """Accept browser trickle ICE candidates for an already-negotiated peer connection."""
    await get_voice_session_manager().add_ice_candidates(payload)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
