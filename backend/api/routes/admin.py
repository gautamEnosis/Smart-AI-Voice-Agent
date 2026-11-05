"""Administrative endpoints for the enterprise management console."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from backend.api.audition import VoiceAuditionUnavailable, synthesize_audition
from backend.api.auth import verify_admin_key
from backend.api.knowledge import MAX_DOCUMENT_BYTES, KnowledgeHubService
from backend.api.runtime import get_admin_telemetry, get_provider_runtime_store
from backend.api.schemas import (
    DashboardResponse,
    KnowledgeDocumentResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    ProviderSettingsResponse,
    ProviderSettingsUpdate,
    ServiceHealth,
    VoiceAuditionRequest,
    VoiceProfileResponse,
)
from backend.config.settings import get_settings
from backend.engine.audio.voice_catalog import list_available_voices
from backend.engine.providers.base import ProviderConfigurationError

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["Enterprise Admin"],
    dependencies=[Depends(verify_admin_key)],
)


@lru_cache(maxsize=1)
def get_knowledge_hub() -> KnowledgeHubService:
    """Return the shared local Knowledge Hub service."""
    return KnowledgeHubService(get_settings())


def _bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard() -> DashboardResponse:
    """Return the latest operational snapshot without fabricated call metrics."""
    provider_settings = get_provider_runtime_store().response()
    active_ready = next(
        option.configured
        for option in provider_settings.providers
        if option.provider == provider_settings.active.provider
    )
    knowledge_hub = get_knowledge_hub()
    return DashboardResponse(
        active_provider=provider_settings.active,
        latest_latency=get_admin_telemetry().latest_turn(),
        services=[
            ServiceHealth(
                name="Language model",
                status="ready" if active_ready else "attention",
                detail=(
                    f"{provider_settings.active.provider} is selected"
                    if active_ready
                    else f"{provider_settings.active.provider} needs a server-side API key"
                ),
            ),
            ServiceHealth(
                name="Knowledge retrieval",
                status="ready",
                detail="Tenant-isolated local retrieval is available",
            ),
            ServiceHealth(
                name="Confirmation gate",
                status="ready",
                detail="Booking and other writes remain confirmation-gated",
            ),
        ],
        tool_calls_last_hour=get_admin_telemetry().tool_calls_last_hour(),
        knowledge_documents=knowledge_hub.document_count(),
    )


@router.get("/providers", response_model=ProviderSettingsResponse)
def get_provider_settings() -> ProviderSettingsResponse:
    """Return safe provider configuration and readiness state."""
    return get_provider_runtime_store().response()


@router.put("/providers", response_model=ProviderSettingsResponse)
def update_provider_settings(payload: ProviderSettingsUpdate) -> ProviderSettingsResponse:
    """Hot-swap the runtime provider choice after validating its server-side key."""
    try:
        return get_provider_runtime_store().update(payload)
    except ProviderConfigurationError as exc:
        raise _bad_request(str(exc)) from exc


@router.get("/voices", response_model=list[VoiceProfileResponse])
def get_voice_catalog() -> list[VoiceProfileResponse]:
    """Expose voice metadata for browser-side selection without provider credentials."""
    return [
        VoiceProfileResponse(
            voice_id=profile.voice_id,
            name=profile.name,
            provider=profile.provider,
            gender=profile.gender,
            accent=profile.accent,
            recommended_speed=profile.recommended_speed,
            description=profile.description,
            role_fit=list(profile.role_fit),
        )
        for profile in sorted(list_available_voices(), key=lambda item: (item.provider, item.name))
    ]


@router.post("/voice/audition")
def audition_voice(payload: VoiceAuditionRequest) -> Response:
    """Return a real WAV audition or an actionable setup error."""
    try:
        audio = synthesize_audition(payload, get_settings())
    except VoiceAuditionUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/knowledge/documents", response_model=list[KnowledgeDocumentResponse])
def list_knowledge_documents(tenant_id: str = "grand-meridian") -> list[KnowledgeDocumentResponse]:
    """List successfully indexed documents for a tenant."""
    try:
        return get_knowledge_hub().list_documents(tenant_id)
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc


@router.post("/knowledge/upload", response_model=KnowledgeDocumentResponse)
async def upload_knowledge_document(
    document: UploadFile = File(...),
    tenant_id: str = Form("grand-meridian"),
) -> KnowledgeDocumentResponse:
    """Upload and stage-index a tenant document, bounded to a safe local size."""
    content = await document.read(MAX_DOCUMENT_BYTES + 1)
    if len(content) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Documents must be 10 MB or smaller.")
    try:
        return await run_in_threadpool(
            get_knowledge_hub().save_and_index,
            tenant_id,
            document.filename or "document.txt",
            content,
        )
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc


@router.post("/knowledge/search", response_model=KnowledgeSearchResponse)
def search_knowledge(payload: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    """Test tenant-scoped retrieval using the same spoken-brevity guardrails as calls."""
    try:
        return get_knowledge_hub().search(payload.tenant_id, payload.query)
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
