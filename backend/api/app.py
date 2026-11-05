"""FastAPI application factory for the OmniVox gateway."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.admin import router as admin_router
from backend.api.routes.voice import router as voice_router
from backend.api.voice_sessions import get_voice_session_manager
from backend.config.settings import get_settings


def create_app() -> FastAPI:
    """Create the local-first HTTP gateway and its browser-safe CORS policy."""
    settings = get_settings()
    origins = list(
        dict.fromkeys(
            origin.strip()
            for configured_origins in (settings.admin_cors_origins, settings.consumer_cors_origins)
            for origin in configured_origins.split(",")
            if origin.strip()
        )
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        """Release live WebRTC peer connections cleanly on gateway shutdown."""
        yield
        await get_voice_session_manager().close()

    application = FastAPI(
        title="OmniVox Gateway",
        version="0.6.0",
        description="Tenant-safe operational and consumer voice controls for OmniVox.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH"],
        allow_headers=["Content-Type", "Authorization"],
    )
    application.include_router(admin_router)
    application.include_router(voice_router)

    @application.get("/health", tags=["System"])
    def health_check() -> dict[str, str]:
        """Return a lightweight local readiness probe."""
        return {"status": "ok", "service": "omnivox-gateway"}

    return application


app = create_app()
