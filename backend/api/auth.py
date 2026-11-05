"""Authentication and access control dependencies for the administrative surface."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.config.settings import AppSettings, get_settings

# Bearer auth scheme (auto_error=False allows returning custom 401 payloads)
_bearer_scheme = HTTPBearer(auto_error=False)


def verify_admin_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer_scheme)],
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> str:
    """Verify the administrator API key using constant-time string comparison.

    Accepts Authorization: Bearer <admin_api_key>.
    Returns the validated token or raises 401 Unauthorized.
    """
    expected_key = settings.admin_api_key.strip() if settings.admin_api_key else ""

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed administrative credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided_key = credentials.credentials.strip()

    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Administrative access is unavailable because no API key is configured.",
        )

    # Constant-time comparison prevents timing side-channel attacks
    if not secrets.compare_digest(provided_key.encode("utf-8"), expected_key.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid administrative API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return provided_key
