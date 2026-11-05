"""Out-loud confirmation gate for destructive and commercial tool writes."""

from __future__ import annotations

import re
from typing import Any

CONFIRMATION_PHRASES = frozenset(
    {
        "yes",
        "yes please",
        "yes confirm the booking",
        "yeah",
        "yeah please",
        "yep",
        "yep please",
        "yup",
        "sure",
        "sure please",
        "correct",
        "confirmed",
        "confirm",
        "okay",
        "ok",
        "go ahead",
        "do it",
        "book it",
        "that's right",
        "that s right",
        "sounds good",
        "absolutely",
    }
)


def user_confirms(last_user_message: str | None) -> bool:
    """Return True only for a short, explicit, unambiguous spoken confirmation.

    Only a complete, explicitly affirmative phrase is accepted. Additional
    qualifiers or conversational content make the utterance ambiguous.
    """
    if not last_user_message:
        return False
    text = last_user_message.strip()
    if not text:
        return False

    normalized = re.sub(r"[^\w']+", " ", text.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if "?" in text:
        return False
    return normalized in CONFIRMATION_PHRASES


def confirmation_granted(
    pending: dict[str, Any] | None,
    kind: str,
    payload: dict[str, Any],
    last_user_message: str | None,
) -> bool:
    """Allow a write only when pending args match and the caller confirmed.

    The first invocation always stages ``pending``. A bare "yes" with no
    matching pending action never commits a booking, order, or cancel.
    """
    if not pending:
        return False
    if pending.get("kind") != kind:
        return False
    if pending.get("payload") != payload:
        return False
    return user_confirms(last_user_message)
