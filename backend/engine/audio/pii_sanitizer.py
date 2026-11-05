"""Real-time PII and PCI-DSS sanitization for audio transcriptions and text frames.

Redacts credit card primary account numbers (PAN), CVVs, and government identifiers
before speech data flows into LLM context, DataChannel captions, or telemetry.
"""

from __future__ import annotations

import re
from typing import Any

try:
    from pipecat.frames.frames import (
        Frame,
        InterimTranscriptionFrame,
        TextFrame,
        TranscriptionFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
except ImportError:  # pragma: no cover - optional runtime dependency
    Frame = Any  # type: ignore[misc, assignment]
    TranscriptionFrame = type("TranscriptionFrame", (), {})  # type: ignore[misc, assignment]
    InterimTranscriptionFrame = type("InterimTranscriptionFrame", (), {})  # type: ignore[misc, assignment]
    TextFrame = type("TextFrame", (), {})  # type: ignore[misc, assignment]

    class FrameDirection:  # type: ignore[no-redef]
        DOWNSTREAM = "downstream"
        UPSTREAM = "upstream"

    class FrameProcessor:  # type: ignore[no-redef]
        async def process_frame(self, frame: Any, direction: Any = None) -> None:
            return None

        async def push_frame(self, frame: Any, direction: Any = None) -> None:
            return None


# Match all PAN-length digit candidates, including values that fail Luhn after STT errors.
CARD_CANDIDATE_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# Match CVV / CVC mentions with optional 'is' or 'of'
CVV_REGEX = re.compile(
    r"(?i)\b(cvv|cvc|cid|security code|card code)(?:\s+(?:is|of))?[\s:]*([0-9]{3,4})\b"
)

# Match US SSN (123-45-6789)
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EIN_REGEX = re.compile(r"\b\d{2}-\d{7}\b")


def is_luhn_valid(number_str: str) -> bool:
    """Verify Luhn checksum for a numeric sequence."""
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, digit in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


def sanitize_pii_text(text: str) -> str:
    """Sanitize sensitive financial and identity data from a text string.

    Redacts:
    - 13-19 digit credit-card-length candidates, including STT-corrupted numbers
    - CVV codes associated with security labels
    - US SSN and EIN patterns
    """
    if not text:
        return text

    # 1. Redact CVV with label
    cleaned = CVV_REGEX.sub(r"\1 [CVV REDACTED]", text)

    # 2. Redact SSN
    cleaned = SSN_REGEX.sub("[SSN REDACTED]", cleaned)

    # 3. Redact employer identification numbers
    cleaned = EIN_REGEX.sub("[TAX ID REDACTED]", cleaned)

    # 4. Redact Credit Card Numbers
    def _replace_card(match: re.Match[str]) -> str:
        digits_only = re.sub(r"\D", "", match.group(0))
        return "[CARD REDACTED]" if 13 <= len(digits_only) <= 19 else match.group(0)

    cleaned = CARD_CANDIDATE_REGEX.sub(_replace_card, cleaned)
    return cleaned


class PIISanitizerProcessor(FrameProcessor):
    """Pipeline processor that sanitizes PII from live transcriptions and text frames."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame, TextFrame)):
            if getattr(frame, "text", None):
                frame.text = sanitize_pii_text(frame.text)
        await self.push_frame(frame, direction)
