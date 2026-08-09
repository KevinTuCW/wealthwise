"""Deterministic PII / secret redaction for anything surfaced to operators or logs.

Advisory explanations and notes can echo investor emails, phone numbers, or
leaked API keys. We scrub them before they leave the pipeline.
"""
import re

_SECRET = re.compile(
    r"\b(sk-[A-Za-z0-9]{8,}|pk-[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{12,}|"
    r"ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CARD = re.compile(r"\b\d{13,19}\b")          # card-like long digit runs
_PHONE = re.compile(r"(?<!\d)\+?\d[\d\s().-]{7,}\d(?!\d)")


def redact(text: str) -> str:
    """Mask emails, phones, card-like numbers, and known secret token shapes."""
    if not text:
        return text
    t = _SECRET.sub("[redacted-secret]", text)
    t = _EMAIL.sub("[redacted-email]", t)
    t = _CARD.sub("[redacted-number]", t)
    t = _PHONE.sub("[redacted-phone]", t)
    return t
