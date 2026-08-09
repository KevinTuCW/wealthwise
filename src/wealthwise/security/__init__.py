"""wealthwise.security — deterministic input sanitization and PII/secret redaction.

Two responsibilities:
1. User-input facing: normalize + detect_injection to block prompt-injection
   attacks in investor goals, queries, and other user-controlled strings.
2. LLM-data facing: neutralize_untrusted wraps external text (market data,
   news, analyst commentary) so it cannot hijack the advisory pipeline.
3. Output facing: redact scrubs PII and secret tokens before surfacing to
   operators or logs.
"""
from wealthwise.security.sanitize import detect_injection, neutralize_untrusted, normalize
from wealthwise.security.redact import redact

__all__ = ["detect_injection", "neutralize_untrusted", "normalize", "redact"]
