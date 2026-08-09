"""Langfuse connectivity check for WealthWise.

If tracing is disabled or keys are missing, prints a clear message and exits 0.
If keys are present, attempts a minimal Langfuse client init + flush and reports success.

Must be import-safe offline and exit 0 with no keys configured.
"""
from __future__ import annotations

import argparse
import sys
import time

from wealthwise.config import get_settings
from wealthwise.obs import configure_langfuse_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Send a minimal WealthWise Langfuse smoke trace."
    )
    parser.add_argument("--name", default="wealthwise.langfuse_smoke")
    args = parser.parse_args(argv)

    settings = get_settings()

    if not settings.tracing_enabled:
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            print(
                "tracing disabled / no keys — skipping "
                "(set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and "
                "ENABLE_LANGFUSE_TRACING=true to enable)",
                file=sys.stderr,
            )
        else:
            print(
                "tracing disabled — skipping "
                "(set ENABLE_LANGFUSE_TRACING=true to enable)",
                file=sys.stderr,
            )
        return 0

    configure_langfuse_env()
    from langfuse import get_client, observe  # langfuse v4

    @observe(name=args.name)
    def _smoke() -> dict:
        return {"service": "wealthwise", "ts": time.time()}

    payload = _smoke()
    client = get_client()
    client.flush()
    print(f"sent {args.name}: {payload['service']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
