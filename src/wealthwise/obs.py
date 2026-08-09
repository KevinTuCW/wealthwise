from functools import wraps
import os

from wealthwise.config import get_settings


def _langfuse_settings() -> tuple[bool, str, str, str]:
    s = get_settings()
    return (s.enable_langfuse_tracing,
            s.langfuse_public_key.strip(),
            s.langfuse_secret_key.strip(),
            s.langfuse_base_url.strip())


def tracing_enabled() -> bool:
    enabled, public_key, secret_key, _ = _langfuse_settings()
    return bool(enabled and public_key and secret_key)


def configure_langfuse_env() -> None:
    _, public_key, secret_key, base_url = _langfuse_settings()
    if public_key and secret_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
    if base_url:
        os.environ.setdefault("LANGFUSE_HOST", base_url)


def traced(name: str):
    """Decorator: record fn as a Langfuse observation when configured, else run it directly.

    The check is per-call so tests can disable tracing regardless of a local .env.
    """
    def decorate(fn):
        @wraps(fn)
        def run(*args, **kwargs):
            if not tracing_enabled():
                return fn(*args, **kwargs)
            configure_langfuse_env()
            from langfuse import observe            # langfuse v4
            return observe(name=name)(fn)(*args, **kwargs)
        return run
    return decorate
