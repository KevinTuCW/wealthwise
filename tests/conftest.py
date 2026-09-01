import pytest

from wealthwise import config, obs


@pytest.fixture(autouse=True)
def _disable_tracing(monkeypatch):
    monkeypatch.setattr(obs, "tracing_enabled", lambda: False)


@pytest.fixture(autouse=True)
def _isolate_developer_env(monkeypatch):
    """Keep the suite hermetic even when a real `.env` sits in the repo root.

    `Settings` reads `.env`, and a working `.env` carries
    `USE_REAL_PROVIDERS=true` plus live keys — so `pytest` on a configured
    machine quietly left the offline stack and started calling GLM /
    SiliconFlow / AkShare for real: slow, flaky, billable, and nothing like the
    "fully offline" run the README advertises.

    `ENABLE_FACTOR_SCORING` is pinned for the same reason, one step removed: it
    changes *which* names selection returns, so a developer who has flipped it on
    would see the ranking assertions fail against a pipeline that is working
    correctly. Tests that want the factor path ask for it explicitly on `deps`.

    Environment variables outrank the dotenv source, so forcing these switches
    here is enough — but `get_settings` is `lru_cache`d and is warmed at import
    time, before any fixture runs, so the cache has to be dropped on both sides.
    """
    monkeypatch.setenv("USE_REAL_PROVIDERS", "false")
    monkeypatch.setenv("ENABLE_LANGFUSE_TRACING", "false")
    monkeypatch.setenv("ENABLE_FACTOR_SCORING", "false")
    config.get_settings.cache_clear()
    try:
        yield
    finally:
        config.get_settings.cache_clear()
