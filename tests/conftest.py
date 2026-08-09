import pytest

from wealthwise import obs


@pytest.fixture(autouse=True)
def _disable_tracing(monkeypatch):
    monkeypatch.setattr(obs, "tracing_enabled", lambda: False)
