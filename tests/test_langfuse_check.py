"""Tests for wealthwise.langfuse_check — offline-safe behaviour."""
from __future__ import annotations

import pytest

from wealthwise.langfuse_check import main


def test_main_exits_0_offline_no_keys():
    """main() must exit 0 when no Langfuse keys are configured (offline default)."""
    # No keys in the test environment; tracing_enabled will be False → skip path
    result = main([])
    assert result == 0


def test_main_exits_0_offline_with_name_arg():
    """main() with a custom --name arg must still exit 0 offline."""
    result = main(["--name", "wealthwise.test_smoke"])
    assert result == 0
