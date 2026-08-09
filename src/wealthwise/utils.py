"""Shared utility helpers — keep minimal (YAGNI)."""
from __future__ import annotations


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Return x bounded to the closed interval [lo, hi]."""
    return max(lo, min(hi, x))
