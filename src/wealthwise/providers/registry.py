"""SourceRegistry — aggregate multiple signal-reading providers.

Mirrors shopscout's registry pattern. Providers registered here must
expose a `signal_readings(signal: str) -> list[Reading]` method.
The registry gathers all readings for a signal and hands them to
ConsensusResolver, returning a single ConsensusResult.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from wealthwise.providers.consensus import ConsensusResolver, ConsensusResult, Reading


@runtime_checkable
class SignalProvider(Protocol):
    """Minimal interface for providers that can be registered in SourceRegistry."""

    name: str

    def signal_readings(self, signal: str) -> list[Reading]:
        ...


class SourceRegistry:
    """Aggregate multiple SignalProviders, resolve signals via ConsensusResolver.

    Usage::

        registry = SourceRegistry(resolver=ConsensusResolver(threshold=0.2))
        registry.register(provider_a)
        registry.register(provider_b)
        result = registry.resolve("volatility")
    """

    def __init__(self, resolver: ConsensusResolver) -> None:
        self._resolver = resolver
        self._providers: list[SignalProvider] = []

    def register(self, provider: SignalProvider) -> None:
        self._providers.append(provider)

    def resolve(self, signal: str) -> ConsensusResult:
        """Gather readings for `signal` from all providers and reconcile."""
        readings: list[Reading] = []
        for provider in self._providers:
            readings.extend(provider.signal_readings(signal))
        return self._resolver.resolve(readings)
