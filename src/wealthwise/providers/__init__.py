"""Data-provider layer public API.

Import the Protocols and the factory here so callers only need:
    from wealthwise.providers import MarketProvider, build_provider
"""
from wealthwise.providers.base import FXProvider, MacroProvider, MarketProvider
from wealthwise.providers.consensus import ConsensusResolver, Reading
from wealthwise.providers.registry import SourceRegistry
from wealthwise.providers.sample import (
    SampleFXProvider,
    SampleMacroProvider,
    SampleMarketProvider,
)

__all__ = [
    # Protocols
    "MarketProvider",
    "MacroProvider",
    "FXProvider",
    # Sample providers
    "SampleMarketProvider",
    "SampleMacroProvider",
    "SampleFXProvider",
    # Consensus
    "ConsensusResolver",
    "Reading",
    # Registry
    "SourceRegistry",
]
