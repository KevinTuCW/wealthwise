"""Data-provider layer public API.

Import the Protocols and the shared building blocks here so callers only need:
    from wealthwise.providers import MarketProvider, ConsensusMarketProvider

Live providers that pull network dependencies (Tencent, Sina, AkShare, k-line
history) are deliberately absent: importing this package must stay free of
`requests` and `akshare`, so those are imported from their own modules at the
point of use.
"""
from wealthwise.providers.base import (
    FXProvider,
    HistoryProvider,
    MacroProvider,
    MarketProvider,
)
from wealthwise.providers.consensus import ConsensusResolver, Reading
from wealthwise.providers.consensus_provider import (
    ConsensusMacroProvider,
    ConsensusMarketProvider,
    QualitativeMacroSource,
)
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
    "HistoryProvider",
    # Sample providers
    "SampleMarketProvider",
    "SampleMacroProvider",
    "SampleFXProvider",
    # Consensus
    "ConsensusResolver",
    "Reading",
    "ConsensusMarketProvider",
    "ConsensusMacroProvider",
    "QualitativeMacroSource",
    # Registry
    "SourceRegistry",
]
