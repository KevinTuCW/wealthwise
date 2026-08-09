"""AdvisoryDeps — dependency-injection container for the advisory pipeline.

Pure data container: no logic, no side effects.  Mirror of shopscout.funnel.deps.FunnelDeps.
All expert nodes receive an AdvisoryDeps instance so they can be tested in isolation
by swapping providers / jury clients without touching any business logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from wealthwise.llm import ModelClient
from wealthwise.providers.base import FXProvider, MacroProvider, MarketProvider
from wealthwise.rag.embed import Embedder
from wealthwise.rag.store import Retriever


@dataclass(frozen=True)
class AdvisoryDeps:
    """Everything the expert-agent nodes need, injected so each stays testable.

    Swap any field for a test double without modifying the node functions.
    """

    # --- data providers ---
    market: MarketProvider
    macro: MacroProvider
    fx: FXProvider

    # --- LLM jury ---
    jury_clients: list[ModelClient]

    # --- RAG retrievers ---
    policy_retriever: Retriever
    research_retriever: Retriever

    # --- optional embedder (for future per-run RAG indexing) ---
    embedder: Embedder | None = None

    # --- domain threshold params ---
    max_fx_exposure: float = 0.5          # max fraction of portfolio in non-CNY
    risk_budget_method: str = "risk_parity"  # "risk_parity" | "equal_weight"
    max_llm_judgments: int = 12           # budget guardrail: max jury calls per run
