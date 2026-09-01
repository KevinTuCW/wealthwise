"""AdvisoryDeps — dependency-injection container for the advisory pipeline.

Pure data container: no logic, no side effects.  Mirror of shopscout.funnel.deps.FunnelDeps.
All expert nodes receive an AdvisoryDeps instance so they can be tested in isolation
by swapping providers / jury clients without touching any business logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from wealthwise.llm import ModelClient
from wealthwise.providers.base import (
    FXProvider,
    HistoryProvider,
    MacroProvider,
    MarketProvider,
)
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

    # --- optional daily price history (momentum / realized volatility) ---
    # Only consulted when factor scoring is on. Optional because the offline
    # stack has no history source: sample candidates already carry a
    # `volatility` metric, so three of the five factors still score without it.
    history: HistoryProvider | None = None

    # --- domain threshold params ---
    max_fx_exposure: float = 0.5          # max fraction of portfolio in non-CNY
    risk_budget_method: str = "risk_parity"  # "risk_parity" | "equal_weight"
    max_llm_judgments: int = 12           # budget guardrail: max jury calls per run
    # Rank equity candidates by the multi-factor composite instead of the
    # size-then-valuation rule. Mirrors settings.enable_factor_scoring; carried
    # on deps so a test can flip it without touching process-wide config.
    enable_factor_scoring: bool = False
    # Exclude candidates whose price two feeds disagree on from selection.
    drop_on_data_disagreement: bool = True
    # Fraction of *PASS* verdicts the jury also reviews (0..1, deterministic per
    # profile). Cross-validation used to fire only where the rules already said
    # DOWNGRADE/REJECT — it double-checked true positives only, while the
    # dangerous direction (a mis-rated asset sailing through as PASS) had no
    # second pair of eyes. The jury can still only tighten a verdict, never
    # soften one, so this cannot weaken the hard gate.
    jury_review_pass_rate: float = 1.0
