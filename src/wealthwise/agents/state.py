"""AdvisoryState data contract — core shared state passed between all agents.

All fields use pydantic v2 BaseModel. Mutable defaults (lists/dicts) use
default_factory so each model instance gets its own collection.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InvestorProfile(BaseModel):
    """Investor suitability profile (China A-share classification C1–C5)."""

    risk_level: Literal["C1", "C2", "C3", "C4", "C5"]
    investable: float                          # investable assets in CNY
    horizon_years: int                         # investment horizon (years)
    goals: list[str]                           # e.g. ["retirement", "education"]
    liquidity_min: float                       # minimum liquidity ratio (0..1)
    accept_cross_border: bool                  # accepts HK/US cross-border exposure
    holdings: list[str] = Field(default_factory=list)  # current symbol holdings


class AssetCandidate(BaseModel):
    """A single instrument under evaluation by an expert agent."""

    symbol: str
    market: Literal["A", "HK", "US"]
    asset_class: Literal["equity", "bond", "cash", "alt"]
    name: str
    currency: str
    r_level: Literal["R1", "R2", "R3", "R4", "R5"]
    metrics: dict = Field(default_factory=dict)   # signal bag (PE, duration, vol…)
    tags: list[str] = Field(default_factory=list)


class PortfolioAllocation(BaseModel):
    """Proposed allocation output from the portfolio optimizer."""

    weights: dict[str, float]          # symbol → weight; must sum to ~1
    class_weights: dict[str, float]    # asset_class → aggregate weight
    portfolio_r_level: str             # overall risk level label (e.g. "R3")
    fx_exposure: float                 # fraction of portfolio in non-CNY assets
    metrics: dict = Field(default_factory=dict)  # Sharpe, vol, max-DD, etc.


class ComplianceVerdict(BaseModel):
    """Output from the compliance guardrail agent."""

    decision: Literal["PASS", "DOWNGRADE", "REJECT"]
    matched: bool                               # investor profile matched portfolio risk
    violations: list[str] = Field(default_factory=list)
    disclosures: list[str] = Field(default_factory=list)
    confidence: float                           # 0..1


class AdvisoryState(BaseModel):
    """Shared state envelope threaded through the entire advisory pipeline.

    Supervisor and expert agents read from / write into this object.
    All list/dict fields use default_factory to avoid mutable-default aliasing.
    """

    # --- inputs ---
    profile: InvestorProfile | None = None
    goal_constraints: dict = Field(default_factory=dict)   # parsed goal/constraint bag
    macro_view: dict = Field(default_factory=dict)         # macro-context from RAG

    # --- intermediate expert outputs ---
    equity_candidates: list[AssetCandidate] = Field(default_factory=list)
    fixedincome_candidates: list[AssetCandidate] = Field(default_factory=list)

    # --- final outputs ---
    portfolio: PortfolioAllocation | None = None
    # Whole-lot order list derived from `portfolio`, plus entry/rebalance/channel
    # guidance. Weights alone are not executable — see portfolio/execution.py.
    execution_plan: dict = Field(default_factory=dict)
    compliance: ComplianceVerdict | None = None
    explanation: str = ""                  # natural-language advisory explanation
    confidence: float = 0.0               # overall pipeline confidence (0..1)

    # --- pipeline bookkeeping ---
    status: str = "pending"               # pending / running / done / failed
    notes: list[str] = Field(default_factory=list)          # agent-authored audit notes
    trace_events: list[dict] = Field(default_factory=list)  # Langfuse-style span records
    budget_spent: int = 0                 # LLM calls consumed (guardrail counter)
    tokens_used: int = 0                  # total tokens consumed
